"""Opt-in analytic-solution benchmarks for equation files.

Equation files declare their known solution after the equations:

    # Expected function: f = sqrt(x) + sqrt(y)

Use one line per output field. Expressions are parsed with a small, safe
mathematical grammar; Python eval is never used.

The training benchmarks are deliberately opt-in because every test trains a
fresh SIREN for 500 iterations. Run all 36 cases with:

    SOLVEANYTHING_RUN_BENCHMARKS=1 \
        python -m unittest discover -s tests -p "test_analytic_mse.py" -v

Select individual cases by filename stem:

    SOLVEANYTHING_RUN_BENCHMARKS=1 \
    SOLVEANYTHING_CASES=01_direct_bilinear,07_implicit_square_root \
        python -m unittest discover -s tests -p "test_analytic_mse.py" -v
"""

import ast
import contextlib
import io
import math
import operator
import os
import re
import unittest
from pathlib import Path

import numpy as np
import torch

import solveanything as solver


EQUATION_DIRECTORY = Path(__file__).with_name("equations")
EXPECTED_PATTERN = re.compile(
    r"^\s*#\s*Expected function:\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$"
)

# Keep these aligned with the solver configuration being benchmarked.
NB_ITERATIONS = 500
NB_SAMPLES = 1000
LEARNING_RATE = 1e-4
LR_GAMMA = 0.99
LOSS_WEIGHTING = "legacy"
HIDDEN_LAYERS = 4
HIDDEN_FEATURES = 256
FIRST_OMEGA_0 = 10.0
HIDDEN_OMEGA_0 = 30.0
SEED = 0
GRID_RESOLUTION = 81
MSE_TOLERANCE = 1e-2
DEVICE = "cpu"

RUN_BENCHMARKS = os.environ.get("SOLVEANYTHING_RUN_BENCHMARKS") == "1"
SELECTED_CASES = {
    case.strip()
    for case in os.environ.get("SOLVEANYTHING_CASES", "").split(",")
    if case.strip()
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
EXPECTED_FUNCTIONS = {
    "abs": np.abs,
    "cos": np.cos,
    "exp": np.exp,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
}


def parse_equation_file(path):
    """Return solver equations and expected expressions from path."""
    equations = []
    expected_functions = {}

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        expected_match = EXPECTED_PATTERN.match(raw_line)
        if expected_match:
            field, expression = expected_match.groups()
            if field in expected_functions:
                raise ValueError(
                    f"{path}:{line_number}: duplicate expected function for {field!r}"
                )
            # Validate the syntax immediately and keep the parsed tree for reuse.
            expected_functions[field] = ast.parse(expression, mode="eval").body
            continue

        stripped = raw_line.strip()
        if stripped and not stripped.startswith("#"):
            equations.append(stripped)

    if not equations:
        raise ValueError(f"{path}: no equations found")
    if not expected_functions:
        raise ValueError(f"{path}: no '# Expected function:' metadata found")

    return equations, expected_functions


def _evaluate_expected_node(node, x, y):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.Name):
        values = {"x": x, "y": y, "pi": math.pi}
        if node.id not in values:
            raise ValueError(f"unsupported name: {node.id!r}")
        return values[node.id]

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](
            _evaluate_expected_node(node.operand, x, y)
        )

    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        return BINARY_OPERATORS[type(node.op)](
            _evaluate_expected_node(node.left, x, y),
            _evaluate_expected_node(node.right, x, y),
        )

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in EXPECTED_FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        return EXPECTED_FUNCTIONS[node.func.id](
            _evaluate_expected_node(node.args[0], x, y)
        )

    raise ValueError(f"unsupported expected-function expression: {ast.dump(node)}")


def evaluate_expected_function(node, x, y):
    """Evaluate a validated expected-expression AST on NumPy coordinates."""
    value = np.asarray(_evaluate_expected_node(node, x, y), dtype=np.float64)
    if value.ndim == 0:
        value = np.full_like(x, value.item(), dtype=np.float64)
    try:
        value = np.broadcast_to(value, x.shape)
    except ValueError as error:
        raise ValueError(
            f"expected function has shape {value.shape}, not {x.shape}"
        ) from error
    if not np.isfinite(value).all():
        raise ValueError("expected function produced a non-finite value")
    return value


def train_and_measure_mse(path):
    """Train one default SIREN and return absolute grid MSE for every field."""
    equations, expected_functions = parse_equation_file(path)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with contextlib.redirect_stdout(io.StringIO()):
        variables, domains = solver.parse_equations(equations)
    if set(variables) != set(expected_functions):
        raise AssertionError(
            f"{path.name}: solver fields {variables!r} do not match expected fields "
            f"{sorted(expected_functions)!r}"
        )

    field_indices = {
        variable: index for index, variable in enumerate(variables)
    }
    model = solver.Siren(
        in_features=2,
        hidden_features=HIDDEN_FEATURES,
        hidden_layers=HIDDEN_LAYERS,
        out_features=len(variables),
        first_omega_0=FIRST_OMEGA_0,
        hidden_omega_0=HIDDEN_OMEGA_0,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=LR_GAMMA
    )

    model.train()
    for iteration in range(NB_ITERATIONS):
        optimizer.zero_grad(set_to_none=True)
        loss = solver.compute_loss(
            equations,
            domains,
            model,
            field_indices,
            NB_SAMPLES,
            DEVICE,
            loss_weighting=LOSS_WEIGHTING,
        )
        if not torch.isfinite(loss):
            raise AssertionError(
                f"{path.name}: non-finite training loss at iteration {iteration}"
            )
        loss.backward()
        optimizer.step()
        scheduler.step()

    coordinates = torch.cartesian_prod(
        torch.linspace(0.0, 1.0, GRID_RESOLUTION, device=DEVICE),
        torch.linspace(0.0, 1.0, GRID_RESOLUTION, device=DEVICE),
    )
    model.eval()
    with torch.no_grad():
        predictions = model(
            coordinates[:, 0:1], coordinates[:, 1:2]
        ).cpu().numpy()

    x = coordinates[:, 0].cpu().numpy().astype(np.float64)
    y = coordinates[:, 1].cpu().numpy().astype(np.float64)
    mse_by_field = {}
    for field, expected_node in expected_functions.items():
        expected = evaluate_expected_function(expected_node, x, y)
        predicted = predictions[:, field_indices[field]].astype(np.float64)
        mse_by_field[field] = float(np.mean((predicted - expected) ** 2))
    return mse_by_field


EQUATION_FILES = tuple(sorted(EQUATION_DIRECTORY.glob("*.txt")))


class AnalyticMetadataTests(unittest.TestCase):
    def test_all_36_equation_files_have_valid_expected_functions(self):
        self.assertEqual(len(EQUATION_FILES), 36)
        stems = {path.stem for path in EQUATION_FILES}
        unknown_cases = SELECTED_CASES - stems
        self.assertFalse(
            unknown_cases,
            f"Unknown SOLVEANYTHING_CASES: {sorted(unknown_cases)!r}",
        )

        sample_x = np.array([0.0, 0.25, 1.0])
        sample_y = np.array([1.0, 0.5, 0.0])
        for path in EQUATION_FILES:
            with self.subTest(path=path.name):
                equations, expected_functions = parse_equation_file(path)
                with contextlib.redirect_stdout(io.StringIO()):
                    variables, _ = solver.parse_equations(equations)
                self.assertEqual(set(variables), set(expected_functions))
                for expected_node in expected_functions.values():
                    values = evaluate_expected_function(
                        expected_node, sample_x, sample_y
                    )
                    self.assertEqual(values.shape, sample_x.shape)


@unittest.skipUnless(
    RUN_BENCHMARKS,
    "set SOLVEANYTHING_RUN_BENCHMARKS=1 to run training benchmarks",
)
class AnalyticMSEBenchmarkTests(unittest.TestCase):
    """One independently selectable training test per equation file."""


def _make_benchmark_test(path):
    def test(self):
        if SELECTED_CASES and path.stem not in SELECTED_CASES:
            self.skipTest("not selected by SOLVEANYTHING_CASES")

        mse_by_field = train_and_measure_mse(path)
        for field, mse in mse_by_field.items():
            print(f"{path.stem} [{field}] MSE={mse:.8g}")
            with self.subTest(field=field):
                self.assertLessEqual(
                    mse,
                    MSE_TOLERANCE,
                    f"{path.name} [{field}] MSE {mse:.8g} exceeds "
                    f"{MSE_TOLERANCE:.8g}",
                )

    return test


for _equation_path in EQUATION_FILES:
    setattr(
        AnalyticMSEBenchmarkTests,
        f"test_{_equation_path.stem}",
        _make_benchmark_test(_equation_path),
    )


if __name__ == "__main__":
    unittest.main()
