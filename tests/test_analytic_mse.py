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

import os
import unittest
from pathlib import Path

import torch

from solveanything import (
    evaluate_expected_function,
    parse_equation_file,
    parse_expected_functions,
    train_model,
)


EQUATION_DIRECTORY = Path(__file__).with_name("equations")

SEED = 0
GRID_RESOLUTION = 81
MSE_TOLERANCE = 1e-2

RUN_BENCHMARKS = 1  # os.environ.get("SOLVEANYTHING_RUN_BENCHMARKS") == "1"
SELECTED_CASES = {
    case.strip()
    for case in os.environ.get("SOLVEANYTHING_CASES", "").split(",")
    if case.strip()
}


def train_and_measure_mse(path):
    """Train one default SIREN and return absolute grid MSE for every field."""
    expected_functions = parse_expected_functions(path)

    torch.manual_seed(SEED)

    equations, variables, domains = parse_equation_file(path, verbose=False)
    if set(variables) != set(expected_functions):
        raise AssertionError(
            f"{path.name}: solver fields {variables!r} do not match expected fields "
            f"{sorted(expected_functions)!r}"
        )

    field_indices = {variable: index for index, variable in enumerate(variables)}
    model = train_model(
        equations,
        variables,
        domains,
        progress=False,
    )

    coordinates = torch.cartesian_prod(
        torch.linspace(0.0, 1.0, GRID_RESOLUTION),
        torch.linspace(0.0, 1.0, GRID_RESOLUTION),
    )
    model.eval()
    with torch.no_grad():
        predictions = model(coordinates[:, 0:1], coordinates[:, 1:2])

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    mse_by_field = {}
    for field, expected_node in expected_functions.items():
        expected = evaluate_expected_function(expected_node, x, y)
        predicted = predictions[:, field_indices[field]]
        mse_by_field[field] = torch.mean((predicted - expected) ** 2).item()
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

        sample_x = torch.tensor([0.0, 0.25, 1.0])
        sample_y = torch.tensor([1.0, 0.5, 0.0])
        for path in EQUATION_FILES:
            with self.subTest(path=path.name):
                expected_functions = parse_expected_functions(path)
                _, variables, _ = parse_equation_file(path, verbose=False)
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
