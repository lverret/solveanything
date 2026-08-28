import argparse
import ast
import math
import random
import re
import time
import numpy as np
import torch
import operator
import matplotlib.pyplot as plt
import torch.nn.functional as F

from inspect import signature
from tqdm import trange
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter


class InvalidFormula(Exception):
    def __init__(self, formula, reason, *args):
        self.message = f"'{formula}' ({reason})"
        super(InvalidFormula, self).__init__(self.message, *args)


# -----------------------------------------------------------------------------
# Neural architectures


class SineLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
        else:
            self.linear.weight.uniform_(
                -np.sqrt(6 / self.in_features) / self.omega_0,
                np.sqrt(6 / self.in_features) / self.omega_0,
            )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


MAX_ANALYTIC_DERIVATIVE_ORDER = 4


def _sine_derivatives(argument_derivatives, sine, cosine):
    """Apply ``sin`` to pure spatial derivatives through fourth order."""
    derivatives = [sine]

    if len(argument_derivatives) > 1:
        first = argument_derivatives[1]
        derivatives.append(cosine * first)
    if len(argument_derivatives) > 2:
        second = argument_derivatives[2]
        derivatives.append(cosine * second - sine * first.square())
    if len(argument_derivatives) > 3:
        third = argument_derivatives[3]
        derivatives.append(
            cosine * third
            - 3 * sine * first * second
            - cosine * first.pow(3)
        )
    if len(argument_derivatives) > 4:
        fourth = argument_derivatives[4]
        derivatives.append(
            cosine * fourth
            - 4 * sine * first * third
            - 3 * sine * second.square()
            - 6 * cosine * first.square() * second
            + sine * first.pow(4)
        )
    return derivatives


class Siren(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        first_omega_0=3.0,
        hidden_omega_0=3.0,
    ):
        super().__init__()
        self.net = [
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        ]
        for _ in range(hidden_layers):
            self.net.append(
                (
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        omega_0=hidden_omega_0,
                    )
                )
            )
        final_linear = torch.nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                np.sqrt(6 / hidden_features) / hidden_omega_0,
            )
        self.net.append(final_linear)
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))

    def forward_with_derivatives(self, x, y, derivative_orders):
        """Evaluate outputs and requested pure derivatives in one forward pass.

        SIREN derivatives can be propagated exactly through each linear and
        sine layer. This avoids constructing nested ``autograd.grad`` graphs,
        while ordinary autograd still computes parameter gradients from the
        resulting residual loss.
        """
        derivative_orders = tuple(
            dict.fromkeys(
                order for order in derivative_orders if order != (0, 0)
            )
        )
        if any(
            x_order and y_order
            or max(x_order, y_order) > MAX_ANALYTIC_DERIVATIVE_ORDER
            for x_order, y_order in derivative_orders
        ):
            raise ValueError("Only pure derivatives through fourth order are supported")

        max_x_order = max(
            (x_order for x_order, _ in derivative_orders), default=0
        )
        max_y_order = max(
            (y_order for _, y_order in derivative_orders), default=0
        )
        value = torch.cat([x, y], dim=-1)
        x_derivatives = [value]
        y_derivatives = [value]
        if max_x_order:
            x_derivatives.extend(
                [
                    torch.cat([torch.ones_like(x), torch.zeros_like(y)], dim=-1),
                    *[
                        torch.zeros_like(value)
                        for _ in range(max_x_order - 1)
                    ],
                ]
            )
        if max_y_order:
            y_derivatives.extend(
                [
                    torch.cat([torch.zeros_like(x), torch.ones_like(y)], dim=-1),
                    *[
                        torch.zeros_like(value)
                        for _ in range(max_y_order - 1)
                    ],
                ]
            )

        batch_size = value.size(0)
        for layer in self.net[:-1]:
            derivative_inputs = [
                value,
                *x_derivatives[1:],
                *y_derivatives[1:],
            ]
            transformed = F.linear(
                torch.cat(derivative_inputs, dim=0), layer.linear.weight
            ).split(batch_size, dim=0)
            argument = layer.omega_0 * (transformed[0] + layer.linear.bias)
            sine = torch.sin(argument)
            cosine = torch.cos(argument)
            offset = 1

            if max_x_order:
                x_arguments = [
                    argument,
                    *(
                        layer.omega_0 * derivative
                        for derivative in transformed[
                            offset : offset + max_x_order
                        ]
                    ),
                ]
                x_derivatives = _sine_derivatives(x_arguments, sine, cosine)
                value = x_derivatives[0]
                offset += max_x_order
            if max_y_order:
                y_arguments = [
                    argument,
                    *(
                        layer.omega_0 * derivative
                        for derivative in transformed[
                            offset : offset + max_y_order
                        ]
                    ),
                ]
                y_derivatives = _sine_derivatives(y_arguments, sine, cosine)
                value = y_derivatives[0]
            if not max_x_order and not max_y_order:
                value = sine

            x_derivatives[0] = value
            y_derivatives[0] = value

        final_layer = self.net[-1]
        requested_inputs = [value]
        for x_order, y_order in derivative_orders:
            requested_inputs.append(
                x_derivatives[x_order] if x_order else y_derivatives[y_order]
            )
        transformed = F.linear(
            torch.cat(requested_inputs, dim=0), final_layer.weight
        ).split(batch_size, dim=0)
        outputs = {(0, 0): transformed[0] + final_layer.bias}
        outputs.update(
            {
                order: derivative
                for order, derivative in zip(derivative_orders, transformed[1:])
            }
        )
        return outputs


def _activation(name):
    """Return an activation module used by the configurable model factory."""
    activations = {
        "tanh": torch.nn.Tanh,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        raise ValueError(
            f"Unknown activation {name!r}; choose from {sorted(activations)}"
        ) from error


class WeightFactorizedLinear(torch.nn.Module):
    """Linear layer with trainable row scale and direction parameters."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        initial = torch.empty(out_features, in_features)
        torch.nn.init.kaiming_uniform_(initial, a=np.sqrt(5))
        norms = initial.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.direction = torch.nn.Parameter(initial / norms)
        self.log_scale = torch.nn.Parameter(norms.log())
        if bias:
            bound = 1 / np.sqrt(in_features)
            self.bias = torch.nn.Parameter(
                torch.empty(out_features).uniform_(-bound, bound)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, value):
        weight = self.direction * self.log_scale.exp()
        return F.linear(value, weight, self.bias)


class MLP(torch.nn.Module):
    """A coordinate MLP with optional random weight factorization."""

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        activation="tanh",
        random_weight_factorization=False,
    ):
        super().__init__()
        linear = (
            WeightFactorizedLinear if random_weight_factorization else torch.nn.Linear
        )
        layers = [linear(in_features, hidden_features), _activation(activation)]
        for _ in range(max(0, hidden_layers - 1)):
            layers.extend(
                [linear(hidden_features, hidden_features), _activation(activation)]
            )
        layers.append(linear(hidden_features, out_features))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


class FourierFeatureMLP(torch.nn.Module):
    """Tanh MLP preceded by a frozen Gaussian Fourier feature map."""

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        fourier_features=64,
        fourier_sigma=2.0,
        activation="tanh",
    ):
        super().__init__()
        self.register_buffer(
            "projection",
            torch.randn(in_features, fourier_features) * float(fourier_sigma),
        )
        self.mlp = MLP(
            2 * fourier_features,
            hidden_features,
            hidden_layers,
            out_features,
            activation=activation,
        )

    def forward(self, x, y):
        coordinates = torch.cat([x, y], dim=-1)
        phase = 2 * np.pi * coordinates @ self.projection
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        midpoint = features.size(-1) // 2
        return self.mlp(features[:, :midpoint], features[:, midpoint:])


class ModifiedMLP(torch.nn.Module):
    """Gated MLP used in the PINN gradient-pathology literature."""

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        activation="tanh",
    ):
        super().__init__()
        self.activation = _activation(activation)
        self.encoder_u = torch.nn.Linear(in_features, hidden_features)
        self.encoder_v = torch.nn.Linear(in_features, hidden_features)
        self.input_layer = torch.nn.Linear(in_features, hidden_features)
        self.hidden = torch.nn.ModuleList(
            torch.nn.Linear(hidden_features, hidden_features)
            for _ in range(max(0, hidden_layers - 1))
        )
        self.output_layer = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x, y):
        coordinates = torch.cat([x, y], dim=-1)
        encoder_u = self.activation(self.encoder_u(coordinates))
        encoder_v = self.activation(self.encoder_v(coordinates))
        hidden = self.activation(self.input_layer(coordinates))
        for layer in self.hidden:
            gate = self.activation(layer(hidden))
            hidden = (1 - gate) * encoder_u + gate * encoder_v
        return self.output_layer(hidden)


class PirateNet(torch.nn.Module):
    """Compact physics-informed residual network with zero-initialized blocks."""

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        activation="tanh",
    ):
        super().__init__()
        self.activation = _activation(activation)
        self.input_layer = torch.nn.Linear(in_features, hidden_features)
        self.blocks = torch.nn.ModuleList()
        self.residual_scales = torch.nn.ParameterList()
        for _ in range(hidden_layers):
            self.blocks.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_features, hidden_features),
                    _activation(activation),
                    torch.nn.Linear(hidden_features, hidden_features),
                    _activation(activation),
                )
            )
            self.residual_scales.append(torch.nn.Parameter(torch.zeros(())))
        self.output_layer = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x, y):
        hidden = self.activation(self.input_layer(torch.cat([x, y], dim=-1)))
        for block, scale in zip(self.blocks, self.residual_scales):
            hidden = hidden + scale * block(hidden)
        return self.output_layer(hidden)


class MultiHeadMLP(torch.nn.Module):
    """Coordinate MLP with a shared trunk and one scalar head per field."""

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        activation="tanh",
    ):
        super().__init__()
        layers = [
            torch.nn.Linear(in_features, hidden_features),
            _activation(activation),
        ]
        for _ in range(max(0, hidden_layers - 1)):
            layers.extend(
                [
                    torch.nn.Linear(hidden_features, hidden_features),
                    _activation(activation),
                ]
            )
        self.trunk = torch.nn.Sequential(*layers)
        self.heads = torch.nn.ModuleList(
            torch.nn.Linear(hidden_features, 1) for _ in range(out_features)
        )

    def forward(self, x, y):
        hidden = self.trunk(torch.cat([x, y], dim=-1))
        return torch.cat([head(hidden) for head in self.heads], dim=-1)


def build_model(
    architecture,
    in_features,
    hidden_features,
    hidden_layers,
    out_features,
    first_omega_0=10.0,
    hidden_omega_0=30.0,
    activation="tanh",
    fourier_features=64,
    fourier_sigma=2.0,
):
    """Build one of the coordinate-network backbones used by experiments."""
    architecture = architecture.lower()
    common = dict(
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
    )
    if architecture == "siren":
        model = Siren(
            **common,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )
    elif architecture == "mlp":
        model = MLP(**common, activation=activation)
    elif architecture == "rwf_mlp":
        model = MLP(
            **common,
            activation=activation,
            random_weight_factorization=True,
        )
    elif architecture == "fourier_mlp":
        model = FourierFeatureMLP(
            **common,
            activation=activation,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
        )
    elif architecture == "modified_mlp":
        model = ModifiedMLP(**common, activation=activation)
    elif architecture == "piratenet":
        model = PirateNet(**common, activation=activation)
    elif architecture == "multihead_mlp":
        model = MultiHeadMLP(**common, activation=activation)
    else:
        raise ValueError(
            "Unknown architecture {!r}; choose siren, mlp, rwf_mlp, "
            "fourier_mlp, modified_mlp, piratenet, or multihead_mlp".format(
                architecture
            )
        )
    return model


# -----------------------------------------------------------------------------
# Helper mathematical functions for defining the equations to solve


def sqrt(y):
    return torch.sqrt(torch.as_tensor(y))


def sin(y):
    return torch.sin(torch.as_tensor(y))


def cos(y):
    return torch.cos(torch.as_tensor(y))


def exp(y):
    return torch.exp(torch.as_tensor(y))


def abs(y):
    return torch.abs(torch.as_tensor(y))


def tanh(y):
    return torch.tanh(torch.as_tensor(y))


def image(path, x, y):
    image = np.array(Image.open(path).convert("L"))
    image = torch.tensor(image).rot90(-1) / 255
    image = image.to(x.device)
    px, py = ((x * image.size(0)).long(), (y * image.size(1)).long())
    return image[px, py]


def grad(y, x):
    return torch.autograd.grad(
        y, [x], grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


MATH_FUNS = {
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "exp": exp,
    "abs": abs,
    "tanh": tanh,
}


FUNS = {
    **MATH_FUNS,
    "image": image,
    "grad": grad,
}


CONSTANTS = {"pi": np.pi}
SECTION_PATTERN = re.compile(r"^\s*#\s*(Equations|Solution)\s*$", re.IGNORECASE)

# -----------------------------------------------------------------------------
# Parser functions


def parse_equations(equations, verbose=True):
    variables = {}
    domains = []
    for formula in equations:
        splits = formula.split("=")
        if len(splits) != 2:
            raise InvalidFormula(formula, "Not a equation")
        lhs, rhs = splits
        fixed_coordinates = {"x": set(), "y": set()}
        parse(
            formula,
            ast.parse(lhs.strip(), mode="eval").body,
            variables,
            fixed_coordinates,
        )
        parse(
            formula,
            ast.parse(rhs.strip(), mode="eval").body,
            variables,
            fixed_coordinates,
        )
        domain = {
            coordinate: next(iter(values)) if len(values) == 1 else np.nan
            for coordinate, values in fixed_coordinates.items()
        }
        domains.append(domain)
        log = f"Parsed equation {len(domains)}: 'for "
        for inp in ["x", "y"]:
            if np.isnan(domain[inp]):
                log += f"{inp} in (0, 1), "
            else:
                log += f"{inp} = {domain[inp]}, "
        if verbose:
            print(log[:-2] + f",  {formula}'")
    variables = list(variables.keys())
    if verbose:
        print(f"Found {len(variables)} unknown function(s) to approximate: {variables}")
    return variables, domains


def _read_problem_sections(input_file):
    sections = {"equations": [], "solution": []}
    seen_sections = set()
    current_section = None

    with open(input_file, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            section_match = SECTION_PATTERN.match(raw_line)
            if section_match is not None:
                section = section_match.group(1).lower()
                if section in seen_sections:
                    raise ValueError(
                        f"{input_file}:{line_number}: duplicate "
                        f"'#{section_match.group(1)}' section"
                    )
                if section == "solution" and "equations" not in seen_sections:
                    raise ValueError(
                        f"{input_file}:{line_number}: '# Solution' must follow "
                        "'# Equations'"
                    )
                seen_sections.add(section)
                current_section = section
                continue

            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if current_section is None:
                raise ValueError(
                    f"{input_file}:{line_number}: content must be placed below "
                    "'# Equations' or '# Solution'"
                )
            sections[current_section].append((line_number, line))

    if "equations" not in seen_sections:
        raise ValueError(f"{input_file}: missing '# Equations' section")
    if not sections["equations"]:
        raise ValueError(f"{input_file}: '# Equations' section is empty")
    if "solution" in seen_sections and not sections["solution"]:
        raise ValueError(f"{input_file}: '# Solution' section is empty")

    return sections, "solution" in seen_sections


def _parse_solution_equations(solution_equations, input_file="<solution>"):
    """Parse ``field = expression`` formulas from a ``# Solution`` section."""
    solution_functions = {}
    for line_number, formula in solution_equations:
        splits = formula.split("=")
        if len(splits) != 2:
            raise ValueError(
                f"{input_file}:{line_number}: solution must have the form "
                "'field = expression'"
            )
        lhs, rhs = (part.strip() for part in splits)
        lhs_node = ast.parse(lhs, filename=str(input_file), mode="eval").body
        if not isinstance(lhs_node, ast.Name):
            raise ValueError(
                f"{input_file}:{line_number}: solution field must be a name"
            )
        field = lhs_node.id
        if field in solution_functions:
            raise ValueError(
                f"{input_file}:{line_number}: duplicate solution for {field!r}"
            )

        node = ast.parse(rhs, filename=str(input_file), mode="eval").body
        parse(
            formula,
            node,
            variables={},
            fixed_coordinates={"x": set(), "y": set()},
            functions=MATH_FUNS,
            allow_unknown_functions=False,
        )
        solution_functions[field] = node
    return solution_functions


def parse_problem_file(input_file, verbose=True):
    """Parse equations and an optional analytic solution from a problem file."""
    sections, has_solution = _read_problem_sections(input_file)
    equations = [formula for _, formula in sections["equations"]]
    variables, domains = parse_equations(equations, verbose=verbose)
    solution_functions = _parse_solution_equations(
        sections["solution"], input_file=input_file
    )

    if has_solution and set(solution_functions) != set(variables):
        missing = sorted(set(variables) - set(solution_functions))
        unknown = sorted(set(solution_functions) - set(variables))
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise ValueError(
            f"{input_file}: '# Solution' does not match the equations "
            f"({', '.join(details)})"
        )

    return equations, variables, domains, solution_functions


def parse_equation_file(input_file, verbose=True):
    """Compatibility wrapper returning equations, variables and domains."""
    equations, variables, domains, _ = parse_problem_file(
        input_file, verbose=verbose
    )
    return equations, variables, domains


def parse(
    formula,
    node,
    variables,
    fixed_coordinates,
    functions=None,
    allow_unknown_functions=True,
):
    if functions is None:
        functions = FUNS

    def parse_child(child):
        return parse(
            formula,
            child,
            variables,
            fixed_coordinates,
            functions,
            allow_unknown_functions,
        )

    if isinstance(node, ast.Constant):
        return variables
    elif isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return parse_child(node.operand)
    elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
        parse_child(node.left)
        parse_child(node.right)
        return variables
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in functions:
            if len(node.args) != len(signature(functions[node.func.id]).parameters):
                raise InvalidFormula(
                    formula, f"Invalid nb of args for '{node.func.id}'"
                )
            for arg in node.args:
                parse_child(arg)
            return variables
        elif isinstance(node.func, ast.Name) and allow_unknown_functions:
            if not all(isinstance(arg, (ast.Constant, ast.Name)) for arg in node.args):
                raise InvalidFormula(formula, f"Found invalid arg for '{node.func.id}'")
            if len(node.args) != 2:
                raise InvalidFormula(
                    formula, f"Invalid nb of args for '{node.func.id}'"
                )
            if (
                isinstance(node.args[0], ast.Name)
                and node.args[0].id != "x"
                or isinstance(node.args[1], ast.Name)
                and node.args[1].id != "y"
            ):
                raise InvalidFormula(
                    formula, f"'{node.func.id}' takes as args only (x, y) in that order"
                )
            for inp, arg in zip(["x", "y"], node.args):
                if isinstance(arg, ast.Constant):
                    value = arg.value
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise InvalidFormula(
                            formula, f"Found invalid arg for '{node.func.id}'"
                        )
                    if not 0 <= value <= 1:
                        raise InvalidFormula(
                            formula, "Only functions in [0, 1] x [0, 1] are supported"
                        )
                    fixed_coordinates[inp].add(float(value))
            variables[node.func.id] = None
            return variables
    elif isinstance(node, ast.Name):
        if node.id in ["x", "y"] or node.id in CONSTANTS:
            return variables
        elif allow_unknown_functions:
            parse_child(
                ast.Call(
                    func=ast.Name(id=node.id, ctx=ast.Load()),
                    args=[
                        ast.Name(id="x", ctx=ast.Load()),
                        ast.Name(id="y", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            )
            return variables
    raise InvalidFormula(formula, "Found unsupported token(s)")


def _record_coordinate(used_coordinates, coordinate, value):
    values = used_coordinates[coordinate]
    if all(existing is not value for existing in values):
        values.append(value)


def _coordinate_value(node, coordinate, samples, coordinate_cache, used_coordinates):
    if isinstance(node, ast.Name):
        value = samples[coordinate]
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        key = (coordinate, float(node.value))
        if key not in coordinate_cache:
            coordinate_cache[key] = torch.full_like(
                samples[coordinate],
                float(node.value),
                requires_grad=samples[coordinate].requires_grad,
            )
        value = coordinate_cache[key]
    else:
        raise RuntimeError(f"Invalid coordinate argument: {ast.dump(node)}")
    _record_coordinate(used_coordinates, coordinate, value)
    return value


def _merge_coordinates(source, destination):
    for coordinate in ["x", "y"]:
        for value in source[coordinate]:
            _record_coordinate(destination, coordinate, value)


def _coordinate_signature(coordinate_nodes):
    return tuple(
        ast.dump(node, annotate_fields=False, include_attributes=False)
        for node in coordinate_nodes
    )


def _direct_field_derivative(node, field_indices):
    """Return a direct field derivative without evaluating its AST."""
    derivative_order = [0, 0]
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "grad"
    ):
        coordinate_node = node.args[1]
        if not isinstance(coordinate_node, ast.Name) or coordinate_node.id not in (
            "x",
            "y",
        ):
            return None
        coordinate_index = 0 if coordinate_node.id == "x" else 1
        derivative_order[coordinate_index] += 1
        node = node.args[0]

    if not any(derivative_order):
        return None
    if isinstance(node, ast.Name) and node.id in field_indices:
        field = node.id
        coordinate_nodes = (
            ast.Name(id="x", ctx=ast.Load()),
            ast.Name(id="y", ctx=ast.Load()),
        )
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in field_indices
        and len(node.args) == 2
    ):
        field = node.func.id
        coordinate_nodes = tuple(node.args)
    else:
        return None

    return field, coordinate_nodes, tuple(derivative_order)


def _supports_analytic_derivative(derivative_order):
    x_order, y_order = derivative_order
    return (
        not (x_order and y_order)
        and max(derivative_order) <= MAX_ANALYTIC_DERIVATIVE_ORDER
    )


def _collect_derivative_requests(node, field_indices):
    requests = {}

    def visit(child):
        derivative = _direct_field_derivative(child, field_indices)
        if derivative is not None and _supports_analytic_derivative(derivative[2]):
            _, coordinate_nodes, derivative_order = derivative
            signature = _coordinate_signature(coordinate_nodes)
            requests.setdefault(signature, set()).add(derivative_order)
            return
        for grandchild in ast.iter_child_nodes(child):
            visit(grandchild)

    visit(node)
    return {
        signature: tuple(sorted(orders))
        for signature, orders in requests.items()
    }


def evaluate(
    node,
    samples,
    model,
    field_indices,
    coordinate_cache=None,
    used_coordinates=None,
    evaluation_cache=None,
):
    if coordinate_cache is None:
        coordinate_cache = {}
    if used_coordinates is None:
        used_coordinates = {"x": [], "y": []}
    if evaluation_cache is None:
        evaluation_cache = {
            "model_outputs": {},
            "derivative_outputs": {},
            "derivative_requests": {},
        }
    model_cache = evaluation_cache["model_outputs"]
    derivative_cache = evaluation_cache["derivative_outputs"]
    derivative_requests = evaluation_cache["derivative_requests"]

    direct_derivative = _direct_field_derivative(node, field_indices)
    if (
        direct_derivative is not None
        and _supports_analytic_derivative(direct_derivative[2])
        and hasattr(model, "forward_with_derivatives")
    ):
        field, coordinate_nodes, derivative_order = direct_derivative
        coordinates = [
            _coordinate_value(
                coordinate_node,
                coordinate,
                samples,
                coordinate_cache,
                used_coordinates,
            )
            for coordinate, coordinate_node in zip(["x", "y"], coordinate_nodes)
        ]
        signature = _coordinate_signature(coordinate_nodes)
        requested_orders = derivative_requests.get(
            signature, (derivative_order,)
        )
        cache_key = tuple(coordinates)
        if cache_key not in derivative_cache:
            derivative_cache[cache_key] = model.forward_with_derivatives(
                *coordinates, requested_orders
            )
            model_cache[cache_key] = derivative_cache[cache_key][(0, 0)]
        field_index = field_indices[field]
        return derivative_cache[cache_key][derivative_order][
            :, field_index : field_index + 1
        ]

    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](
            evaluate(
                node.operand,
                samples,
                model,
                field_indices,
                coordinate_cache,
                used_coordinates,
                evaluation_cache,
            )
        )
    elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](
            evaluate(
                node.left,
                samples,
                model,
                field_indices,
                coordinate_cache,
                used_coordinates,
                evaluation_cache,
            ),
            evaluate(
                node.right,
                samples,
                model,
                field_indices,
                coordinate_cache,
                used_coordinates,
                evaluation_cache,
            ),
        )
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function_name = node.func.id
        if function_name == "grad":
            coordinate_node = node.args[1]
            if not isinstance(coordinate_node, ast.Name) or coordinate_node.id not in [
                "x",
                "y",
            ]:
                raise RuntimeError("grad expects x or y as its second argument")
            coordinate = coordinate_node.id
            local_coordinates = {"x": [], "y": []}
            value = evaluate(
                node.args[0],
                samples,
                model,
                field_indices,
                coordinate_cache,
                local_coordinates,
                evaluation_cache,
            )
            targets = local_coordinates[coordinate]
            if not targets:
                targets = [samples[coordinate]]
                _record_coordinate(local_coordinates, coordinate, targets[0])
            result = sum(
                (target * 0 for target in targets), torch.zeros_like(targets[0])
            )
            if torch.is_tensor(value) and value.requires_grad:
                derivatives = torch.autograd.grad(
                    value,
                    targets,
                    grad_outputs=torch.ones_like(value),
                    create_graph=True,
                    allow_unused=True,
                )
                for derivative in derivatives:
                    if derivative is not None:
                        result = result + derivative
            _merge_coordinates(local_coordinates, used_coordinates)
            return result
        elif function_name in FUNS:
            return FUNS[function_name](
                *(
                    evaluate(
                        arg,
                        samples,
                        model,
                        field_indices,
                        coordinate_cache,
                        used_coordinates,
                        evaluation_cache,
                    )
                    for arg in node.args
                )
            )
        elif function_name in field_indices:
            coordinates = [
                _coordinate_value(
                    arg,
                    coordinate,
                    samples,
                    coordinate_cache,
                    used_coordinates,
                )
                for coordinate, arg in zip(["x", "y"], node.args)
            ]
            field_index = field_indices[function_name]
            cache_key = tuple(coordinates)
            signature = _coordinate_signature(node.args)
            requested_orders = derivative_requests.get(signature)
            if (
                requested_orders
                and cache_key not in derivative_cache
                and hasattr(model, "forward_with_derivatives")
            ):
                derivative_cache[cache_key] = model.forward_with_derivatives(
                    *coordinates, requested_orders
                )
                model_cache[cache_key] = derivative_cache[cache_key][(0, 0)]
            if cache_key not in model_cache:
                model_cache[cache_key] = model(*coordinates)
            return model_cache[cache_key][:, field_index : field_index + 1]
    elif isinstance(node, ast.Name):
        if node.id in samples:
            if node.id in ["x", "y"]:
                _record_coordinate(used_coordinates, node.id, samples[node.id])
            return samples[node.id]
        elif node.id in CONSTANTS:
            return CONSTANTS[node.id]
        elif node.id in field_indices:
            return evaluate(
                ast.Call(
                    func=ast.Name(id=node.id, ctx=ast.Load()),
                    args=[
                        ast.Name(id="x", ctx=ast.Load()),
                        ast.Name(id="y", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
                samples,
                model,
                field_indices,
                coordinate_cache,
                used_coordinates,
                evaluation_cache,
            )
    raise RuntimeError(f"Unsupported expression: {ast.dump(node)}")


def evaluate_expected_function(node, x, y):
    """Evaluate one parsed solution expression with the shared Torch evaluator."""
    value = evaluate(
        node,
        samples={"x": x, "y": y},
        model=None,
        field_indices={},
    )
    value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    if value.ndim == 0:
        value = value.expand_as(x)
    try:
        value = torch.broadcast_to(value, x.shape)
    except RuntimeError as error:
        raise ValueError(
            f"Expected function has shape {tuple(value.shape)}, not {tuple(x.shape)}"
        ) from error
    if not torch.isfinite(value).all():
        raise ValueError("Expected function produced a non-finite value")
    return value


def evaluate_solution_functions(solution_functions, variables, x, y):
    """Evaluate solution fields in the same order as the model outputs."""
    if set(solution_functions) != set(variables):
        raise ValueError("Solution fields do not match model output fields")
    return torch.stack(
        [
            evaluate_expected_function(solution_functions[variable], x, y)
            for variable in variables
        ],
        dim=-1,
    )


# -----------------------------------------------------------------------------
# Training and visualization functions


def _domain_key(domain):
    return tuple(
        None if np.isnan(domain[coordinate]) else float(domain[coordinate])
        for coordinate in ("x", "y")
    )


class CollocationSampler:
    """Reusable IID, scrambled Sobol, fixed Sobol, or wall-mixture sampler."""

    def __init__(self, method="iid", seed=0, wall_fraction=0.5, wall_width=0.1):
        if method not in {"iid", "sobol", "fixed_sobol", "wall_mixture"}:
            raise ValueError(f"Unknown sampling method {method!r}")
        self.method = method
        self.seed = int(seed)
        self.wall_fraction = float(wall_fraction)
        self.wall_width = float(wall_width)
        self._engines = {}
        self._fixed_samples = {}

    def _unit_samples(self, domain, nb_samples):
        key = _domain_key(domain)
        free_coordinates = sum(value is None for value in key)
        if not free_coordinates:
            return torch.empty(nb_samples, 0)
        if self.method == "iid":
            engine_key = ("iid", key)
            generator = self._engines.setdefault(
                engine_key,
                torch.Generator().manual_seed(self.seed + len(self._engines)),
            )
            return torch.rand(nb_samples, free_coordinates, generator=generator)
        engine_key = ("sobol", key)
        engine = self._engines.setdefault(
            engine_key,
            torch.quasirandom.SobolEngine(
                free_coordinates,
                scramble=True,
                seed=self.seed + len(self._engines),
            ),
        )
        return engine.draw(nb_samples)

    def sample(self, domain, nb_samples, device, requires_grad=True):
        key = (_domain_key(domain), int(nb_samples))
        if self.method == "fixed_sobol" and key in self._fixed_samples:
            unit = self._fixed_samples[key].clone()
        else:
            unit = self._unit_samples(domain, nb_samples)
            if self.method == "fixed_sobol":
                self._fixed_samples[key] = unit.clone()

        if self.method == "wall_mixture" and all(
            np.isnan(domain[coordinate]) for coordinate in ("x", "y")
        ):
            wall_count = min(nb_samples, round(nb_samples * self.wall_fraction))
            if wall_count:
                generator_key = ("wall", _domain_key(domain))
                generator = self._engines.setdefault(
                    generator_key, torch.Generator().manual_seed(self.seed + 7919)
                )
                axis = torch.randint(0, 2, (wall_count,), generator=generator)
                side = torch.randint(0, 2, (wall_count,), generator=generator)
                distance = (
                    torch.rand(wall_count, generator=generator).square()
                    * self.wall_width
                )
                rows = torch.arange(wall_count)
                unit[rows, axis] = torch.where(side.bool(), 1 - distance, distance)

        unit = unit.to(device=device)
        samples = {}
        free_index = 0
        for coordinate in ("x", "y"):
            if np.isnan(domain[coordinate]):
                value = unit[:, free_index : free_index + 1]
                free_index += 1
            else:
                value = torch.full(
                    (nb_samples, 1), float(domain[coordinate]), device=device
                )
            samples[coordinate] = value.detach().requires_grad_(requires_grad)
        return samples


def generate_samples(domain, nb_samples, device, requires_grad=True, sampler=None):
    if sampler is not None:
        return sampler.sample(domain, nb_samples, device, requires_grad)
    samples = {"x": None, "y": None}
    for inp in ["x", "y"]:
        if np.isnan(domain[inp]):
            samples[inp] = torch.rand(nb_samples, 1, device=device)
        else:
            samples[inp] = torch.full(
                (nb_samples, 1), float(domain[inp]), device=device
            )
        samples[inp].requires_grad_(requires_grad)
    return samples


def _requires_autograd_derivatives(node, field_indices, analytic_derivatives):
    derivative = _direct_field_derivative(node, field_indices)
    if (
        analytic_derivatives
        and derivative is not None
        and _supports_analytic_derivative(derivative[2])
    ):
        return False
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "grad"
    ):
        return True
    return any(
        _requires_autograd_derivatives(child, field_indices, analytic_derivatives)
        for child in ast.iter_child_nodes(node)
    )


def compile_residuals(equations, field_indices, model):
    """Parse equations once and annotate their derivative requirements."""
    analytic_derivatives = hasattr(model, "forward_with_derivatives")
    residuals = []
    for formula in equations:
        lhs, rhs = formula.split("=")
        node = ast.parse(f"{lhs} - ({rhs})".strip(), mode="eval").body
        residuals.append(
            (
                node,
                _collect_derivative_requests(node, field_indices),
                _requires_autograd_derivatives(
                    node, field_indices, analytic_derivatives
                ),
            )
        )
    return residuals


def _compile_residuals(equations, field_indices, model):
    """Backward-compatible alias for :func:`compile_residuals`."""
    return compile_residuals(equations, field_indices, model)


def _merged_derivative_requests(residuals):
    requests = {}
    for _, derivative_requests, _ in residuals:
        for coordinate_signature, orders in derivative_requests.items():
            requests.setdefault(coordinate_signature, set()).update(orders)
    return {
        coordinate_signature: tuple(sorted(orders))
        for coordinate_signature, orders in requests.items()
    }


def evaluate_residuals(
    residuals,
    domains,
    model,
    field_indices,
    nb_samples,
    device,
    sampler=None,
    grouped=True,
):
    """Evaluate equation residuals, optionally sharing samples/model passes.

    The returned list contains ``(residual_tensor, samples)`` pairs in equation
    order. Equations on the same geometric domain share both a collocation set
    and an evaluation cache when ``grouped`` is true.
    """
    if len(residuals) != len(domains):
        raise ValueError("Residual and domain counts do not match")
    if isinstance(nb_samples, int):
        sample_counts = [nb_samples] * len(residuals)
    else:
        sample_counts = list(nb_samples)
        if len(sample_counts) != len(residuals):
            raise ValueError("Sample-count and residual counts do not match")
    if grouped:
        grouped_indices = {}
        for index, domain in enumerate(domains):
            grouped_indices.setdefault(_domain_key(domain), []).append(index)
        groups = list(grouped_indices.values())
    else:
        groups = [[index] for index in range(len(residuals))]

    evaluations = [None] * len(residuals)
    for indices in groups:
        group_residuals = [residuals[index] for index in indices]
        group_counts = {int(sample_counts[index]) for index in indices}
        if len(group_counts) != 1:
            raise ValueError(
                "Equations grouped on one domain must use the same sample count"
            )
        group_sample_count = group_counts.pop()
        requires_grad = any(item[2] for item in group_residuals)
        samples = generate_samples(
            domains[indices[0]],
            group_sample_count,
            device,
            requires_grad=requires_grad,
            sampler=sampler,
        )
        evaluation_cache = {
            "model_outputs": {},
            "derivative_outputs": {},
            "derivative_requests": _merged_derivative_requests(group_residuals),
        }
        coordinate_cache = {}
        for index in indices:
            node = residuals[index][0]
            value = evaluate(
                node,
                samples,
                model,
                field_indices,
                coordinate_cache=coordinate_cache,
                evaluation_cache=evaluation_cache,
            )
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32, device=device)
                value = value.expand(group_sample_count, 1)
            evaluations[index] = (value, samples)
    return evaluations


def reduce_residual(residual, penalty="mae", huber_delta=0.01, weights=None):
    """Reduce a pointwise residual using a PINN training penalty."""
    if penalty == "mae":
        values = residual.abs()
    elif penalty == "mse":
        values = residual.square()
    elif penalty in {"huber", "pseudo_huber"}:
        delta = float(huber_delta)
        values = delta**2 * (torch.sqrt(1 + (residual / delta).square()) - 1)
    else:
        raise ValueError(f"Unknown residual penalty {penalty!r}")
    if weights is not None:
        values = values * weights
    return values.mean()


def compute_equation_losses(
    residuals,
    domains,
    model,
    field_indices,
    nb_samples,
    device,
    sampler=None,
    grouped=True,
    penalty="mae",
    huber_delta=0.01,
):
    """Return one reduced loss and one raw evaluation per equation."""
    evaluations = evaluate_residuals(
        residuals,
        domains,
        model,
        field_indices,
        nb_samples,
        device,
        sampler=sampler,
        grouped=grouped,
    )
    losses = [
        reduce_residual(value, penalty=penalty, huber_delta=huber_delta)
        for value, _ in evaluations
    ]
    return losses, evaluations


def compute_loss(
    equations,
    domains,
    model,
    field_indices,
    nb_samples,
    device,
    residuals=None,
):
    if residuals is None:
        residuals = _compile_residuals(equations, field_indices, model)
    losses = []
    for (
        node,
        derivative_requests,
        requires_coordinate_gradients,
    ), domain in zip(residuals, domains):
        samples = generate_samples(
            domain,
            nb_samples,
            device,
            requires_grad=requires_coordinate_gradients,
        )
        res = evaluate(
            node,
            samples,
            model,
            field_indices,
            evaluation_cache={
                "model_outputs": {},
                "derivative_outputs": {},
                "derivative_requests": derivative_requests,
            },
        )
        equation_loss = torch.mean(torch.abs(res))
        weight = 0.1 if np.isnan(domain["x"]) and np.isnan(domain["y"]) else 0.9
        losses.append(weight * equation_loss)
    return torch.stack(losses).mean()


def _is_interior_domain(domain):
    return np.isnan(domain["x"]) and np.isnan(domain["y"])


def _training_progress(iteration, started, nb_iter, max_seconds):
    step_progress = iteration / max(1, nb_iter)
    if max_seconds is None:
        return min(1.0, step_progress)
    time_progress = (time.monotonic() - started) / max(1e-6, max_seconds)
    return min(1.0, max(step_progress, time_progress))


def _combine_training_losses(equation_losses, domains, loss_balance, progress):
    boundary_losses = [
        loss
        for loss, domain in zip(equation_losses, domains)
        if not _is_interior_domain(domain)
    ]
    interior_losses = [
        loss
        for loss, domain in zip(equation_losses, domains)
        if _is_interior_domain(domain)
    ]

    # Direct-function examples can contain only full-domain equations. Keep the
    # grouped loss useful beyond boundary-value PDEs by accepting either group.
    if not boundary_losses:
        return torch.stack(interior_losses).mean()
    if not interior_losses:
        return torch.stack(boundary_losses).mean()

    boundary = torch.stack(boundary_losses).mean()
    interior = torch.stack(interior_losses).mean()
    if loss_balance == "equal_groups":
        return 0.5 * (boundary + interior)
    if loss_balance == "boundary_pde_ramp":
        pde_weight = min(1.0, 0.1 + 3.0 * progress)
        return (boundary + pde_weight * interior) / (1.0 + pde_weight)
    if loss_balance == "legacy":
        weighted = [
            (0.1 if _is_interior_domain(domain) else 0.9) * loss
            for loss, domain in zip(equation_losses, domains)
        ]
        return torch.stack(weighted).mean()
    raise ValueError(f"Unknown loss balance {loss_balance!r}")


def train_model(
    equations,
    variables,
    domains,
    nb_iter=50000,
    nb_samples=2048,
    lr=0.0003,
    min_lr=0.00001,
    lr_schedule="cosine",
    lr_gamma=0.99,
    hidden_layers=4,
    hidden_features=256,
    first_omega_0=3.0,
    hidden_omega_0=3.0,
    sampling="iid",
    grouped=True,
    penalty="mse",
    loss_balance="boundary_pde_ramp",
    seed=0,
    max_seconds=180.0,
    device="cpu",
    progress=True,
    iteration_callback=None,
):
    """Train and return a low-frequency SIREN for the parsed variables.

    ``iteration_callback`` receives ``(iteration, model, loss_value)`` after
    each optimizer step. It is used by the CLI for optional frame collection
    without coupling the reusable training loop to GIF generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    field_indices = {
        variable: index for index, variable in enumerate(variables)
    }
    model = Siren(
        in_features=2,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=len(variables),
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
    ).to(device)
    residuals = _compile_residuals(equations, field_indices, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_schedule not in {"cosine", "exponential", "constant"}:
        raise ValueError(f"Unknown learning-rate schedule {lr_schedule!r}")
    scheduler = None
    if lr_schedule == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_gamma
        )
    sampler = CollocationSampler(method=sampling, seed=seed)
    iterations = (
        trange(nb_iter, desc="Solving equation(s)") if progress else range(nb_iter)
    )
    started = time.monotonic()
    deadline = None
    if max_seconds is not None:
        if max_seconds <= 4:
            raise ValueError("max_seconds must be greater than 4 seconds")
        # Match the cavity benchmark: leave four seconds for validation and
        # process/GIF bookkeeping inside the advertised runtime budget.
        deadline = started + max_seconds - 4.0

    model.train()
    for iteration in iterations:
        if deadline is not None and time.monotonic() >= deadline:
            break
        training_progress = _training_progress(
            iteration, started, nb_iter, max_seconds
        )
        optimizer.zero_grad(set_to_none=True)
        sample_counts = [nb_samples] * len(equations)
        for index, domain in enumerate(domains):
            if not np.isnan(domain["x"]) and not np.isnan(domain["y"]):
                sample_counts[index] = 1
        equation_losses, _ = compute_equation_losses(
            residuals,
            domains,
            model,
            field_indices,
            sample_counts,
            device,
            sampler=sampler,
            grouped=grouped,
            penalty=penalty,
        )
        loss = _combine_training_losses(
            equation_losses,
            domains,
            loss_balance=loss_balance,
            progress=training_progress,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at iteration {iteration}"
            )
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        elif lr_schedule == "cosine":
            current_lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * training_progress)
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr

        loss_value = loss.item()
        if progress:
            iterations.set_postfix(
                loss=loss_value,
                lr=optimizer.param_groups[0]["lr"],
            )
        if iteration_callback is not None:
            iteration_callback(iteration, model, loss_value)

    return model


def _plot_limits(minimum, maximum):
    if minimum == maximum:
        padding = max(0.5, abs(minimum) * 0.05)
        return minimum - padding, maximum + padding
    return minimum, maximum


def make_gif(frames, variables, output_file, solution=None):
    """Animate approximations and optionally show analytic solutions below them."""
    if not frames:
        raise ValueError("Cannot generate a GIF without frames")

    frames = [np.asarray(frame) for frame in frames]
    if solution is not None:
        solution = np.asarray(solution)
        if solution.shape != frames[0].shape:
            raise ValueError(
                f"Solution grid has shape {solution.shape}, expected "
                f"{frames[0].shape}"
            )

    row_count = 2 if solution is not None else 1
    column_count = len(variables)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        squeeze=False,
        figsize=(4.8 * column_count, 4.0 * row_count),
    )
    approximation_images = []

    for field_index, variable in enumerate(variables):
        if solution is not None:
            minimum = min(
                solution[:, :, field_index].min(),
                *(frame[:, :, field_index].min() for frame in frames),
            )
            maximum = max(
                solution[:, :, field_index].max(),
                *(frame[:, :, field_index].max() for frame in frames),
            )
            field_limits = _plot_limits(minimum, maximum)
        else:
            field_limits = _plot_limits(
                frames[0][:, :, field_index].min(),
                frames[0][:, :, field_index].max(),
            )

        approximation_axis = axes[0, field_index]
        approximation_image = approximation_axis.imshow(
            frames[0][:, :, field_index],
            extent=(0, 1, 0, 1),
            vmin=field_limits[0],
            vmax=field_limits[1],
        )
        approximation_images.append(approximation_image)
        fig.colorbar(approximation_image, ax=approximation_axis)
        approximation_axis.set_title(f"Approximation: {variable}")
        approximation_axis.set_xlabel("x")
        approximation_axis.set_ylabel("y")
        approximation_axis.margins(0)

        if solution is not None:
            solution_axis = axes[1, field_index]
            solution_image = solution_axis.imshow(
                solution[:, :, field_index],
                extent=(0, 1, 0, 1),
                vmin=field_limits[0],
                vmax=field_limits[1],
            )
            fig.colorbar(solution_image, ax=solution_axis)
            solution_axis.set_title(f"Solution: {variable}")
            solution_axis.set_xlabel("x")
            solution_axis.set_ylabel("y")
            solution_axis.margins(0)

    fig.tight_layout()

    def animate(frame_index):
        out = frames[frame_index]
        for field_index, image_artist in enumerate(approximation_images):
            z = out[:, :, field_index]
            image_artist.set_data(z)
            if solution is None:
                image_artist.set_clim(*_plot_limits(z.min(), z.max()))
        return approximation_images

    ani = FuncAnimation(fig, animate, frames=len(frames))
    pbar = trange(len(frames), desc="Generating GIF")
    ani.save(
        output_file,
        writer=PillowWriter(fps=len(frames) / 3),
        progress_callback=lambda i, n: pbar.update(1),
    )
    pbar.close()
    plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="problem file containing '# Equations' and optional '# Solution' sections",
    )
    parser.add_argument(
        "--output_file", "-o", type=str, default="out.gif", help="output gif filename"
    )
    parser.add_argument(
        "--nb_iter",
        type=int,
        default=50000,
        help="maximum optimizer steps (default: 50000)",
    )
    parser.add_argument(
        "--nb_samples",
        type=int,
        default=2048,
        help="samples per geometric domain (default: 2048)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003, help="peak learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.00001, help="cosine-decay floor"
    )
    parser.add_argument(
        "--lr_schedule",
        choices=("cosine", "exponential", "constant"),
        default="cosine",
        help="learning-rate schedule (default: cosine)",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.99,
        help="per-iteration exponential LR factor",
    )
    parser.add_argument(
        "--hidden_layers", type=int, default=4, help="number of hidden layers"
    )
    parser.add_argument(
        "--hidden_features", type=int, default=256, help="size of the hidden features"
    )
    parser.add_argument(
        "--omega_0", type=float, default=3.0, help="first SIREN omega_0"
    )
    parser.add_argument(
        "--hidden_omega_0",
        type=float,
        default=3.0,
        help="hidden-layer SIREN omega_0",
    )
    parser.add_argument(
        "--sampling",
        choices=("iid", "sobol", "fixed_sobol", "wall_mixture"),
        default="iid",
        help="collocation sampler (default: iid)",
    )
    parser.add_argument(
        "--grouped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="share samples and field evaluations by geometric domain",
    )
    parser.add_argument(
        "--penalty",
        choices=("mse", "mae", "pseudo_huber"),
        default="mse",
        help="pointwise residual penalty (default: mse)",
    )
    parser.add_argument(
        "--loss_balance",
        choices=("boundary_pde_ramp", "equal_groups", "legacy"),
        default="boundary_pde_ramp",
        help="boundary/interior loss weighting (default: boundary_pde_ramp)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Python, NumPy, and Torch seed"
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=180.0,
        help="training wall-clock budget (default: 180)",
    )
    parser.add_argument(
        "--resolution", type=int, default=128, help="image resolution for the gif"
    )
    parser.add_argument(
        "--nb_frames", type=int, default=50, help="number of frames for the gif"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device; auto selects CUDA when available (default: auto)",
    )
    parser.add_argument(
        "--no_gif", action="store_true", help="skip GIF frame collection and export"
    )
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    equations, variables, domains, solution_functions = parse_problem_file(
        args.input_file
    )
    frames = []
    frame_coordinates = None
    animation_started = time.monotonic()
    animation_state = {
        "next_frame_fraction": 0.0,
        "last_frame_iteration": None,
    }
    if not args.no_gif:
        frame_coordinates = torch.cartesian_prod(
            torch.linspace(0, 1, args.resolution),
            torch.linspace(0, 1, args.resolution),
        ).to(args.device)

    def append_frame(iteration, model):
        with torch.no_grad():
            out = model(
                frame_coordinates[:, 0:1], frame_coordinates[:, 1:2]
            )
        out = out.view(args.resolution, args.resolution, out.size(-1))
        out = out.rot90().cpu().numpy()
        frames.append(out)
        animation_state["last_frame_iteration"] = iteration

    def collect_frame(iteration, model, loss_value):
        current_fraction = min(
            1.0,
            (time.monotonic() - animation_started)
            / max(1e-6, args.max_seconds - 4.0),
        )
        if current_fraction >= animation_state["next_frame_fraction"]:
            append_frame(iteration, model)
            animation_state["next_frame_fraction"] += 1.0 / max(
                1, args.nb_frames
            )

    model = train_model(
        equations,
        variables,
        domains,
        nb_iter=args.nb_iter,
        nb_samples=args.nb_samples,
        lr=args.lr,
        min_lr=args.min_lr,
        lr_schedule=args.lr_schedule,
        lr_gamma=args.lr_gamma,
        hidden_layers=args.hidden_layers,
        hidden_features=args.hidden_features,
        first_omega_0=args.omega_0,
        hidden_omega_0=args.hidden_omega_0,
        sampling=args.sampling,
        grouped=args.grouped,
        penalty=args.penalty,
        loss_balance=args.loss_balance,
        seed=args.seed,
        max_seconds=args.max_seconds,
        device=args.device,
        iteration_callback=None if args.no_gif else collect_frame,
    )

    if not args.no_gif:
        if (
            animation_state["last_frame_iteration"] is None
            or len(frames) < args.nb_frames
        ):
            append_frame(args.nb_iter, model)
        solution = None
        if solution_functions:
            with torch.no_grad():
                solution = evaluate_solution_functions(
                    solution_functions,
                    variables,
                    frame_coordinates[:, 0],
                    frame_coordinates[:, 1],
                )
            solution = solution.view(
                args.resolution, args.resolution, len(variables)
            )
            solution = solution.rot90().cpu().numpy()
        make_gif(frames, variables, args.output_file, solution=solution)
