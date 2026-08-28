import ast
import copy
import json
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


COORDINATE_NAMES = ("x", "y", "t")


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

    def forward(self, *coordinates):
        return self.net(torch.cat(coordinates, dim=-1))

    def forward_with_derivatives(self, *args):
        """Evaluate outputs and requested pure derivatives in one forward pass.

        SIREN derivatives can be propagated exactly through each linear and
        sine layer. This avoids constructing nested ``autograd.grad`` graphs,
        while ordinary autograd still computes parameter gradients from the
        resulting residual loss.
        """
        if len(args) < 2:
            raise ValueError("Expected coordinates followed by derivative orders")
        *coordinates, derivative_orders = args
        if len(coordinates) != self.net[0].in_features:
            raise ValueError(
                f"Expected {self.net[0].in_features} coordinates, "
                f"got {len(coordinates)}"
            )
        zero_order = (0,) * len(coordinates)
        derivative_orders = tuple(
            dict.fromkeys(
                order for order in derivative_orders if order != zero_order
            )
        )
        if any(
            len(order) != len(coordinates)
            or sum(component > 0 for component in order) > 1
            or max(order, default=0) > MAX_ANALYTIC_DERIVATIVE_ORDER
            for order in derivative_orders
        ):
            raise ValueError("Only pure derivatives through fourth order are supported")

        maximum_orders = [
            max((order[index] for order in derivative_orders), default=0)
            for index in range(len(coordinates))
        ]
        value = torch.cat(coordinates, dim=-1)
        coordinate_derivatives = []
        for coordinate_index, maximum_order in enumerate(maximum_orders):
            derivatives = [value]
            if maximum_order:
                basis = torch.zeros_like(value)
                basis[:, coordinate_index : coordinate_index + 1] = 1
                derivatives.extend(
                    [
                        basis,
                        *[
                            torch.zeros_like(value)
                            for _ in range(maximum_order - 1)
                        ],
                    ]
                )
            coordinate_derivatives.append(derivatives)

        batch_size = value.size(0)
        for layer in self.net[:-1]:
            derivative_inputs = [
                value,
                *[
                    derivative
                    for derivatives in coordinate_derivatives
                    for derivative in derivatives[1:]
                ],
            ]
            transformed = F.linear(
                torch.cat(derivative_inputs, dim=0), layer.linear.weight
            ).split(batch_size, dim=0)
            argument = layer.omega_0 * (transformed[0] + layer.linear.bias)
            sine = torch.sin(argument)
            cosine = torch.cos(argument)
            offset = 1
            next_derivatives = []
            for maximum_order in maximum_orders:
                if maximum_order:
                    arguments = [
                        argument,
                        *(
                            layer.omega_0 * derivative
                            for derivative in transformed[
                                offset : offset + maximum_order
                            ]
                        ),
                    ]
                    derivatives = _sine_derivatives(arguments, sine, cosine)
                    offset += maximum_order
                else:
                    derivatives = [sine]
                next_derivatives.append(derivatives)
            coordinate_derivatives = next_derivatives
            value = sine
            for derivatives in coordinate_derivatives:
                derivatives[0] = value

        final_layer = self.net[-1]
        requested_inputs = [value]
        for order in derivative_orders:
            coordinate_index = next(
                index for index, component in enumerate(order) if component
            )
            requested_inputs.append(
                coordinate_derivatives[coordinate_index][order[coordinate_index]]
            )
        transformed = F.linear(
            torch.cat(requested_inputs, dim=0), final_layer.weight
        ).split(batch_size, dim=0)
        outputs = {zero_order: transformed[0] + final_layer.bias}
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

    def forward(self, *coordinates):
        return self.net(torch.cat(coordinates, dim=-1))


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

    def forward(self, *inputs):
        coordinates = torch.cat(inputs, dim=-1)
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

    def forward(self, *inputs):
        coordinates = torch.cat(inputs, dim=-1)
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

    def forward(self, *coordinates):
        hidden = self.activation(self.input_layer(torch.cat(coordinates, dim=-1)))
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

    def forward(self, *coordinates):
        hidden = self.trunk(torch.cat(coordinates, dim=-1))
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
    coordinate_names=None,
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
    if coordinate_names is None:
        coordinate_names = COORDINATE_NAMES[:in_features]
    if len(coordinate_names) != in_features:
        raise ValueError("coordinate_names must match in_features")
    model.coordinate_names = tuple(coordinate_names)
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
SECTION_PATTERN = re.compile(
    r"^\s*#\s*(Domains|Equations|Solution)\s*$", re.IGNORECASE
)
DOMAIN_ANNOTATION = re.compile(
    r"^(?P<formula>.+?)\s+@\s+(?P<domain>[A-Za-z_]\w*)\s*$"
)

# -----------------------------------------------------------------------------
# Parser functions


def _split_domain_annotation(formula):
    match = DOMAIN_ANNOTATION.match(formula)
    if match is None:
        return formula, None
    return match.group("formula").strip(), match.group("domain")


def _geometry_scalar(node, formula):
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _geometry_scalar(node.operand, formula)
        value = -operand if isinstance(node.op, ast.USub) else operand
    elif isinstance(node, ast.Name) and node.id in CONSTANTS:
        value = CONSTANTS[node.id]
    elif (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Mult, ast.Div, ast.Add, ast.Sub))
    ):
        left = _geometry_scalar(node.left, formula)
        right = _geometry_scalar(node.right, formula)
        value = OPS[type(node.op)](left, right)
    else:
        raise InvalidFormula(formula, "Geometry values must be numeric")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise InvalidFormula(formula, "Geometry values must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidFormula(formula, "Geometry values must be finite")
    return value


def _geometry_interval(node, formula, allow_scalar=True):
    if isinstance(node, (ast.Tuple, ast.List)):
        if len(node.elts) != 2:
            raise InvalidFormula(formula, "Coordinate ranges need two endpoints")
        lower, upper = (_geometry_scalar(item, formula) for item in node.elts)
        if not lower < upper:
            raise InvalidFormula(formula, "Coordinate ranges must be increasing")
        return [lower, upper]
    if not allow_scalar:
        raise InvalidFormula(formula, "Expected a two-value range")
    return _geometry_scalar(node, formula)


def _parse_geometry_node(node, formula, named_geometries):
    if isinstance(node, ast.Name):
        try:
            return copy.deepcopy(named_geometries[node.id])
        except KeyError as error:
            raise InvalidFormula(formula, f"Unknown domain {node.id!r}") from error
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise InvalidFormula(formula, "Expected a geometry constructor")
    name = node.func.id.lower()
    if name in {"difference", "union"}:
        if node.keywords or len(node.args) < 2:
            raise InvalidFormula(formula, f"{name} expects at least two domains")
        parts = [
            _parse_geometry_node(argument, formula, named_geometries)
            for argument in node.args
        ]
        if name == "difference":
            return {"type": "difference", "outer": parts[0], "holes": parts[1:]}
        part_coordinates = {
            _geometry_coordinate_names(part) for part in parts
        }
        if len(part_coordinates) != 1:
            raise InvalidFormula(
                formula, "union parts must use the same coordinates"
            )
        return {"type": "union", "parts": parts}
    if node.args:
        raise InvalidFormula(formula, f"{name} accepts keyword arguments only")
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords:
        raise InvalidFormula(formula, "Expanded geometry keywords are unsupported")
    if name in {"box", "rectangle"}:
        unknown = set(keywords) - set(COORDINATE_NAMES)
        if unknown or not {"x", "y"}.issubset(keywords):
            raise InvalidFormula(
                formula,
                "box expects x and y ranges, with optional t",
            )
        coordinates = {
            coordinate: _geometry_interval(value, formula)
            for coordinate, value in keywords.items()
        }
        return {"type": "box", "coordinates": coordinates}
    if name in {"circle", "disk"}:
        unknown = set(keywords) - {"center", "radius", "t"}
        if unknown or "center" not in keywords or "radius" not in keywords:
            raise InvalidFormula(
                formula,
                f"{name} expects center=(x, y), radius=r, and optional t",
            )
        center_node = keywords["center"]
        if not isinstance(center_node, (ast.Tuple, ast.List)) or len(center_node.elts) != 2:
            raise InvalidFormula(formula, "center must contain x and y")
        center = [_geometry_scalar(item, formula) for item in center_node.elts]
        radius = _geometry_scalar(keywords["radius"], formula)
        if radius <= 0:
            raise InvalidFormula(formula, "radius must be positive")
        geometry = {"type": name, "center": center, "radius": radius}
        if "t" in keywords:
            geometry["t"] = _geometry_interval(keywords["t"], formula)
        return geometry
    raise InvalidFormula(
        formula,
        "Unknown geometry; choose box, circle, disk, difference, or union",
    )


def parse_domain_definitions(domain_equations, input_file="<domains>"):
    """Parse safe named geometry expressions from a ``# Domains`` section."""
    named = {}
    for line_number, formula in domain_equations:
        parts = formula.split("=", 1)
        if len(parts) != 2 or not re.fullmatch(r"[A-Za-z_]\w*", parts[0].strip()):
            raise ValueError(
                f"{input_file}:{line_number}: domain must have the form "
                "'name = geometry(...)'"
            )
        name, expression = parts[0].strip(), parts[1].strip()
        if name in named:
            raise ValueError(f"{input_file}:{line_number}: duplicate domain {name!r}")
        node = ast.parse(expression, filename=str(input_file), mode="eval").body
        try:
            named[name] = _parse_geometry_node(node, formula, named)
        except InvalidFormula as error:
            raise ValueError(f"{input_file}:{line_number}: {error}") from error
    return named


def _geometry_coordinate_names(geometry):
    kind = geometry["type"]
    if kind == "box":
        return tuple(
            coordinate
            for coordinate in COORDINATE_NAMES
            if coordinate in geometry["coordinates"]
        )
    if kind in {"circle", "disk"}:
        return ("x", "y", "t") if "t" in geometry else ("x", "y")
    if kind == "difference":
        return _geometry_coordinate_names(geometry["outer"])
    names = set()
    for part in geometry["parts"]:
        names.update(_geometry_coordinate_names(part))
    return tuple(coordinate for coordinate in COORDINATE_NAMES if coordinate in names)


def _geometry_bounds(geometry):
    kind = geometry["type"]
    if kind == "box":
        return copy.deepcopy(geometry["coordinates"])
    if kind in {"circle", "disk"}:
        cx, cy = geometry["center"]
        radius = geometry["radius"]
        bounds = {"x": [cx - radius, cx + radius], "y": [cy - radius, cy + radius]}
        if "t" in geometry:
            bounds["t"] = copy.deepcopy(geometry["t"])
        return bounds
    if kind == "difference":
        return _geometry_bounds(geometry["outer"])
    part_bounds = [_geometry_bounds(part) for part in geometry["parts"]]
    bounds = {}
    for coordinate in _geometry_coordinate_names(geometry):
        intervals = []
        for item in part_bounds:
            value = item[coordinate]
            intervals.append(value if isinstance(value, list) else [value, value])
        lower = min(value[0] for value in intervals)
        upper = max(value[1] for value in intervals)
        bounds[coordinate] = lower if lower == upper else [lower, upper]
    return bounds


def _domain_from_geometry(name, geometry, coordinate_names):
    geometry_names = _geometry_coordinate_names(geometry)
    if tuple(geometry_names) != tuple(coordinate_names):
        raise ValueError(
            f"Domain {name!r} uses coordinates {geometry_names}, expected "
            f"{tuple(coordinate_names)}"
        )
    bounds = _geometry_bounds(geometry)
    domain = {"_name": name, "_geometry": copy.deepcopy(geometry), "_bounds": bounds}
    for coordinate in coordinate_names:
        value = bounds[coordinate]
        domain[coordinate] = float(value) if not isinstance(value, list) else np.nan
    return domain


def infer_coordinate_names(equations, named_geometries=None):
    """Infer whether a problem uses ``(x, y)`` or ``(x, y, t)``."""
    uses_time = False
    for annotated_formula in equations:
        formula, _ = _split_domain_annotation(annotated_formula)
        for expression in formula.split("=", 1):
            if any(
                isinstance(node, ast.Name) and node.id == "t"
                for node in ast.walk(ast.parse(expression.strip(), mode="eval"))
            ):
                uses_time = True
                break
        if uses_time:
            break
    if named_geometries:
        uses_time = uses_time or any(
            "t" in _geometry_coordinate_names(geometry)
            for geometry in named_geometries.values()
        )
    return ("x", "y", "t") if uses_time else ("x", "y")


def coordinate_names_from_domains(domains):
    if not domains:
        raise ValueError("At least one equation domain is required")
    return tuple(
        coordinate for coordinate in COORDINATE_NAMES if coordinate in domains[0]
    )


def parse_equations(equations, verbose=True, named_geometries=None):
    named_geometries = named_geometries or {}
    coordinate_names = infer_coordinate_names(equations, named_geometries)
    named_domains = {
        name: _domain_from_geometry(name, geometry, coordinate_names)
        for name, geometry in named_geometries.items()
    }
    variables = {}
    domains = []
    for annotated_formula in equations:
        formula, domain_name = _split_domain_annotation(annotated_formula)
        splits = formula.split("=")
        if len(splits) != 2:
            raise InvalidFormula(formula, "Not a equation")
        lhs, rhs = splits
        fixed_coordinates = {coordinate: set() for coordinate in coordinate_names}
        parse(
            formula,
            ast.parse(lhs.strip(), mode="eval").body,
            variables,
            fixed_coordinates,
            coordinate_names=coordinate_names,
        )
        parse(
            formula,
            ast.parse(rhs.strip(), mode="eval").body,
            variables,
            fixed_coordinates,
            coordinate_names=coordinate_names,
        )
        if domain_name is not None:
            try:
                domain = copy.deepcopy(named_domains[domain_name])
            except KeyError as error:
                raise InvalidFormula(formula, f"Unknown domain {domain_name!r}") from error
        else:
            domain = {
                coordinate: next(iter(values)) if len(values) == 1 else np.nan
                for coordinate, values in fixed_coordinates.items()
            }
            domain["_bounds"] = {
                coordinate: (
                    float(domain[coordinate])
                    if not np.isnan(domain[coordinate])
                    else [0.0, 1.0]
                )
                for coordinate in coordinate_names
            }
        domains.append(domain)
        log = f"Parsed equation {len(domains)}: 'for "
        for inp in coordinate_names:
            if np.isnan(domain[inp]):
                bounds = domain["_bounds"][inp]
                log += f"{inp} in ({bounds[0]}, {bounds[1]}), "
            else:
                log += f"{inp} = {domain[inp]}, "
        if verbose:
            print(log[:-2] + f",  {formula}'")
    variables = list(variables.keys())
    if verbose:
        print(f"Found {len(variables)} unknown function(s) to approximate: {variables}")
    return variables, domains


def _read_problem_sections(input_file):
    sections = {"domains": [], "equations": [], "solution": []}
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
                if section == "domains" and seen_sections:
                    raise ValueError(
                        f"{input_file}:{line_number}: '# Domains' must be first"
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
                    "'# Domains', '# Equations', or '# Solution'"
                )
            sections[current_section].append((line_number, line))

    if "equations" not in seen_sections:
        raise ValueError(f"{input_file}: missing '# Equations' section")
    if not sections["equations"]:
        raise ValueError(f"{input_file}: '# Equations' section is empty")
    if "solution" in seen_sections and not sections["solution"]:
        raise ValueError(f"{input_file}: '# Solution' section is empty")

    return sections, "solution" in seen_sections


def _parse_solution_equations(
    solution_equations, coordinate_names, input_file="<solution>"
):
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
            fixed_coordinates={coordinate: set() for coordinate in coordinate_names},
            functions=MATH_FUNS,
            allow_unknown_functions=False,
            coordinate_names=coordinate_names,
        )
        solution_functions[field] = node
    return solution_functions


def parse_problem_file(input_file, verbose=True):
    """Parse equations and an optional analytic solution from a problem file."""
    sections, has_solution = _read_problem_sections(input_file)
    annotated_equations = [formula for _, formula in sections["equations"]]
    named_geometries = parse_domain_definitions(
        sections["domains"], input_file=input_file
    )
    variables, domains = parse_equations(
        annotated_equations, verbose=verbose, named_geometries=named_geometries
    )
    equations = [
        _split_domain_annotation(formula)[0] for formula in annotated_equations
    ]
    coordinate_names = coordinate_names_from_domains(domains)
    solution_functions = _parse_solution_equations(
        sections["solution"], coordinate_names, input_file=input_file
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
    coordinate_names=("x", "y"),
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
            coordinate_names,
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
            if len(node.args) != len(coordinate_names):
                raise InvalidFormula(
                    formula, f"Invalid nb of args for '{node.func.id}'"
                )
            for coordinate, argument in zip(coordinate_names, node.args):
                if isinstance(argument, ast.Name) and argument.id == coordinate:
                    continue
                try:
                    _geometry_scalar(argument, formula)
                except InvalidFormula as error:
                    if isinstance(argument, ast.Name):
                        raise InvalidFormula(
                            formula,
                            f"'{node.func.id}' takes coordinates "
                            f"{tuple(coordinate_names)} in that order",
                        ) from error
                    raise InvalidFormula(
                        formula, f"Found invalid arg for '{node.func.id}'"
                    ) from error
            for inp, arg in zip(coordinate_names, node.args):
                if not (isinstance(arg, ast.Name) and arg.id == inp):
                    fixed_coordinates[inp].add(_geometry_scalar(arg, formula))
            variables[node.func.id] = None
            return variables
    elif isinstance(node, ast.Name):
        if node.id in coordinate_names or node.id in CONSTANTS:
            return variables
        elif allow_unknown_functions:
            parse_child(
                ast.Call(
                    func=ast.Name(id=node.id, ctx=ast.Load()),
                    args=[
                        *[
                            ast.Name(id=coordinate, ctx=ast.Load())
                            for coordinate in coordinate_names
                        ],
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
    else:
        try:
            numeric_value = _geometry_scalar(node, ast.unparse(node))
        except InvalidFormula as error:
            raise RuntimeError(
                f"Invalid coordinate argument: {ast.dump(node)}"
            ) from error
        key = (coordinate, numeric_value)
        if key not in coordinate_cache:
            coordinate_cache[key] = torch.full_like(
                samples[coordinate],
                numeric_value,
                requires_grad=samples[coordinate].requires_grad,
            )
        value = coordinate_cache[key]
    _record_coordinate(used_coordinates, coordinate, value)
    return value


def _merge_coordinates(source, destination):
    for coordinate in source:
        for value in source[coordinate]:
            _record_coordinate(destination, coordinate, value)


def _coordinate_signature(coordinate_nodes):
    return tuple(
        ast.dump(node, annotate_fields=False, include_attributes=False)
        for node in coordinate_nodes
    )


def _direct_field_derivative(node, field_indices, coordinate_names):
    """Return a direct field derivative without evaluating its AST."""
    derivative_order = [0] * len(coordinate_names)
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "grad"
    ):
        coordinate_node = node.args[1]
        if (
            not isinstance(coordinate_node, ast.Name)
            or coordinate_node.id not in coordinate_names
        ):
            return None
        coordinate_index = coordinate_names.index(coordinate_node.id)
        derivative_order[coordinate_index] += 1
        node = node.args[0]

    if not any(derivative_order):
        return None
    if isinstance(node, ast.Name) and node.id in field_indices:
        field = node.id
        coordinate_nodes = tuple(
            ast.Name(id=coordinate, ctx=ast.Load())
            for coordinate in coordinate_names
        )
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in field_indices
        and len(node.args) == len(coordinate_names)
    ):
        field = node.func.id
        coordinate_nodes = tuple(node.args)
    else:
        return None

    return field, coordinate_nodes, tuple(derivative_order)


def _supports_analytic_derivative(derivative_order):
    return (
        sum(component > 0 for component in derivative_order) <= 1
        and max(derivative_order) <= MAX_ANALYTIC_DERIVATIVE_ORDER
    )


def _collect_derivative_requests(node, field_indices, coordinate_names):
    requests = {}

    def visit(child):
        derivative = _direct_field_derivative(
            child, field_indices, coordinate_names
        )
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
    coordinate_names = tuple(samples)
    if coordinate_cache is None:
        coordinate_cache = {}
    if used_coordinates is None:
        used_coordinates = {coordinate: [] for coordinate in coordinate_names}
    if evaluation_cache is None:
        evaluation_cache = {
            "model_outputs": {},
            "derivative_outputs": {},
            "derivative_requests": {},
        }
    model_cache = evaluation_cache["model_outputs"]
    derivative_cache = evaluation_cache["derivative_outputs"]
    derivative_requests = evaluation_cache["derivative_requests"]

    direct_derivative = _direct_field_derivative(
        node, field_indices, coordinate_names
    )
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
            for coordinate, coordinate_node in zip(
                coordinate_names, coordinate_nodes
            )
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
            model_cache[cache_key] = derivative_cache[cache_key][
                (0,) * len(coordinate_names)
            ]
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
            if (
                not isinstance(coordinate_node, ast.Name)
                or coordinate_node.id not in coordinate_names
            ):
                raise RuntimeError(
                    f"grad expects one of {coordinate_names} as its second argument"
                )
            coordinate = coordinate_node.id
            local_coordinates = {
                name: [] for name in coordinate_names
            }
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
                for coordinate, arg in zip(coordinate_names, node.args)
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
                model_cache[cache_key] = derivative_cache[cache_key][
                    (0,) * len(coordinate_names)
                ]
            if cache_key not in model_cache:
                model_cache[cache_key] = model(*coordinates)
            return model_cache[cache_key][:, field_index : field_index + 1]
    elif isinstance(node, ast.Name):
        if node.id in samples:
            if node.id in coordinate_names:
                _record_coordinate(used_coordinates, node.id, samples[node.id])
            return samples[node.id]
        elif node.id in CONSTANTS:
            return CONSTANTS[node.id]
        elif node.id in field_indices:
            return evaluate(
                ast.Call(
                    func=ast.Name(id=node.id, ctx=ast.Load()),
                    args=[
                        ast.Name(id=coordinate, ctx=ast.Load())
                        for coordinate in coordinate_names
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


def evaluate_expected_function(node, samples):
    """Evaluate one parsed solution expression with the shared Torch evaluator."""
    value = evaluate(
        node,
        samples=samples,
        model=None,
        field_indices={},
    )
    reference = next(iter(samples.values()))
    value = torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device
    )
    if value.ndim == 0:
        value = value.expand_as(reference)
    try:
        value = torch.broadcast_to(value, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"Expected function has shape {tuple(value.shape)}, "
            f"not {tuple(reference.shape)}"
        ) from error
    if not torch.isfinite(value).all():
        raise ValueError("Expected function produced a non-finite value")
    return value


def evaluate_solution_functions(solution_functions, variables, samples):
    """Evaluate solution fields in the same order as the model outputs."""
    if set(solution_functions) != set(variables):
        raise ValueError("Solution fields do not match model output fields")
    return torch.cat(
        [
            evaluate_expected_function(solution_functions[variable], samples)
            for variable in variables
        ],
        dim=-1,
    )


# -----------------------------------------------------------------------------
# Training and visualization functions


def _domain_key(domain):
    payload = {}
    for key, value in domain.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        payload[key] = value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _as_interval(value):
    return value if isinstance(value, list) else [value, value]


def _geometry_latent_dimension(geometry):
    kind = geometry["type"]
    if kind == "box":
        return sum(
            isinstance(value, list)
            for value in geometry["coordinates"].values()
        )
    if kind == "circle":
        return 1 + int(isinstance(geometry.get("t"), list))
    if kind == "disk":
        return 2 + int(isinstance(geometry.get("t"), list))
    if kind == "difference":
        return _geometry_latent_dimension(geometry["outer"])
    return 1 + max(
        _geometry_latent_dimension(part) for part in geometry["parts"]
    )


def domain_dimension(domain):
    """Return the number of freely sampled dimensions in an equation domain."""
    geometry = domain.get("_geometry")
    if geometry is not None:
        if geometry["type"] == "union":
            return max(
                domain_dimension(
                    _domain_from_geometry(
                        domain.get("_name", "union"),
                        part,
                        _geometry_coordinate_names(part),
                    )
                )
                for part in geometry["parts"]
            )
        return _geometry_latent_dimension(geometry)
    return sum(
        np.isnan(domain[coordinate])
        for coordinate in coordinate_names_from_domains([domain])
    )


def is_interior_domain(domain):
    return domain_dimension(domain) == len(coordinate_names_from_domains([domain]))


def _geometry_contains(geometry, samples):
    kind = geometry["type"]
    if kind == "box":
        keep = torch.ones_like(next(iter(samples.values())), dtype=torch.bool)
        for coordinate, value in geometry["coordinates"].items():
            lower, upper = _as_interval(value)
            keep &= samples[coordinate] >= lower
            keep &= samples[coordinate] <= upper
        return keep.flatten()
    if kind in {"circle", "disk"}:
        cx, cy = geometry["center"]
        radius_squared = geometry["radius"] ** 2
        squared_distance = (
            (samples["x"] - cx).square() + (samples["y"] - cy).square()
        )
        tolerance = max(1e-8, radius_squared * 1e-5)
        if kind == "circle":
            return ((squared_distance - radius_squared).abs() <= tolerance).flatten()
        return (squared_distance <= radius_squared).flatten()
    if kind == "difference":
        keep = _geometry_contains(geometry["outer"], samples)
        for hole in geometry["holes"]:
            keep &= ~_geometry_contains(hole, samples)
        return keep
    masks = [_geometry_contains(part, samples) for part in geometry["parts"]]
    return torch.stack(masks).any(dim=0)


def _transform_geometry(geometry, unit):
    kind = geometry["type"]
    if kind == "box":
        samples = {}
        column = 0
        template = (
            unit[:, :1]
            if unit.size(1)
            else torch.empty(unit.size(0), 1, dtype=unit.dtype)
        )
        for coordinate in COORDINATE_NAMES:
            if coordinate not in geometry["coordinates"]:
                continue
            value = geometry["coordinates"][coordinate]
            if isinstance(value, list):
                lower, upper = value
                samples[coordinate] = lower + (upper - lower) * unit[:, column : column + 1]
                column += 1
            else:
                samples[coordinate] = torch.full_like(template, float(value))
        return samples
    if kind in {"circle", "disk"}:
        if kind == "circle":
            radius = torch.full_like(unit[:, :1], float(geometry["radius"]))
            angle_column = 0
            next_column = 1
        else:
            radius = float(geometry["radius"]) * torch.sqrt(unit[:, :1])
            angle_column = 1
            next_column = 2
        angle = 2 * math.pi * unit[:, angle_column : angle_column + 1]
        cx, cy = geometry["center"]
        samples = {
            "x": cx + radius * torch.cos(angle),
            "y": cy + radius * torch.sin(angle),
        }
        if "t" in geometry:
            value = geometry["t"]
            if isinstance(value, list):
                lower, upper = value
                samples["t"] = lower + (upper - lower) * unit[
                    :, next_column : next_column + 1
                ]
            else:
                samples["t"] = torch.full_like(unit[:, :1], float(value))
        return samples
    if kind == "union":
        part_count = len(geometry["parts"])
        choices = torch.clamp((unit[:, 0] * part_count).long(), max=part_count - 1)
        combined = {}
        for part_index, part in enumerate(geometry["parts"]):
            rows = choices == part_index
            if not rows.any():
                continue
            part_dimension = _geometry_latent_dimension(part)
            values = _transform_geometry(part, unit[rows, 1 : 1 + part_dimension])
            for coordinate, value in values.items():
                combined.setdefault(
                    coordinate, torch.empty(unit.size(0), 1, dtype=unit.dtype)
                )[rows] = value
        return combined
    raise ValueError("difference geometries require rejection sampling")


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

    def _unit_samples(self, key, nb_samples, dimensions):
        if not dimensions:
            return torch.empty(nb_samples, 0)
        if self.method == "iid":
            engine_key = ("iid", key, dimensions)
            generator = self._engines.setdefault(
                engine_key,
                torch.Generator().manual_seed(self.seed + len(self._engines)),
            )
            return torch.rand(nb_samples, dimensions, generator=generator)
        engine_key = ("sobol", key, dimensions)
        engine = self._engines.setdefault(
            engine_key,
            torch.quasirandom.SobolEngine(
                dimensions,
                scramble=True,
                seed=self.seed + len(self._engines),
            ),
        )
        return engine.draw(nb_samples)

    def _sample_geometry(self, geometry, nb_samples, key):
        if geometry["type"] != "difference":
            dimensions = _geometry_latent_dimension(geometry)
            unit = self._unit_samples(key, nb_samples, dimensions)
            return _transform_geometry(geometry, unit)

        accepted = []
        remaining = nb_samples
        attempts = 0
        while remaining:
            attempts += 1
            if attempts > 100:
                raise RuntimeError("Could not sample enough points outside geometry holes")
            candidate_count = max(remaining * 2, 64)
            outer = geometry["outer"]
            dimensions = _geometry_latent_dimension(outer)
            unit = self._unit_samples(
                f"{key}:rejection:{attempts}", candidate_count, dimensions
            )
            candidates = _transform_geometry(outer, unit)
            keep = torch.ones(candidate_count, dtype=torch.bool)
            for hole in geometry["holes"]:
                keep &= ~_geometry_contains(hole, candidates)
            take = min(remaining, int(keep.sum()))
            if take:
                indices = torch.nonzero(keep, as_tuple=False)[:take, 0]
                accepted.append(
                    {coordinate: value[indices] for coordinate, value in candidates.items()}
                )
                remaining -= take
        return {
            coordinate: torch.cat([batch[coordinate] for batch in accepted], dim=0)
            for coordinate in accepted[0]
        }

    def sample(self, domain, nb_samples, device, requires_grad=True):
        key = (_domain_key(domain), int(nb_samples))
        if self.method == "fixed_sobol" and key in self._fixed_samples:
            samples = {
                coordinate: value.clone()
                for coordinate, value in self._fixed_samples[key].items()
            }
        else:
            geometry = domain.get("_geometry")
            if geometry is None:
                geometry = {
                    "type": "box",
                    "coordinates": copy.deepcopy(domain["_bounds"]),
                }
            samples = self._sample_geometry(geometry, nb_samples, key[0])
            if self.method == "fixed_sobol":
                self._fixed_samples[key] = {
                    coordinate: value.clone()
                    for coordinate, value in samples.items()
                }

        geometry = domain.get("_geometry")
        if (
            self.method == "wall_mixture"
            and geometry is not None
            and geometry["type"] == "box"
            and all(
                isinstance(geometry["coordinates"].get(coordinate), list)
                for coordinate in ("x", "y")
            )
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
                for local_axis, coordinate in enumerate(("x", "y")):
                    lower, upper = geometry["coordinates"][coordinate]
                    selected = axis == local_axis
                    if selected.any():
                        normalized = torch.where(
                            side[selected].bool(),
                            1 - distance[selected],
                            distance[selected],
                        )
                        samples[coordinate][rows[selected], 0] = (
                            lower + (upper - lower) * normalized
                        )

        return {
            coordinate: value.to(device=device)
            .detach()
            .requires_grad_(requires_grad)
            for coordinate, value in samples.items()
        }


def generate_samples(domain, nb_samples, device, requires_grad=True, sampler=None):
    if sampler is None:
        sampler = CollocationSampler(method="iid")
    return sampler.sample(domain, nb_samples, device, requires_grad)


def _requires_autograd_derivatives(
    node, field_indices, analytic_derivatives, coordinate_names
):
    derivative = _direct_field_derivative(
        node, field_indices, coordinate_names
    )
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
        _requires_autograd_derivatives(
            child, field_indices, analytic_derivatives, coordinate_names
        )
        for child in ast.iter_child_nodes(node)
    )


def compile_residuals(equations, field_indices, model):
    """Parse equations once and annotate their derivative requirements."""
    analytic_derivatives = hasattr(model, "forward_with_derivatives")
    coordinate_names = tuple(
        getattr(model, "coordinate_names", COORDINATE_NAMES[:2])
    )
    residuals = []
    for formula in equations:
        lhs, rhs = formula.split("=")
        node = ast.parse(f"{lhs} - ({rhs})".strip(), mode="eval").body
        residuals.append(
            (
                node,
                _collect_derivative_requests(
                    node, field_indices, coordinate_names
                ),
                _requires_autograd_derivatives(
                    node,
                    field_indices,
                    analytic_derivatives,
                    coordinate_names,
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
        weight = 0.1 if is_interior_domain(domain) else 0.9
        losses.append(weight * equation_loss)
    return torch.stack(losses).mean()


def _is_interior_domain(domain):
    """Backward-compatible alias for :func:`is_interior_domain`."""
    return is_interior_domain(domain)


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
    each optimizer step, allowing library clients to collect diagnostics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    field_indices = {
        variable: index for index, variable in enumerate(variables)
    }
    coordinate_names = coordinate_names_from_domains(domains)
    model = Siren(
        in_features=len(coordinate_names),
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=len(variables),
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
    ).to(device)
    model.coordinate_names = coordinate_names
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
        # Leave four seconds for validation and caller-side bookkeeping inside
        # the advertised runtime budget.
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
            if domain_dimension(domain) == 0:
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


def make_static_plot(frame, variables, output_file, extent=(0, 1, 0, 1)):
    """Save one final multi-field approximation image."""
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != len(variables):
        raise ValueError(
            f"Frame has shape {frame.shape}, expected (height, width, "
            f"{len(variables)})"
        )

    column_count = len(variables)
    fig, axes = plt.subplots(
        1,
        column_count,
        squeeze=False,
        figsize=(4.8 * column_count, 4.0),
    )
    for field_index, variable in enumerate(variables):
        field = frame[:, :, field_index]
        if not np.isfinite(field).any():
            raise ValueError(f"Field {variable!r} has no finite values")
        field_limits = _plot_limits(np.nanmin(field), np.nanmax(field))
        axis = axes[0, field_index]
        image_artist = axis.imshow(
            field,
            extent=extent,
            origin="lower",
            vmin=field_limits[0],
            vmax=field_limits[1],
        )
        fig.colorbar(image_artist, ax=axis)
        axis.set_title(f"Approximation: {variable}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.margins(0)

    fig.tight_layout()
    fig.savefig(str(output_file), dpi=150)
    plt.close(fig)
