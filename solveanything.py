import argparse
import ast
import re
import numpy as np
import torch
import operator
import matplotlib.pyplot as plt

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
                samples[coordinate], float(node.value), requires_grad=True
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


def evaluate(
    node,
    samples,
    model,
    field_indices,
    coordinate_cache=None,
    used_coordinates=None,
):
    if coordinate_cache is None:
        coordinate_cache = {}
    if used_coordinates is None:
        used_coordinates = {"x": [], "y": []}

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
            ),
            evaluate(
                node.right,
                samples,
                model,
                field_indices,
                coordinate_cache,
                used_coordinates,
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
            return model(*coordinates)[:, field_index : field_index + 1]
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


def generate_samples(domain, nb_samples, device):
    samples = {"x": None, "y": None}
    for inp in ["x", "y"]:
        if np.isnan(domain[inp]):
            samples[inp] = torch.rand(nb_samples, 1)
        else:
            samples[inp] = torch.ones(nb_samples, 1) * domain[inp]
        samples[inp] = samples[inp].clone().detach().requires_grad_(True).to(device)
    return samples


def compute_loss(
    equations,
    domains,
    model,
    field_indices,
    nb_samples,
    device,
):
    losses = []
    for formula, domain in zip(equations, domains):
        lhs, rhs = formula.split("=")
        samples = generate_samples(domain, nb_samples, device)
        res = evaluate(
            ast.parse(f"{lhs} - ({rhs})".strip(), mode="eval").body,
            samples,
            model,
            field_indices,
        )
        equation_loss = torch.mean(torch.abs(res))
        weight = 0.1 if np.isnan(domain["x"]) and np.isnan(domain["y"]) else 0.9
        losses.append(weight * equation_loss)
    return torch.stack(losses).mean()


def train_model(
    equations,
    variables,
    domains,
    nb_iter=500,
    nb_samples=1000,
    lr=0.0001,
    lr_gamma=0.99,
    hidden_layers=4,
    hidden_features=256,
    first_omega_0=10.0,
    hidden_omega_0=30.0,
    device="cpu",
    progress=True,
    iteration_callback=None,
):
    """Train and return a SIREN that approximates the parsed variables.

    ``iteration_callback`` receives ``(iteration, model, loss_value)`` after
    each optimizer step. It is used by the CLI for optional frame collection
    without coupling the reusable training loop to GIF generation.
    """
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=lr_gamma
    )
    iterations = (
        trange(nb_iter, desc="Solving equation(s)") if progress else range(nb_iter)
    )

    model.train()
    for iteration in iterations:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(
            equations,
            domains,
            model,
            field_indices,
            nb_samples,
            device,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at iteration {iteration}"
            )
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = loss.item()
        if progress:
            iterations.set_postfix(loss=loss_value)
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
    parser.add_argument("--nb_iter", type=int, default=500, help="number of iterations")
    parser.add_argument(
        "--nb_samples", type=int, default=1000, help="number of uniform samples"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
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
        "--omega_0", type=float, default=10.0, help="first omega_0 of siren"
    )
    parser.add_argument(
        "--resolution", type=int, default=128, help="image resolution for the gif"
    )
    parser.add_argument(
        "--nb_frames", type=int, default=50, help="number of frames for the gif"
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    parser.add_argument(
        "--no_gif", action="store_true", help="skip GIF frame collection and export"
    )
    args = parser.parse_args()

    equations, variables, domains, solution_functions = parse_problem_file(
        args.input_file
    )
    frames = []

    def collect_frame(iteration, model, loss_value):
        if iteration % max(1, (args.nb_iter // args.nb_frames)) == 0:
            xy = torch.cartesian_prod(
                torch.linspace(0, 1, args.resolution),
                torch.linspace(0, 1, args.resolution),
            ).to(args.device)
            with torch.no_grad():
                out = model(xy[:, 0:1], xy[:, 1:2])
            out = out.view(args.resolution, args.resolution, out.size(-1))
            out = out.rot90().cpu().numpy()
            frames.append(out)

    model = train_model(
        equations,
        variables,
        domains,
        nb_iter=args.nb_iter,
        nb_samples=args.nb_samples,
        lr=args.lr,
        lr_gamma=args.lr_gamma,
        hidden_layers=args.hidden_layers,
        hidden_features=args.hidden_features,
        first_omega_0=args.omega_0,
        device=args.device,
        iteration_callback=None if args.no_gif else collect_frame,
    )

    if not args.no_gif:
        solution = None
        if solution_functions:
            xy = torch.cartesian_prod(
                torch.linspace(0, 1, args.resolution),
                torch.linspace(0, 1, args.resolution),
            ).to(args.device)
            with torch.no_grad():
                solution = evaluate_solution_functions(
                    solution_functions,
                    variables,
                    xy[:, 0],
                    xy[:, 1],
                )
            solution = solution.view(
                args.resolution, args.resolution, len(variables)
            )
            solution = solution.rot90().cpu().numpy()
        make_gif(frames, variables, args.output_file, solution=solution)
