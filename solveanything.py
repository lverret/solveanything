import argparse
import ast
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
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


FUNS = {
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "exp": exp,
    "abs": abs,
    "tanh": tanh,
    "image": image,
    "grad": grad,
}

# -----------------------------------------------------------------------------
# Parser functions


def parse_equations(equations):
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
        print(log[:-2] + f",  {formula}'")
    variables = list(variables.keys())
    print(f"Found {len(variables)} unknown function(s) to approximate: {variables}")
    return variables, domains


def parse(formula, node, variables, fixed_coordinates):
    if isinstance(node, ast.Constant):
        return variables
    elif isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return parse(formula, node.operand, variables, fixed_coordinates)
    elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
        parse(formula, node.left, variables, fixed_coordinates)
        parse(formula, node.right, variables, fixed_coordinates)
        return variables
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in FUNS:
            if len(node.args) != len(signature(FUNS[node.func.id]).parameters):
                raise InvalidFormula(
                    formula, f"Invalid nb of args for '{node.func.id}'"
                )
            for arg in node.args:
                parse(formula, arg, variables, fixed_coordinates)
            return variables
        elif isinstance(node.func, ast.Name) and node.func.id not in FUNS:
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
        if node.id in ["x", "y"]:
            return variables
        else:
            parse(
                formula,
                ast.Call(
                    func=ast.Name(id=node.id, ctx=ast.Load()),
                    args=[
                        ast.Name(id="x", ctx=ast.Load()),
                        ast.Name(id="y", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
                variables,
                fixed_coordinates,
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
        if node.id in ["x", "y"]:
            _record_coordinate(used_coordinates, node.id, samples[node.id])
            return samples[node.id]
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


def make_gif(frames, vars):
    ims = []
    nr = int(np.sqrt(len(vars)))
    nc = len(vars) // nr + len(vars) % nr
    fig, axs = plt.subplots(nr, nc, figsize=(4.8 * nc, 4.0 * nr))
    if nr == 1:
        axs = np.array((axs,))
    if nc == 1:
        axs = np.array((axs,))
    for k, var in enumerate(vars):
        i, j = k // nc, k % nc
        ims.append(axs[i, j].imshow(frames[0][:, :, k], extent=(0, 1, 0, 1)))
        fig.colorbar(ims[k], ax=axs[i, j])
        axs[i, j].set_xlabel("x")
        axs[i, j].set_ylabel("y")
        axs[i, j].set_title(var)
        axs[i, j].margins(0)
    for k in range(len(vars), nc * nr):
        i, j = k // nc, k % nc
        fig.delaxes(axs[i][j])
    fig.tight_layout()

    def animate(i):
        out = frames[i]
        for k in range(out.shape[-1]):
            z = out[:, :, k]
            ims[k].set_data(z)
            ims[k].set_clim(z.min(), z.max())

    ani = FuncAnimation(fig, animate, frames=len(frames))
    pbar = trange(len(frames), desc="Generating GIF")
    ani.save(
        args.output_file,
        writer=PillowWriter(fps=len(frames) / 3),
        progress_callback=lambda i, n: pbar.update(1),
    )


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", "-i", type=str, help="input filename with one equation per line"
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

    with open(args.input_file, "r") as f:
        data = f.read()
    equations = data.splitlines()

    vars, domains = parse_equations(equations)
    field_indices = {var: index for index, var in enumerate(vars)}

    model = Siren(
        in_features=2,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=len(vars),
        first_omega_0=args.omega_0,
        hidden_omega_0=30.0,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        if args.lr_gamma != 1.0
        else None
    )

    pbar = trange(args.nb_iter, desc="Solving equation(s)")
    frames = []

    for it in pbar:
        loss = compute_loss(
            equations,
            domains,
            model,
            field_indices,
            args.nb_samples,
            args.device,
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        pbar.set_postfix(loss=loss.item())
        if not args.no_gif and it % max(1, (args.nb_iter // args.nb_frames)) == 0:
            xy = torch.cartesian_prod(
                torch.linspace(0, 1, args.resolution),
                torch.linspace(0, 1, args.resolution),
            ).to(args.device)
            out = model(xy[:, 0:1], xy[:, 1:2])
            out = out.view(args.resolution, args.resolution, out.size(-1))
            out = out.rot90().cpu().data.numpy()
            frames.append(out)

    if not args.no_gif:
        make_gif(frames, vars)
