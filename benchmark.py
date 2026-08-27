#!/usr/bin/env python3
"""Benchmark analytic solutions for every problem file in a folder."""

import argparse
import sys
from pathlib import Path

import torch

from solveanything import (
    evaluate_solution_functions,
    parse_problem_file,
    train_model,
)


DEFAULT_GRID_RESOLUTION = 81
DEFAULT_SEED = 0


def benchmark_problem(path, resolution, seed, device):
    """Train the default SIREN and return its aggregate absolute grid MSE."""
    equations, variables, domains, solution_functions = parse_problem_file(
        path, verbose=False
    )
    if not solution_functions:
        raise ValueError(f"{path}: benchmark requires a '# Solution' section")

    torch.manual_seed(seed)
    model = train_model(
        equations,
        variables,
        domains,
        device=device,
        progress=False,
    )

    coordinates = torch.cartesian_prod(
        torch.linspace(0.0, 1.0, resolution, device=device),
        torch.linspace(0.0, 1.0, resolution, device=device),
    )
    model.eval()
    with torch.no_grad():
        prediction = model(coordinates[:, 0:1], coordinates[:, 1:2])
        solution = evaluate_solution_functions(
            solution_functions,
            variables,
            coordinates[:, 0],
            coordinates[:, 1],
        )
    return torch.mean((prediction - solution) ** 2).item()


def print_ascii_table(rows):
    headers = ("File", "MSE")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def print_row(row):
        print(
            "| "
            + " | ".join(
                value.ljust(width) for value, width in zip(row, widths)
            )
            + " |"
        )

    print(separator)
    print_row(headers)
    print(separator)
    for row in rows:
        print_row(row)
    print(separator)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the default SIREN for every matching problem file and print "
            "its MSE against the '# Solution' formulas."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path("examples"),
        help="folder containing problem .txt files (default: examples)",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="filename glob used inside the folder (default: *.txt)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_GRID_RESOLUTION,
        help=f"evaluation-grid width (default: {DEFAULT_GRID_RESOLUTION})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Torch seed reset before each problem (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device used for training (default: cpu)"
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"not a directory: {args.folder}")
    if args.resolution < 2:
        parser.error("--resolution must be at least 2")

    problem_files = sorted(
        path for path in args.folder.glob(args.pattern) if path.is_file()
    )
    if not problem_files:
        parser.error(
            f"no files matching {args.pattern!r} found in {str(args.folder)!r}"
        )

    rows = []
    errors = []
    for path in problem_files:
        print(f"Benchmarking {path.name}...", file=sys.stderr, flush=True)
        try:
            mse = benchmark_problem(
                path,
                resolution=args.resolution,
                seed=args.seed,
                device=args.device,
            )
            rows.append((path.name, f"{mse:.8e}"))
        except Exception as error:
            rows.append((path.name, "ERROR"))
            errors.append((path, error))

    print_ascii_table(rows)
    if errors:
        print("\nErrors:", file=sys.stderr)
        for path, error in errors:
            print(f"- {path.name}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
