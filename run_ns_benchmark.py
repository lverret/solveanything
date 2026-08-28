#!/usr/bin/env python3
"""Benchmark PINN-like neural solvers on classical 2D Navier--Stokes cases.

Each run is a Cartesian product of a physical problem and a compatible
approach.  A problem owns its equation file, reference metric, geometry, and
visualisation metadata; an approach owns only model/training choices.
Reference data are used after training and never enter a loss or stopping rule.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, PillowWriter

from solveanything import (
    CollocationSampler,
    build_model,
    compile_residuals,
    compute_equation_losses,
    coordinate_names_from_domains,
    domain_dimension,
    evaluate_solution_functions,
    is_interior_domain,
    make_static_plot,
    parse_problem_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "benchmark_results" / "navier_stokes"
MAX_SECONDS_ALLOWED = 200.0
MAX_VRAM_GB_ALLOWED = 8.0
RESULT_SCHEMA_VERSION = 2


def find_ffmpeg_executable():
    """Return an available FFmpeg binary without requiring a system install."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (OSError, RuntimeError):
        return None
    if executable and Path(executable).is_file() and os.access(executable, os.X_OK):
        return executable
    return None


def artifact_suffix(problem):
    """Choose a portable final-artifact format for a physical problem."""
    if "t" not in problem["bounds"]:
        return ".png"
    return ".mp4" if find_ffmpeg_executable() else ".gif"


# Ghia, Ghia & Shin (1982), Tables I and II.  The Re=400 value at x=0.9063
# is retained exactly as published, despite the frequently noted likely typo.
GHIA_U_COORDINATES = np.array(
    [
        1.0000,
        0.9766,
        0.9688,
        0.9609,
        0.9531,
        0.8516,
        0.7344,
        0.6172,
        0.5000,
        0.4531,
        0.2813,
        0.1719,
        0.1016,
        0.0703,
        0.0625,
        0.0547,
        0.0000,
    ],
    dtype=np.float32,
)
GHIA_V_COORDINATES = np.array(
    [
        1.0000,
        0.9688,
        0.9609,
        0.9531,
        0.9453,
        0.9063,
        0.8594,
        0.8047,
        0.5000,
        0.2344,
        0.2266,
        0.1563,
        0.0938,
        0.0781,
        0.0703,
        0.0625,
        0.0000,
    ],
    dtype=np.float32,
)
GHIA_U = {
    100: np.array(
        [
            1.00000,
            0.84123,
            0.78871,
            0.73722,
            0.68717,
            0.23151,
            0.00332,
            -0.13641,
            -0.20581,
            -0.21090,
            -0.15662,
            -0.10150,
            -0.06434,
            -0.04775,
            -0.04192,
            -0.03717,
            0.00000,
        ],
        dtype=np.float32,
    ),
    400: np.array(
        [
            1.00000,
            0.75837,
            0.68439,
            0.61756,
            0.55892,
            0.29093,
            0.16256,
            0.02135,
            -0.11477,
            -0.17119,
            -0.32726,
            -0.24299,
            -0.14612,
            -0.10338,
            -0.09266,
            -0.08186,
            0.00000,
        ],
        dtype=np.float32,
    ),
    1000: np.array(
        [
            1.00000,
            0.65928,
            0.57492,
            0.51117,
            0.46604,
            0.33304,
            0.18719,
            0.05702,
            -0.06080,
            -0.10648,
            -0.27805,
            -0.38289,
            -0.29730,
            -0.22220,
            -0.20196,
            -0.18109,
            0.00000,
        ],
        dtype=np.float32,
    ),
}
GHIA_V = {
    100: np.array(
        [
            0.00000,
            -0.05906,
            -0.07391,
            -0.08864,
            -0.10313,
            -0.16914,
            -0.22445,
            -0.24533,
            0.05454,
            0.17527,
            0.17507,
            0.16077,
            0.12317,
            0.10890,
            0.10091,
            0.09233,
            0.00000,
        ],
        dtype=np.float32,
    ),
    400: np.array(
        [
            0.00000,
            -0.12146,
            -0.15663,
            -0.19254,
            -0.22847,
            -0.23827,
            -0.44993,
            -0.38598,
            0.05186,
            0.30174,
            0.30203,
            0.28124,
            0.22965,
            0.20920,
            0.19713,
            0.18360,
            0.00000,
        ],
        dtype=np.float32,
    ),
    1000: np.array(
        [
            0.00000,
            -0.21388,
            -0.27669,
            -0.33714,
            -0.39188,
            -0.51500,
            -0.42665,
            -0.31966,
            0.02526,
            0.32235,
            0.33075,
            0.37095,
            0.32627,
            0.30353,
            0.29012,
            0.27485,
            0.00000,
        ],
        dtype=np.float32,
    ),
}


DEFAULT_TRAINING = {
    "architecture": "siren",
    "hidden_features": 256,
    "hidden_layers": 4,
    "first_omega_0": 3.0,
    "hidden_omega_0": 3.0,
    "activation": "tanh",
    "fourier_features": 64,
    "fourier_sigma": 2.0,
    "samples": 2048,
    "boundary_samples": 512,
    "sampling": "iid",
    "grouped": True,
    "penalty": "mse",
    "huber_delta": 0.01,
    "loss_balance": "boundary_pde_ramp",
    "optimizer": "adam",
    "weight_decay": 1e-6,
    "lr": 3e-4,
    "min_lr": 1e-5,
    "lr_schedule": "cosine",
    "max_steps": 50000,
    "max_seconds": 180.0,
    "seed": 0,
    "residual_mode": "autodiff",
    "fd_resolution": 65,
    "fd_order": 2,
    "artifact_resolution": 96,
    "video_frames": 32,
    "video_fps": 8,
    "artifact_seconds_reserve": 15.0,
}


def built_in_problems():
    """Return physical cases and their post-training reference metrics."""
    return [
        {
            "id": "cavity_re100",
            "name": "Lid-driven cavity, Re=100",
            "family": "cavity",
            "reynolds": 100,
            "viscosity": 0.01,
            "input_file": "benchmarks/navier_stokes/lid_driven_cavity_re100.txt",
            "metric_name": "E_Ghia",
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "enabled": True,
        },
        {
            "id": "cavity_re400",
            "name": "Lid-driven cavity, Re=400",
            "family": "cavity",
            "reynolds": 400,
            "viscosity": 0.0025,
            "input_file": "benchmarks/navier_stokes/lid_driven_cavity_re400.txt",
            "metric_name": "E_Ghia",
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "enabled": True,
        },
        {
            "id": "cavity_re1000",
            "name": "Lid-driven cavity, Re=1000",
            "family": "cavity",
            "reynolds": 1000,
            "viscosity": 0.001,
            "input_file": "benchmarks/navier_stokes/lid_driven_cavity_re1000.txt",
            "metric_name": "E_Ghia",
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "enabled": True,
        },
        {
            "id": "cylinder_re20",
            "name": "DFG cylinder 2D-1, Re=20",
            "family": "cylinder",
            "reynolds": 20,
            "viscosity": 0.001,
            "input_file": "benchmarks/navier_stokes/cylinder_dfg_re20.txt",
            "metric_name": "E_DFG",
            "bounds": {"x": [0.0, 2.2], "y": [0.0, 0.41]},
            "obstacle": {"center": [0.2, 0.2], "radius": 0.05},
            "reference": {
                "C_D": 5.57953523384,
                "C_L": 0.010618948146,
                "delta_p": 0.11752016697,
            },
            "mean_velocity": 0.2,
            "enabled": True,
        },
        {
            "id": "cylinder_re100",
            "name": "DFG cylinder 2D-3, Re=100",
            "family": "cylinder",
            "reynolds": 100,
            "viscosity": 0.001,
            "input_file": "benchmarks/navier_stokes/cylinder_dfg_re100.txt",
            "metric_name": "E_DFG",
            "bounds": {"x": [0.0, 2.2], "y": [0.0, 0.41], "t": [0.0, 8.0]},
            "obstacle": {"center": [0.2, 0.2], "radius": 0.05},
            "reference": {
                "C_D_max": 2.950921575,
                "C_L_max": 0.47795,
                "delta_p_t8": 0.1116,
            },
            "mean_velocity": 1.0,
            "enabled": True,
        },
        {
            "id": "taylor_green_re100",
            "name": "Taylor--Green vortex, Re=100",
            "family": "taylor_green",
            "reynolds": 100,
            "viscosity": 0.01,
            "input_file": "benchmarks/navier_stokes/taylor_green_re100.txt",
            "metric_name": "E_TGV",
            "bounds": {"x": [0.0, 2 * math.pi], "y": [0.0, 2 * math.pi], "t": [0.0, 1.0]},
            "enabled": True,
        },
        {
            "id": "taylor_green_re1000",
            "name": "Taylor--Green vortex, Re=1000",
            "family": "taylor_green",
            "reynolds": 1000,
            "viscosity": 0.001,
            "input_file": "benchmarks/navier_stokes/taylor_green_re1000.txt",
            "metric_name": "E_TGV",
            "bounds": {"x": [0.0, 2 * math.pi], "y": [0.0, 2 * math.pi], "t": [0.0, 1.0]},
            "enabled": True,
        },
    ]


def built_in_approaches():
    """Return architecture/training alternatives shared across problems."""
    return [
        {
            "id": "siren_low",
            "name": "Low-frequency SIREN (3, 3)",
            "config": {},
            "enabled": True,
        },
        {
            "id": "siren_compact",
            "name": "Compact SIREN, width 192/depth 3",
            "config": {"hidden_features": 192, "hidden_layers": 3},
            "enabled": True,
        },
        {
            "id": "tanh_mlp",
            "name": "Tanh MLP",
            "config": {"architecture": "mlp"},
            "enabled": True,
        },
        {
            "id": "fourier_mlp",
            "name": "Fourier-feature MLP, sigma 2",
            "config": {"architecture": "fourier_mlp", "fourier_sigma": 2.0},
            "enabled": True,
        },
        {
            "id": "modified_mlp",
            "name": "Modified gated MLP",
            "config": {"architecture": "modified_mlp"},
            "enabled": True,
        },
        {
            "id": "siren_fixed_sobol",
            "name": "Low-frequency SIREN + fixed Sobol points",
            "config": {"sampling": "fixed_sobol"},
            "enabled": True,
        },
        {
            "id": "siren_equal_groups",
            "name": "Low-frequency SIREN + equal boundary/PDE loss",
            "config": {"loss_balance": "equal_groups"},
            "enabled": True,
        },
        {
            "id": "fd_fourier",
            "name": "Structured finite differences + Fourier MLP",
            "families": ["cavity"],
            "config": {
                "architecture": "fourier_mlp",
                "residual_mode": "finite_difference",
            },
            "enabled": True,
        },
    ]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device(device_name, max_vram_gb):
    device = torch.device(device_name)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    limit_bytes = min(properties.total_memory, int(max_vram_gb * 1024**3))
    torch.cuda.set_per_process_memory_fraction(
        limit_bytes / properties.total_memory, device
    )
    return device


def training_progress(step, started, config, training_deadline):
    step_fraction = step / max(1, int(config["max_steps"]))
    duration = max(1e-6, training_deadline - started)
    time_fraction = (time.monotonic() - started) / duration
    return min(1.0, max(step_fraction, time_fraction))


def combine_losses(equation_losses, domains, strategy, progress):
    boundary = [
        loss
        for loss, domain in zip(equation_losses, domains)
        if not is_interior_domain(domain)
    ]
    interior = [
        loss
        for loss, domain in zip(equation_losses, domains)
        if is_interior_domain(domain)
    ]
    if not boundary:
        return torch.stack(interior).mean()
    if not interior:
        return torch.stack(boundary).mean()
    boundary_loss = torch.stack(boundary).mean()
    pde_loss = torch.stack(interior).mean()
    if strategy == "equal_groups":
        return 0.5 * (boundary_loss + pde_loss)
    if strategy == "boundary_pde_ramp":
        pde_weight = min(1.0, 0.1 + 3.0 * progress)
        return (boundary_loss + pde_weight * pde_loss) / (1 + pde_weight)
    raise ValueError(f"Unknown loss balance {strategy!r}")


def fd_operators(field, spacing, order):
    """Second/fourth-order spatial stencils used by the cavity ablation."""
    if order == 2:
        center = field[1:-1, 1:-1]
        derivative_x = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2 * spacing)
        derivative_y = (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * spacing)
        laplacian = (
            field[2:, 1:-1]
            + field[:-2, 1:-1]
            + field[1:-1, 2:]
            + field[1:-1, :-2]
            - 4 * center
        ) / spacing**2
        return center, derivative_x, derivative_y, laplacian
    if order == 4:
        center = field[2:-2, 2:-2]
        derivative_x = (
            -field[4:, 2:-2]
            + 8 * field[3:-1, 2:-2]
            - 8 * field[1:-3, 2:-2]
            + field[:-4, 2:-2]
        ) / (12 * spacing)
        derivative_y = (
            -field[2:-2, 4:]
            + 8 * field[2:-2, 3:-1]
            - 8 * field[2:-2, 1:-3]
            + field[2:-2, :-4]
        ) / (12 * spacing)
        second_x = (
            -field[4:, 2:-2]
            + 16 * field[3:-1, 2:-2]
            - 30 * center
            + 16 * field[1:-3, 2:-2]
            - field[:-4, 2:-2]
        ) / (12 * spacing**2)
        second_y = (
            -field[2:-2, 4:]
            + 16 * field[2:-2, 3:-1]
            - 30 * center
            + 16 * field[2:-2, 1:-3]
            - field[2:-2, :-4]
        ) / (12 * spacing**2)
        return center, derivative_x, derivative_y, second_x + second_y
    raise ValueError("fd_order must be 2 or 4")


def cavity_fd_loss(model, field_indices, problem, config, state):
    resolution = int(config["fd_resolution"])
    if "fd_coordinates" not in state:
        axis = torch.linspace(0, 1, resolution, device=state["device"])
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        state["fd_coordinates"] = (x.reshape(-1, 1), y.reshape(-1, 1))
    x, y = state["fd_coordinates"]
    output = model(x, y).reshape(resolution, resolution, -1)
    u = output[:, :, field_indices["u"]]
    v = output[:, :, field_indices["v"]]
    p = output[:, :, field_indices["p"]]
    spacing = 1 / (resolution - 1)
    order = int(config["fd_order"])
    uc, ux, uy, ulap = fd_operators(u, spacing, order)
    vc, vx, vy, vlap = fd_operators(v, spacing, order)
    _, px, py, _ = fd_operators(p, spacing, order)
    nu = float(problem["viscosity"])
    pde = torch.stack(
        [
            (uc * ux + vc * uy + px - nu * ulap).square().mean(),
            (uc * vx + vc * vy + py - nu * vlap).square().mean(),
            (ux + vy).square().mean(),
        ]
    ).mean()
    boundary = torch.stack(
        [
            u[0, :].square().mean(),
            u[-1, :].square().mean(),
            u[:, 0].square().mean(),
            (u[:, -1] - 1).square().mean(),
            v[0, :].square().mean(),
            v[-1, :].square().mean(),
            v[:, 0].square().mean(),
            v[:, -1].square().mean(),
            p[0, 0].square(),
        ]
    ).mean()
    return 0.5 * (boundary + pde)


def build_run_model(config, variables, coordinate_names, device):
    return build_model(
        config["architecture"],
        in_features=len(coordinate_names),
        hidden_features=int(config["hidden_features"]),
        hidden_layers=int(config["hidden_layers"]),
        out_features=len(variables),
        first_omega_0=float(config["first_omega_0"]),
        hidden_omega_0=float(config["hidden_omega_0"]),
        activation=config["activation"],
        fourier_features=int(config["fourier_features"]),
        fourier_sigma=float(config["fourier_sigma"]),
        coordinate_names=coordinate_names,
    ).to(device)


def train_one(problem, config, equations, variables, domains, device, deadline):
    set_seed(int(config["seed"]))
    coordinate_names = coordinate_names_from_domains(domains)
    field_indices = {name: index for index, name in enumerate(variables)}
    model = build_run_model(config, variables, coordinate_names, device)
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
    else:
        raise ValueError("optimizer must be adam or adamw")
    sampler = CollocationSampler(method=config["sampling"], seed=config["seed"])
    residuals = None
    if config["residual_mode"] == "autodiff":
        residuals = compile_residuals(equations, field_indices, model)
    elif config["residual_mode"] != "finite_difference":
        raise ValueError(f"Unknown residual mode {config['residual_mode']!r}")

    reserve = float(config["artifact_seconds_reserve"])
    training_deadline = deadline - reserve
    if training_deadline <= time.monotonic():
        raise ValueError("Runtime budget is too small for the artifact reserve")
    started = time.monotonic()
    state = {"device": device}
    final_loss = math.nan
    steps = 0
    model.train()
    for step in range(int(config["max_steps"])):
        if time.monotonic() >= training_deadline:
            break
        progress = training_progress(step, started, config, training_deadline)
        optimizer.zero_grad(set_to_none=True)
        if config["residual_mode"] == "finite_difference":
            loss = cavity_fd_loss(model, field_indices, problem, config, state)
        else:
            sample_counts = []
            for domain in domains:
                if domain_dimension(domain) == 0:
                    sample_counts.append(1)
                elif is_interior_domain(domain):
                    sample_counts.append(int(config["samples"]))
                else:
                    sample_counts.append(int(config["boundary_samples"]))
            equation_losses, _ = compute_equation_losses(
                residuals,
                domains,
                model,
                field_indices,
                sample_counts,
                device,
                sampler=sampler,
                grouped=bool(config["grouped"]),
                penalty=config["penalty"],
                huber_delta=float(config["huber_delta"]),
            )
            loss = combine_losses(
                equation_losses, domains, config["loss_balance"], progress
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        if config["lr_schedule"] == "cosine":
            lr = float(config["min_lr"]) + 0.5 * (
                float(config["lr"]) - float(config["min_lr"])
            ) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
        elif config["lr_schedule"] != "constant":
            raise ValueError("lr_schedule must be cosine or constant")
        final_loss = float(loss.detach())
        steps = step + 1
    return model, field_indices, {
        "steps": steps,
        "training_seconds": time.monotonic() - started,
        "final_loss": final_loss,
    }


def relative_l2(prediction, reference):
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(
        reference
    ).clamp_min(1e-12)


@torch.no_grad()
def ghia_metric(model, field_indices, problem, device):
    reynolds = int(problem["reynolds"])
    y = torch.as_tensor(GHIA_U_COORDINATES, device=device).view(-1, 1)
    x_mid = torch.full_like(y, 0.5)
    predicted_u = model(x_mid, y)[:, field_indices["u"]]
    x = torch.as_tensor(GHIA_V_COORDINATES, device=device).view(-1, 1)
    y_mid = torch.full_like(x, 0.5)
    predicted_v = model(x, y_mid)[:, field_indices["v"]]
    reference_u = torch.as_tensor(GHIA_U[reynolds], device=device)
    reference_v = torch.as_tensor(GHIA_V[reynolds], device=device)
    error_u = relative_l2(predicted_u, reference_u)
    error_v = relative_l2(predicted_v, reference_v)
    return float(0.5 * (error_u + error_v)), {
        "relative_l2_u": float(error_u),
        "relative_l2_v": float(error_v),
    }


def cylinder_forces(model, field_indices, problem, device, times):
    obstacle = problem["obstacle"]
    cx, cy = obstacle["center"]
    radius = float(obstacle["radius"])
    angle_count = 256
    theta = torch.arange(angle_count, device=device, dtype=torch.float32)
    theta = 2 * math.pi * theta / angle_count
    time_values = torch.as_tensor(times, device=device, dtype=torch.float32)
    theta = theta.repeat(time_values.numel())
    repeated_times = time_values.repeat_interleave(angle_count).view(-1, 1)
    normal_x = torch.cos(theta).view(-1, 1)
    normal_y = torch.sin(theta).view(-1, 1)
    x = (cx + radius * normal_x).detach().requires_grad_(True)
    y = (cy + radius * normal_y).detach().requires_grad_(True)
    coordinates = [x, y]
    if "t" in problem["bounds"]:
        coordinates.append(repeated_times)
    with torch.enable_grad():
        output = model(*coordinates)
        u = output[:, field_indices["u"] : field_indices["u"] + 1]
        v = output[:, field_indices["v"] : field_indices["v"] + 1]
        p = output[:, field_indices["p"] : field_indices["p"] + 1]
        ux, uy = torch.autograd.grad(
            u,
            (x, y),
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
        )
        vx, vy = torch.autograd.grad(
            v,
            (x, y),
            grad_outputs=torch.ones_like(v),
        )
    nu = float(problem["viscosity"])
    sigma_xx = -p + 2 * nu * ux
    sigma_xy = nu * (uy + vx)
    sigma_yy = -p + 2 * nu * vy
    traction_x = sigma_xx * normal_x + sigma_xy * normal_y
    traction_y = sigma_xy * normal_x + sigma_yy * normal_y
    arc_length = 2 * math.pi * radius
    forces_x = traction_x.reshape(-1, angle_count).mean(dim=1) * arc_length
    forces_y = traction_y.reshape(-1, angle_count).mean(dim=1) * arc_length
    diameter = 2 * radius
    scale = 2 / (float(problem["mean_velocity"]) ** 2 * diameter)
    return scale * forces_x.detach(), scale * forces_y.detach()


@torch.no_grad()
def cylinder_pressure_difference(model, field_indices, problem, device, time_value=None):
    x = torch.tensor([[0.15], [0.25]], device=device)
    y = torch.full_like(x, 0.2)
    coordinates = [x, y]
    if time_value is not None:
        coordinates.append(torch.full_like(x, float(time_value)))
    pressure = model(*coordinates)[:, field_indices["p"]]
    return float(pressure[0] - pressure[1])


def cylinder_metric(model, field_indices, problem, device):
    reference = problem["reference"]
    if "t" in problem["bounds"]:
        times = np.linspace(
            problem["bounds"]["t"][0], problem["bounds"]["t"][1], 161
        )
        drag, lift = cylinder_forces(
            model, field_indices, problem, device, times
        )
        diagnostics = {
            "C_D_max": float(drag.abs().max()),
            "C_L_max": float(lift.abs().max()),
            "delta_p_t8": cylinder_pressure_difference(
                model, field_indices, problem, device, time_value=8.0
            ),
        }
    else:
        drag, lift = cylinder_forces(
            model, field_indices, problem, device, [0.0]
        )
        diagnostics = {
            "C_D": float(drag.abs()[0]),
            "C_L": float(lift.abs()[0]),
            "delta_p": cylinder_pressure_difference(
                model, field_indices, problem, device
            ),
        }
    errors = [
        abs(diagnostics[name] - target) / max(abs(target), 1e-12)
        for name, target in reference.items()
    ]
    diagnostics["component_relative_errors"] = dict(zip(reference, errors))
    return float(np.mean(errors)), diagnostics


@torch.no_grad()
def taylor_green_metric(
    model, field_indices, problem, device, variables, solution_functions
):
    if not solution_functions:
        raise ValueError("Taylor--Green metric requires a '# Solution' section")
    axes = {
        "x": torch.linspace(*problem["bounds"]["x"], 32, device=device),
        "y": torch.linspace(*problem["bounds"]["y"], 32, device=device),
        "t": torch.linspace(*problem["bounds"]["t"], 9, device=device),
    }
    x, y, t = torch.meshgrid(axes["x"], axes["y"], axes["t"], indexing="ij")
    samples = {
        "x": x.reshape(-1, 1),
        "y": y.reshape(-1, 1),
        "t": t.reshape(-1, 1),
    }
    prediction = model(*samples.values())
    reference = evaluate_solution_functions(solution_functions, variables, samples)
    errors = {}
    for field in ("u", "v"):
        index = field_indices[field]
        errors[field] = float(relative_l2(prediction[:, index], reference[:, index]))
    return float(np.mean(list(errors.values()))), {
        "relative_l2_u": errors["u"],
        "relative_l2_v": errors["v"],
    }


def evaluate_metric(
    model, field_indices, problem, device, variables, solution_functions
):
    model.eval()
    if problem["family"] == "cavity":
        return ghia_metric(model, field_indices, problem, device)
    if problem["family"] == "cylinder":
        return cylinder_metric(model, field_indices, problem, device)
    if problem["family"] == "taylor_green":
        return taylor_green_metric(
            model,
            field_indices,
            problem,
            device,
            variables,
            solution_functions,
        )
    raise ValueError(f"Unknown problem family {problem['family']!r}")


def evaluate_spatial_frame(model, problem, variables, device, resolution, time_value=None):
    x_axis = torch.linspace(*problem["bounds"]["x"], resolution, device=device)
    y_axis = torch.linspace(*problem["bounds"]["y"], resolution, device=device)
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    coordinates = [x.reshape(-1, 1), y.reshape(-1, 1)]
    if "t" in problem["bounds"]:
        coordinates.append(torch.full_like(coordinates[0], float(time_value)))
    model.eval()
    with torch.no_grad():
        frame = model(*coordinates).reshape(resolution, resolution, len(variables))
    frame = frame.cpu().numpy()
    if problem.get("obstacle"):
        cx, cy = problem["obstacle"]["center"]
        radius = problem["obstacle"]["radius"]
        mask = (x - cx).square() + (y - cy).square() <= radius**2
        frame[mask.cpu().numpy(), :] = np.nan
    return frame


def save_video(model, problem, variables, device, config, output_file):
    resolution = int(config["artifact_resolution"])
    times = np.linspace(
        problem["bounds"]["t"][0],
        problem["bounds"]["t"][1],
        int(config["video_frames"]),
    )
    frames = [
        evaluate_spatial_frame(
            model, problem, variables, device, resolution, time_value=value
        )
        for value in times
    ]
    limits = []
    for index in range(len(variables)):
        values = np.concatenate([frame[:, :, index].ravel() for frame in frames])
        finite = values[np.isfinite(values)]
        lower, upper = float(finite.min()), float(finite.max())
        if lower == upper:
            padding = max(0.5, abs(lower) * 0.05)
            lower, upper = lower - padding, upper + padding
        limits.append((lower, upper))

    fig, axes = plt.subplots(
        1, len(variables), squeeze=False, figsize=(4.8 * len(variables), 4.0)
    )
    images = []
    extent = (
        *problem["bounds"]["x"],
        *problem["bounds"]["y"],
    )
    for index, variable in enumerate(variables):
        axis = axes[0, index]
        image = axis.imshow(
            frames[0][:, :, index],
            extent=extent,
            origin="lower",
            vmin=limits[index][0],
            vmax=limits[index][1],
        )
        fig.colorbar(image, ax=axis)
        axis.set_title(f"{variable}, t={times[0]:.3g}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        images.append(image)
    fig.tight_layout()
    if output_file.suffix.lower() == ".mp4":
        executable = find_ffmpeg_executable()
        if executable is None:
            raise RuntimeError(
                "MP4 output requires FFmpeg; use a .gif output path when it is "
                "unavailable"
            )
        matplotlib.rcParams["animation.ffmpeg_path"] = executable
        writer = FFMpegWriter(fps=int(config["video_fps"]), codec="h264")
    elif output_file.suffix.lower() == ".gif":
        writer = PillowWriter(fps=int(config["video_fps"]))
    else:
        raise ValueError("Temporal artifacts must use an .mp4 or .gif suffix")
    with writer.saving(fig, str(output_file), dpi=120):
        for time_value, frame in zip(times, frames):
            for index, image in enumerate(images):
                image.set_data(frame[:, :, index])
                axes[0, index].set_title(
                    f"{variables[index]}, t={time_value:.3g}"
                )
            writer.grab_frame()
    plt.close(fig)


def save_artifact(model, problem, variables, device, config, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f"{output_file.stem}.tmp{output_file.suffix}")
    if "t" in problem["bounds"]:
        save_video(model, problem, variables, device, config, temporary)
    else:
        frame = evaluate_spatial_frame(
            model,
            problem,
            variables,
            device,
            int(config["artifact_resolution"]),
        )
        extent = (
            *problem["bounds"]["x"],
            *problem["bounds"]["y"],
        )
        make_static_plot(frame, variables, temporary, extent=extent)
    temporary.replace(output_file)


def problem_input_path(problem):
    path = Path(problem["input_file"])
    return path if path.is_absolute() else SCRIPT_DIR / path


def validate_problem(problem):
    required = {"id", "name", "family", "input_file", "metric_name", "bounds"}
    missing = sorted(required - set(problem))
    if missing:
        raise ValueError(f"Problem {problem.get('id', '?')} lacks {missing}")
    if problem["family"] not in {"cavity", "cylinder", "taylor_green"}:
        raise ValueError(f"Unknown problem family {problem['family']!r}")
    path = problem_input_path(problem)
    if not path.is_file():
        raise ValueError(f"Missing equation file: {path}")


def validate_config(config):
    if not 0 < float(config["max_seconds"]) <= MAX_SECONDS_ALLOWED:
        raise ValueError(f"max_seconds must be in (0, {MAX_SECONDS_ALLOWED}]")
    for name in (
        "samples",
        "boundary_samples",
        "hidden_features",
        "max_steps",
    ):
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("artifact_resolution", "video_frames"):
        if int(config[name]) < 2:
            raise ValueError(f"{name} must be at least 2")
    if int(config["video_fps"]) < 1:
        raise ValueError("video_fps must be positive")
    if float(config["artifact_seconds_reserve"]) <= 0:
        raise ValueError("artifact_seconds_reserve must be positive")
    if float(config["artifact_seconds_reserve"]) >= float(config["max_seconds"]):
        raise ValueError("artifact_seconds_reserve must be below max_seconds")
    if config["penalty"] not in {"mse", "mae", "pseudo_huber"}:
        raise ValueError("Unknown penalty")
    if config["loss_balance"] not in {"boundary_pde_ramp", "equal_groups"}:
        raise ValueError("Unknown loss balance")
    if config["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("optimizer must be adam or adamw")
    if config["sampling"] not in {"iid", "sobol", "fixed_sobol", "wall_mixture"}:
        raise ValueError("Unknown sampling method")
    if config["residual_mode"] not in {"autodiff", "finite_difference"}:
        raise ValueError("Unknown residual mode")
    if int(config["fd_order"]) not in {2, 4}:
        raise ValueError("fd_order must be 2 or 4")
    minimum_fd_resolution = 5 if int(config["fd_order"]) == 2 else 7
    if int(config["fd_resolution"]) < minimum_fd_resolution:
        raise ValueError(
            f"fd_resolution must be at least {minimum_fd_resolution}"
        )


def compatible(problem, approach):
    return not approach.get("families") or problem["family"] in approach["families"]


def run_id(problem, approach):
    return f"{problem['id']}__{approach['id']}"


def fingerprint(problem, approach, config):
    digest = hashlib.sha256()
    digest.update(json.dumps(problem, sort_keys=True).encode())
    digest.update(json.dumps(approach, sort_keys=True).encode())
    digest.update(json.dumps(config, sort_keys=True).encode())
    digest.update(problem_input_path(problem).read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update((SCRIPT_DIR / "solveanything.py").read_bytes())
    return digest.hexdigest()[:16]


def run_worker(spec_path):
    with open(spec_path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    result_path = Path(spec["result_path"])
    result = {"status": "error"}
    device = None
    model = None
    try:
        problem = spec["problem"]
        config = spec["config"]
        equations, variables, domains, solution_functions = parse_problem_file(
            problem["input_file"], verbose=False
        )
        if set(variables) != {"u", "v", "p"}:
            raise ValueError("Navier--Stokes cases must approximate u, v, and p")
        device = configure_device(spec["device"], float(spec["max_vram_gb"]))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model, field_indices, training = train_one(
            problem,
            config,
            equations,
            variables,
            domains,
            device,
            float(spec["deadline"]),
        )
        metric, diagnostics = evaluate_metric(
            model,
            field_indices,
            problem,
            device,
            variables,
            solution_functions,
        )
        artifact_path = Path(spec["artifact_path"])
        save_artifact(
            model, problem, variables, device, config, artifact_path
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result.update(training)
        result.update(
            {
                "status": "ok",
                "metric_value": metric,
                "diagnostics": diagnostics,
                "artifact_file": str(artifact_path),
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        if device is not None and device.type == "cuda":
            result["peak_vram_gb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
        else:
            result["peak_vram_gb"] = 0.0
        del model
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
            handle.write("\n")
    return 0 if result["status"] == "ok" else 1


def execute_worker(
    problem, approach, config, output_dir, device_name, max_vram_gb
):
    worker_dir = output_dir / ".workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    identifier = run_id(problem, approach)
    token = f"{identifier}-{fingerprint(problem, approach, config)}"
    spec_path = worker_dir / f"{token}.spec.json"
    result_path = worker_dir / f"{token}.result.json"
    suffix = artifact_suffix(problem)
    artifact_path = output_dir / "artifacts" / problem["id"] / f"{approach['id']}{suffix}"

    def cleanup():
        for path in (spec_path, result_path):
            if path.exists():
                path.unlink()
        try:
            worker_dir.rmdir()
        except OSError:
            pass

    started = time.monotonic()
    worker_problem = copy.deepcopy(problem)
    worker_problem["input_file"] = str(problem_input_path(problem).resolve())
    spec = {
        "problem": worker_problem,
        "approach": approach,
        "config": config,
        "device": device_name,
        "max_vram_gb": max_vram_gb,
        "deadline": started + float(config["max_seconds"]),
        "result_path": str(result_path.resolve()),
        "artifact_path": str(artifact_path.resolve()),
    }
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle)
        handle.write("\n")
    if result_path.exists():
        result_path.unlink()
    if artifact_path.exists():
        artifact_path.unlink()
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        remaining = max(
            0.01, float(config["max_seconds"]) - (time.monotonic() - started)
        )
        stdout, stderr = process.communicate(timeout=remaining)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.communicate()
        cleanup()
        raise
    seconds = time.monotonic() - started
    if timed_out:
        worker_result = {
            "status": "timeout",
            "error": f"hard wall-clock limit of {config['max_seconds']}s reached",
            "peak_vram_gb": None,
        }
    elif result_path.exists():
        with open(result_path, "r", encoding="utf-8") as handle:
            worker_result = json.load(handle)
    else:
        detail = stderr.strip() or stdout.strip() or f"worker exited {process.returncode}"
        worker_result = {
            "status": "error",
            "error": detail,
            "peak_vram_gb": None,
        }
    cleanup()
    return worker_result, seconds


def load_configuration(path):
    problems = built_in_problems()
    approaches = built_in_approaches()
    defaults = copy.deepcopy(DEFAULT_TRAINING)
    run_config = {}
    if path is None:
        return problems, approaches, defaults, run_config
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    defaults.update(payload.get("defaults", {}))
    run_config.update(payload.get("run", {}))
    if payload.get("replace_problems"):
        problems = []
    if payload.get("replace_approaches"):
        approaches = []
    for key, collection in (("problems", problems), ("approaches", approaches)):
        by_id = {item["id"]: item for item in collection}
        for override in payload.get(key, []):
            identifier = override["id"]
            if identifier not in by_id:
                collection.append(copy.deepcopy(override))
                by_id[identifier] = collection[-1]
            else:
                if key == "approaches" and "config" in override:
                    by_id[identifier].setdefault("config", {}).update(
                        override["config"]
                    )
                by_id[identifier].update(
                    {name: value for name, value in override.items() if name != "config"}
                )
    return problems, approaches, defaults, run_config


def parse_set_values(values):
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--set expects KEY=JSON_VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def load_results(path):
    if not path.exists():
        return {"schema_version": RESULT_SCHEMA_VERSION, "runs": []}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Result schema changed; choose a new output directory")
    return payload


def upsert_result(results, result):
    results[:] = [item for item in results if item.get("run_id") != result["run_id"]]
    results.append(result)


def save_results(payload, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "results.json"
    temporary = output_dir / "results.json.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)
    columns = [
        "run_id",
        "problem_id",
        "problem_name",
        "approach_id",
        "approach_name",
        "metric_name",
        "metric_value",
        "status",
        "seconds",
        "training_seconds",
        "steps",
        "peak_vram_gb",
        "seed",
        "artifact_file",
        "fingerprint",
        "diagnostics",
        "error",
    ]
    with open(output_dir / "results.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in payload["runs"]:
            row = {column: result.get(column, "") for column in columns}
            if isinstance(row["diagnostics"], dict):
                row["diagnostics"] = json.dumps(row["diagnostics"], sort_keys=True)
            writer.writerow(row)


def print_catalog(problems, approaches):
    print("Problems")
    for problem in problems:
        suffix = " [disabled]" if not problem.get("enabled", True) else ""
        dimensions = "x,y,t" if "t" in problem["bounds"] else "x,y"
        print(
            f"  {problem['id']:<22} {dimensions:<5} {problem['metric_name']:<8} "
            f"{problem['name']}{suffix}"
        )
    print("\nApproaches")
    for approach in approaches:
        suffix = " [disabled]" if not approach.get("enabled", True) else ""
        family = ",".join(approach.get("families", ["all"]))
        print(f"  {approach['id']:<22} {family:<10} {approach['name']}{suffix}")


def print_rankings(results):
    successful = [item for item in results if item.get("status") == "ok"]
    for problem_id in sorted({item["problem_id"] for item in successful}):
        rows = sorted(
            (item for item in successful if item["problem_id"] == problem_id),
            key=lambda item: item["metric_value"],
        )
        print(f"\n{rows[0]['problem_name']} -- {rows[0]['metric_name']} (lower is better)")
        for rank, row in enumerate(rows, 1):
            print(
                f"  {rank:>2}. {row['approach_id']:<18} "
                f"{row['metric_value']:.6g}  {row['seconds']:.1f}s"
            )


def selected(collection, requested):
    enabled = [item for item in collection if item.get("enabled", True)]
    if not requested:
        return enabled
    requested = set(requested)
    known = {item["id"] for item in collection}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown IDs: {', '.join(unknown)}")
    return [item for item in enabled if item["id"] in requested]


def run(args):
    if args.worker_spec is not None:
        return run_worker(args.worker_spec)
    problems, approaches, defaults, file_run_config = load_configuration(args.config)
    for problem in problems:
        validate_problem(problem)

    if args.write_config:
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "replace_problems": False,
            "replace_approaches": False,
            "run": {
                "output_dir": str(DEFAULT_OUTPUT_DIR),
                "device": "cuda",
                "max_vram_gb": MAX_VRAM_GB_ALLOWED,
            },
            "defaults": defaults,
            "problems": problems,
            "approaches": approaches,
        }
        with open(args.write_config, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {len(problems)} problems and {len(approaches)} approaches")
        return 0
    if args.list:
        print_catalog(problems, approaches)
        return 0

    problem_ids = [item for value in args.problem for item in value.split(",") if item]
    approach_ids = [item for value in args.approach for item in value.split(",") if item]
    problems = selected(problems, problem_ids)
    approaches = selected(approaches, approach_ids)
    overrides = parse_set_values(args.set_values)
    if args.max_seconds is not None:
        overrides["max_seconds"] = args.max_seconds
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.samples is not None:
        overrides["samples"] = args.samples
    if args.seed is not None:
        overrides["seed"] = args.seed

    runs = []
    for problem in problems:
        for approach in approaches:
            if not compatible(problem, approach):
                continue
            config = copy.deepcopy(defaults)
            config.update(approach.get("config", {}))
            config.update(overrides)
            validate_config(config)
            runs.append((problem, approach, config))
    if any("t" in problem["bounds"] for problem, _, _ in runs):
        if find_ffmpeg_executable() is None:
            print(
                "FFmpeg was not found; temporal artifacts will be written as "
                "animated GIFs. Install imageio-ffmpeg to enable H.264 MP4 output."
            )
    if args.dry_run:
        print_catalog(problems, approaches)
        print(f"\n{len(runs)} compatible runs")
        for problem, approach, config in runs:
            print(f"  {run_id(problem, approach)}: {json.dumps(config, sort_keys=True)}")
        return 0

    output_dir = Path(
        args.output_dir or file_run_config.get("output_dir", DEFAULT_OUTPUT_DIR)
    )
    device_name = args.device or file_run_config.get("device", "cuda")
    max_vram_gb = float(
        args.max_vram_gb
        if args.max_vram_gb is not None
        else file_run_config.get("max_vram_gb", MAX_VRAM_GB_ALLOWED)
    )
    if not 0 < max_vram_gb <= MAX_VRAM_GB_ALLOWED:
        raise ValueError(f"max_vram_gb must be in (0, {MAX_VRAM_GB_ALLOWED}]")
    payload = load_results(output_dir / "results.json")
    results = payload["runs"]

    for problem, approach, config in runs:
        identifier = run_id(problem, approach)
        run_fingerprint = fingerprint(problem, approach, config)
        if args.resume and any(
            item.get("run_id") == identifier
            and item.get("status") == "ok"
            and item.get("fingerprint") == run_fingerprint
            for item in results
        ):
            print(f"{identifier}: already complete")
            continue
        print(
            f"{identifier}: {problem['name']} / {approach['name']} "
            f"({config['max_seconds']:.0f}s, seed {config['seed']})"
        )
        result = {
            "run_id": identifier,
            "problem_id": problem["id"],
            "problem_name": problem["name"],
            "approach_id": approach["id"],
            "approach_name": approach["name"],
            "metric_name": problem["metric_name"],
            "metric_value": None,
            "status": "error",
            "seconds": None,
            "steps": 0,
            "peak_vram_gb": 0.0,
            "seed": config["seed"],
            "fingerprint": run_fingerprint,
            "config": config,
        }
        worker_result, seconds = execute_worker(
            problem, approach, config, output_dir, device_name, max_vram_gb
        )
        result.update(worker_result)
        result["seconds"] = seconds
        upsert_result(results, result)
        save_results(payload, output_dir)
        if result["status"] == "ok":
            print(
                f"  {result['metric_name']}={result['metric_value']:.6g}, "
                f"{seconds:.1f}s, {result['steps']} steps, "
                f"peak={result['peak_vram_gb']:.3f} GiB"
            )
        else:
            print(f"  ERROR: {result.get('error', 'unknown error')}")
            if args.fail_fast:
                raise RuntimeError(f"{identifier} failed")
    print_rankings(results)
    print(f"\nResults: {output_dir / 'results.json'}")
    return 0


def make_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark PINN-like solvers on classical Navier--Stokes problems"
    )
    parser.add_argument("--config", type=Path, help="JSON configuration/manifest")
    parser.add_argument("--write-config", type=Path, help="write defaults and exit")
    parser.add_argument("--problem", action="append", default=[], help="problem ID(s)")
    parser.add_argument("--approach", action="append", default=[], help="approach ID(s)")
    parser.add_argument("--output-dir", help="result/artifact directory")
    parser.add_argument("--device", help="Torch device (default: cuda)")
    parser.add_argument("--max-vram-gb", type=float, help="GPU cap, at most 8")
    parser.add_argument("--max-seconds", type=float, help="per-run cap, at most 200")
    parser.add_argument("--max-steps", type=int, help="optimizer-step override")
    parser.add_argument("--samples", type=int, help="interior collocation override")
    parser.add_argument("--seed", type=int, help="seed override")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="override a training/artifact key; repeatable",
    )
    parser.add_argument("--list", action="store_true", help="list problems/approaches")
    parser.add_argument("--dry-run", action="store_true", help="resolve runs only")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip matching successful runs (default: true)",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(make_parser().parse_args()))
