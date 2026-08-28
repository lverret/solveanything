#!/usr/bin/env python3
"""Run the follow-up Re=100 lid-driven-cavity PINN/INR benchmark.

The benchmark uses one ranking metric, ``E_Ghia``: the mean of the relative
L2 errors for the predicted vertical and horizontal velocity centerlines
against the Re=100 data in Ghia, Ghia & Shin (1982).
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
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from solveanything import (
    CollocationSampler,
    build_model,
    compile_residuals,
    compute_equation_losses,
    compute_loss,
    make_static_plot,
    parse_equations,
    parse_problem_file,
    reduce_residual,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "examples" / "37_lid_driven_cavity_re100.txt"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "benchmark_results" / "lid_driven_cavity_re100_followup"
)
MAX_SECONDS_ALLOWED = 200.0
MAX_VRAM_GB_ALLOWED = 8.0
FINAL_PLOT_RESOLUTION = 128
RESULT_SCHEMA_VERSION = 1

# Tables I and II at Re=100 from Ghia, Ghia & Shin (JCP 48, 1982).
GHIA_U_Y = np.array(
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
GHIA_U = np.array(
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
)
GHIA_V_X = np.array(
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
GHIA_V = np.array(
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
)


ROUND1_DEFAULT_TRAINING = {
    "architecture": "siren",
    "hidden_features": 256,
    "hidden_layers": 4,
    "first_omega_0": 10.0,
    "hidden_omega_0": 30.0,
    "activation": "tanh",
    "fourier_features": 64,
    "fourier_sigma": 2.0,
    "hard_constraints": False,
    "compile_model": False,
    "grouped": True,
    "sampling": "iid",
    "samples": 2048,
    "penalty": "mse",
    "huber_delta": 0.01,
    "loss_balance": "equal_groups",
    "relobralo_alpha": 0.999,
    "relobralo_temperature": 0.1,
    "relobralo_lookback_probability": 0.999,
    "optimizer": "adam",
    "lr": 1e-3,
    "min_lr": 1e-5,
    "lr_schedule": "cosine",
    "lr_gamma": 0.99,
    "max_steps": 20000,
    "max_seconds": 60.0,
    "seed": 0,
    "residual_mode": "autodiff",
    "fd_resolution": 65,
    "integral_continuity_weight": 0.0,
    "spatial_weighting": "none",
    "curriculum": "none",
    "rad_refresh": 25,
    "rad_exploration": 0.25,
    "legacy_exact": False,
}

DEFAULT_TRAINING = {
    **ROUND1_DEFAULT_TRAINING,
    "first_omega_0": 3.0,
    "hidden_omega_0": 3.0,
    "loss_balance": "boundary_pde_ramp",
    "lr": 3e-4,
    "max_steps": 20000,
    "max_seconds": 90.0,
    "boundary_samples": None,
    "fd_order": 2,
    "fd_formulation": "advective",
}


def _case(case_id, stage, name, config=None, inherit_stages=None, finalist_rank=None):
    config = dict(config or {})
    if stage == "C":
        config.setdefault("max_seconds", 75.0)
    elif stage == "D":
        config.setdefault("max_seconds", 90.0)
    return {
        "id": case_id,
        "stage": stage,
        "name": name,
        "enabled": True,
        "inherit_stages": list(inherit_stages or []),
        "finalist_rank": finalist_rank,
        "config": config,
    }


def round1_manifest():
    """Return the staged 3 + 10 + 8 + 11 + 9 experiment manifest."""
    cases = [
        _case(
            "A01",
            "A",
            "Exact master baseline",
            {
                "legacy_exact": True,
                "grouped": False,
                "samples": 1000,
                "penalty": "mae",
                "loss_balance": "legacy",
                "lr": 1e-4,
                "lr_schedule": "exponential",
                "max_steps": 500,
                "max_seconds": 200.0,
            },
        ),
        _case(
            "A02",
            "A",
            "Grouped samples and cached field evaluations",
            {
                "samples": 1000,
                "penalty": "mae",
                "loss_balance": "legacy",
                "lr": 1e-4,
                "lr_schedule": "exponential",
                "max_steps": 500,
                "max_seconds": 200.0,
            },
        ),
        _case(
            "A03",
            "A",
            "Grouped evaluation plus torch.compile",
            {
                "samples": 1000,
                "penalty": "mae",
                "loss_balance": "legacy",
                "lr": 1e-4,
                "lr_schedule": "exponential",
                "max_steps": 500,
                "max_seconds": 200.0,
                "compile_model": True,
            },
        ),
        _case("B01", "B", "Adam cosine, peak LR 3e-4", {"lr": 3e-4}),
        _case("B02", "B", "Adam cosine, peak LR 1e-3", {"lr": 1e-3}),
        _case("B03", "B", "Adam cosine, peak LR 3e-3", {"lr": 3e-3}),
        _case("B04", "B", "MAE residual penalty", {"penalty": "mae"}),
        _case(
            "B05",
            "B",
            "Pseudo-Huber residual penalty",
            {"penalty": "pseudo_huber"},
        ),
        _case(
            "B06",
            "B",
            "Legacy 0.9/0.1 equation weights with MSE",
            {"loss_balance": "legacy"},
        ),
        _case(
            "B07",
            "B",
            "ReLoBRaLo relative-loss balancing",
            {"loss_balance": "relobralo"},
        ),
        _case(
            "B08",
            "B",
            "Inverse-gradient-norm balancing",
            {"loss_balance": "gradient_norm"},
        ),
        _case(
            "B09",
            "B",
            "Adam then L-BFGS",
            {"optimizer": "adam_lbfgs", "sampling": "fixed_sobol"},
        ),
        _case(
            "B10",
            "B",
            "SOAP-style preconditioned Adam",
            {"optimizer": "soap", "lr": 3e-4},
        ),
        _case(
            "C01",
            "C",
            "Resampled scrambled Sobol",
            {"sampling": "sobol"},
            ["B"],
        ),
        _case(
            "C02",
            "C",
            "Fixed scrambled Sobol",
            {"sampling": "fixed_sobol"},
            ["B"],
        ),
        _case(
            "C03",
            "C",
            "Residual-adaptive distribution",
            {"sampling": "rad"},
            ["B"],
        ),
        _case(
            "C04",
            "C",
            "Residual-based point attention",
            {"sampling": "fixed_sobol", "spatial_weighting": "residual_attention"},
            ["B"],
        ),
        _case(
            "C05",
            "C",
            "Wall mixture with lid taper and SDF weighting",
            {"sampling": "wall_mixture", "spatial_weighting": "lid_taper_sdf"},
            ["B"],
        ),
        _case(
            "C06",
            "C",
            "Boundary-to-PDE loss ramp",
            {"loss_balance": "boundary_pde_ramp"},
            ["B"],
        ),
        _case(
            "C07",
            "C",
            "Reynolds and convection continuation",
            {"curriculum": "re_convection"},
            ["B"],
        ),
        _case(
            "C08",
            "C",
            "Integral continuity constraints",
            {"integral_continuity_weight": 0.1},
            ["B"],
        ),
        _case("D01", "D", "Current SIREN (10, 30)", {}, ["B", "C"]),
        _case(
            "D02",
            "D",
            "Low-frequency SIREN (3, 3)",
            {"first_omega_0": 3.0, "hidden_omega_0": 3.0},
            ["B", "C"],
        ),
        _case(
            "D03",
            "D",
            "Compact SIREN (5, 10), width 192",
            {
                "first_omega_0": 5.0,
                "hidden_omega_0": 10.0,
                "hidden_features": 192,
            },
            ["B", "C"],
        ),
        _case("D04", "D", "Tanh MLP", {"architecture": "mlp"}, ["B", "C"]),
        _case(
            "D05",
            "D",
            "Fourier-feature MLP, sigma 2",
            {"architecture": "fourier_mlp", "fourier_sigma": 2.0},
            ["B", "C"],
        ),
        _case(
            "D06",
            "D",
            "Fourier-feature MLP, sigma 5",
            {"architecture": "fourier_mlp", "fourier_sigma": 5.0},
            ["B", "C"],
        ),
        _case(
            "D07",
            "D",
            "Random-weight-factorized tanh MLP",
            {"architecture": "rwf_mlp"},
            ["B", "C"],
        ),
        _case(
            "D08",
            "D",
            "Modified gated MLP",
            {"architecture": "modified_mlp"},
            ["B", "C"],
        ),
        _case(
            "D09",
            "D",
            "PirateNet-style residual network",
            {"architecture": "piratenet", "hidden_layers": 3},
            ["B", "C"],
        ),
        _case(
            "D10",
            "D",
            "Multi-head MLP with compatible hard constraints",
            {"architecture": "multihead_mlp", "hard_constraints": True},
            ["B", "C"],
        ),
        _case(
            "D11",
            "D",
            "Finite-difference residual pilot",
            {"architecture": "fourier_mlp", "residual_mode": "finite_difference"},
            ["B", "C"],
        ),
    ]
    for rank in range(1, 4):
        for seed in range(3):
            number = (rank - 1) * 3 + seed + 1
            cases.append(
                _case(
                    f"E{number:02d}",
                    "E",
                    f"Finalist rank {rank}, seed {seed}",
                    {"seed": seed, "max_seconds": 180.0, "max_steps": 50000},
                    finalist_rank=rank,
                )
            )
    assert len(cases) == 41
    return cases


def built_in_manifest():
    """Return focused experiments motivated by the first-round winners."""
    fd_fourier = {
        "architecture": "fourier_mlp",
        "residual_mode": "finite_difference",
        "fourier_sigma": 2.0,
        "max_steps": 50000,
        "max_seconds": 180.0,
    }
    cases = [
        # F: determine whether the discrete winner is explained by architecture,
        # derivative cost, update count, or the discrete operator itself.
        _case(
            "F01",
            "F",
            "Autodiff SIREN (3, 3), 6000-step cap",
            {"max_steps": 6000, "max_seconds": 180.0},
        ),
        _case(
            "F02",
            "F",
            "Finite-difference SIREN (3, 3), 6000-step cap",
            {
                "residual_mode": "finite_difference",
                "max_steps": 6000,
                "max_seconds": 180.0,
            },
        ),
        _case(
            "F03",
            "F",
            "Autodiff Fourier MLP sigma 2, 6000-step cap",
            {
                "architecture": "fourier_mlp",
                "fourier_sigma": 2.0,
                "max_steps": 6000,
                "max_seconds": 180.0,
            },
        ),
        _case(
            "F04",
            "F",
            "Finite-difference Fourier MLP, 6000-step cap",
            {**fd_fourier, "max_steps": 6000},
        ),
        _case(
            "F05",
            "F",
            "Finite-difference Fourier MLP, throughput baseline",
            fd_fourier,
        ),
        _case(
            "F06",
            "F",
            "Finite difference on a 33x33 grid",
            {**fd_fourier, "fd_resolution": 33, "max_seconds": 120.0},
        ),
        _case(
            "F07",
            "F",
            "Finite difference on a 97x97 grid",
            {**fd_fourier, "fd_resolution": 97},
        ),
        _case(
            "F08",
            "F",
            "Fourth-order finite differences on a 65x65 grid",
            {**fd_fourier, "fd_order": 4},
        ),
        _case(
            "F09",
            "F",
            "Conservative finite-difference momentum residual",
            {**fd_fourier, "fd_formulation": "conservative"},
        ),
        _case(
            "F10",
            "F",
            "Finite difference with compatible hard constraints",
            {**fd_fourier, "hard_constraints": True},
        ),
        # G: the Re=100 solution is smooth away from the lid corners, so refine
        # the low-frequency/capacity region that beat every other autodiff INR.
        _case(
            "G01",
            "G",
            "Very-low-frequency SIREN (1, 1)",
            {"first_omega_0": 1.0, "hidden_omega_0": 1.0},
        ),
        _case(
            "G02",
            "G",
            "Low-frequency SIREN (2, 2)",
            {"first_omega_0": 2.0, "hidden_omega_0": 2.0},
        ),
        _case(
            "G03",
            "G",
            "Low-frequency SIREN (4, 4)",
            {"first_omega_0": 4.0, "hidden_omega_0": 4.0},
        ),
        _case(
            "G04",
            "G",
            "SIREN (3, 3), width 128",
            {"hidden_features": 128},
        ),
        _case(
            "G05",
            "G",
            "SIREN (3, 3), width 192",
            {"hidden_features": 192},
        ),
        _case(
            "G06",
            "G",
            "SIREN (3, 3), three hidden layers",
            {"hidden_layers": 3},
        ),
        # H: grouped wall equations currently consume many more sampled values
        # than the one shared interior cloud. Test that balance before adding
        # more adaptive samplers, which were uniformly poor in round one.
        _case(
            "H01",
            "H",
            "SIREN with 256 samples per boundary",
            {"boundary_samples": 256},
        ),
        _case(
            "H02",
            "H",
            "SIREN with 512 samples per boundary",
            {"boundary_samples": 512},
        ),
        _case(
            "H03",
            "H",
            "SIREN with 1024 samples per boundary",
            {"boundary_samples": 1024},
        ),
        _case(
            "H04",
            "H",
            "SIREN with 1024 interior and 512 boundary samples",
            {"samples": 1024, "boundary_samples": 512},
        ),
        _case(
            "H05",
            "H",
            "SIREN without the boundary-to-PDE ramp",
            {"loss_balance": "equal_groups"},
        ),
        _case(
            "H06",
            "H",
            "Low-frequency SIREN with fixed Sobol points",
            {"sampling": "fixed_sobol"},
        ),
        _case(
            "H07",
            "H",
            "Low-frequency SIREN with integral continuity",
            {"integral_continuity_weight": 0.1},
        ),
    ]
    assert len(cases) == 23
    return cases


class SOAP(torch.optim.Optimizer):
    """Small self-contained SOAP-style optimizer for the benchmark.

    Matrix-shaped gradients are rotated into periodically refreshed Shampoo
    eigenbases before Adam moments are applied. Vectors use ordinary Adam.
    This intentionally avoids adding a benchmark-only package dependency.
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.95, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=10,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if group["weight_decay"]:
                    gradient = gradient.add(parameter, alpha=group["weight_decay"])
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                step = state["step"]

                if gradient.ndim == 2 and max(gradient.shape) <= 512:
                    rows, columns = gradient.shape
                    if "left" not in state:
                        state["left"] = torch.zeros(
                            rows, rows, dtype=gradient.dtype, device=gradient.device
                        )
                        state["right"] = torch.zeros(
                            columns,
                            columns,
                            dtype=gradient.dtype,
                            device=gradient.device,
                        )
                        state["ql"] = torch.eye(
                            rows, dtype=gradient.dtype, device=gradient.device
                        )
                        state["qr"] = torch.eye(
                            columns, dtype=gradient.dtype, device=gradient.device
                        )
                        state["exp_avg"] = torch.zeros_like(gradient)
                        state["exp_avg_sq"] = torch.zeros_like(gradient)
                        state["moment_step"] = 0
                    state["left"].mul_(beta2).addmm_(
                        gradient, gradient.T, beta=1.0, alpha=(1 - beta2) / columns
                    )
                    state["right"].mul_(beta2).addmm_(
                        gradient.T, gradient, beta=1.0, alpha=(1 - beta2) / rows
                    )
                    if step == 1 or step % group["precondition_frequency"] == 0:
                        _, state["ql"] = torch.linalg.eigh(state["left"])
                        _, state["qr"] = torch.linalg.eigh(state["right"])
                        state["exp_avg"].zero_()
                        state["exp_avg_sq"].zero_()
                        state["moment_step"] = 0
                    state["moment_step"] += 1
                    moment_step = state["moment_step"]
                    projected = state["ql"].T @ gradient @ state["qr"]
                    first = state["exp_avg"]
                    second = state["exp_avg_sq"]
                    first.mul_(beta1).add_(projected, alpha=1 - beta1)
                    second.mul_(beta2).addcmul_(projected, projected, value=1 - beta2)
                    first_hat = first / (1 - beta1**moment_step)
                    second_hat = second / (1 - beta2**moment_step)
                    direction = (
                        state["ql"]
                        @ (first_hat / (second_hat.sqrt() + group["eps"]))
                        @ state["qr"].T
                    )
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(gradient)
                        state["exp_avg_sq"] = torch.zeros_like(gradient)
                    first = state["exp_avg"]
                    second = state["exp_avg_sq"]
                    first.mul_(beta1).add_(gradient, alpha=1 - beta1)
                    second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                    direction = (first / (1 - beta1**step)) / (
                        (second / (1 - beta2**step)).sqrt() + group["eps"]
                    )
                parameter.add_(direction, alpha=-group["lr"])
        return loss


class ResidualAdaptiveSampler(CollocationSampler):
    """Fixed Sobol bank periodically redistributed by PDE residual magnitude."""

    def __init__(self, seed=0, exploration=0.25):
        super().__init__(method="fixed_sobol", seed=seed)
        self.exploration = float(exploration)

    def update(self, domain, samples, score):
        if not all(np.isnan(domain[coordinate]) for coordinate in ("x", "y")):
            return
        count = samples["x"].size(0)
        keep = round(count * (1 - self.exploration))
        probabilities = score.detach().flatten().cpu().clamp_min(0)
        probabilities = probabilities + 0.01 * probabilities.mean().clamp_min(1e-12)
        selected = torch.multinomial(probabilities, keep, replacement=True)
        current = torch.cat(
            [samples["x"].detach().cpu(), samples["y"].detach().cpu()], dim=1
        )
        exploration = self._unit_samples(domain, count - keep)
        self._fixed_samples[((None, None), count)] = torch.cat(
            [current[selected], exploration], dim=0
        )


class CavityHardConstraint(torch.nn.Module):
    """Enforce compatible ``v`` wall values and the pressure gauge exactly.

    The conflicting horizontal-velocity values at the two lid corners remain
    soft because a continuous neural ansatz cannot enforce both limits there.
    """

    def __init__(self, model, field_indices):
        super().__init__()
        self.model = model
        self.field_indices = dict(field_indices)

    def forward(self, x, y):
        outputs = self.model(x, y)
        constrained = list(outputs.split(1, dim=-1))
        v_index = self.field_indices["v"]
        p_index = self.field_indices["p"]
        constrained[v_index] = x * (1 - x) * y * (1 - y) * constrained[v_index]
        zeros = torch.zeros(1, 1, dtype=x.dtype, device=x.device)
        gauge = self.model(zeros, zeros)[:, p_index : p_index + 1]
        constrained[p_index] = constrained[p_index] - gauge
        return torch.cat(constrained, dim=-1)


def cavity_equations(reynolds=100.0, convection_scale=1.0):
    """Create the standard nondimensional cavity equations in parser syntax."""
    viscosity = 1.0 / float(reynolds)
    alpha = float(convection_scale)
    return [
        "u(0, y) = 0",
        "u(1, y) = 0",
        "u(x, 0) = 0",
        "u(x, 1) = 1",
        "v(0, y) = 0",
        "v(1, y) = 0",
        "v(x, 0) = 0",
        "v(x, 1) = 0",
        "p(0, 0) = 0",
        (
            f"{alpha:.12g} * (u * grad(u, x) + v * grad(u, y)) + "
            f"grad(p, x) - {viscosity:.12g} * (grad(grad(u, x), x) + "
            "grad(grad(u, y), y)) = 0"
        ),
        (
            f"{alpha:.12g} * (u * grad(v, x) + v * grad(v, y)) + "
            f"grad(p, y) - {viscosity:.12g} * (grad(grad(v, x), x) + "
            "grad(grad(v, y), y)) = 0"
        ),
        "grad(u, x) + grad(v, y) = 0",
    ]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_interior(domain):
    return np.isnan(domain["x"]) and np.isnan(domain["y"])


def progress_fraction(step, started, config):
    step_progress = step / max(1, int(config["max_steps"]))
    time_progress = (time.monotonic() - started) / max(1e-6, config["max_seconds"])
    return min(1.0, max(step_progress, time_progress))


def configure_learning_rate(optimizer, config, progress):
    if config["lr_schedule"] == "cosine":
        peak = float(config["lr"])
        floor = float(config["min_lr"])
        learning_rate = floor + 0.5 * (peak - floor) * (
            1 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


def spatial_weights(index, residual, samples, equations, domains, config):
    strategy = config["spatial_weighting"]
    if strategy == "residual_attention":
        magnitude = residual.detach().abs()
        return 1 + 4 * magnitude / magnitude.mean().clamp_min(1e-12)
    if strategy == "lid_taper_sdf":
        domain = domains[index]
        if (
            not is_interior(domain)
            and domain["y"] == 1
            and equations[index].lstrip().startswith("u(")
        ):
            x = samples["x"]
            return (4 * x * (1 - x)).clamp_min(0.05)
        if is_interior(domain):
            x, y = samples["x"], samples["y"]
            distance = torch.minimum(torch.minimum(x, 1 - x), torch.minimum(y, 1 - y))
            weights = 0.25 + 4 * distance
            return weights / weights.mean().clamp_min(1e-12)
    return None


def integral_continuity_loss(model, field_indices, device):
    axis = torch.linspace(0, 1, 64, device=device).view(-1, 1)
    cuts = torch.tensor([0.2, 0.4, 0.6, 0.8], device=device).view(-1, 1)
    losses = []
    for cut in cuts:
        fixed = torch.full_like(axis, cut.item())
        vertical = model(fixed, axis)[:, field_indices["u"]]
        horizontal = model(axis, fixed)[:, field_indices["v"]]
        losses.extend([vertical.mean().square(), horizontal.mean().square()])
    return torch.stack(losses).mean()


def combine_group_losses(
    equation_losses,
    domains,
    model,
    config,
    state,
    progress,
    step,
):
    boundary_indices = [
        index for index, domain in enumerate(domains) if not is_interior(domain)
    ]
    interior_indices = [
        index for index, domain in enumerate(domains) if is_interior(domain)
    ]
    boundary = torch.stack(
        [equation_losses[index] for index in boundary_indices]
    ).mean()
    pde = torch.stack([equation_losses[index] for index in interior_indices]).mean()
    strategy = config["loss_balance"]
    if strategy == "legacy":
        weighted = [
            (0.1 if is_interior(domain) else 0.9) * loss
            for loss, domain in zip(equation_losses, domains)
        ]
        return torch.stack(weighted).mean()
    if strategy == "equal_groups":
        return 0.5 * (boundary + pde)
    if strategy == "boundary_pde_ramp":
        pde_weight = min(1.0, 0.1 + 3 * progress)
        return (boundary + pde_weight * pde) / (1 + pde_weight)
    if strategy == "relobralo":
        current = torch.stack([boundary, pde])
        if "initial_group_losses" not in state:
            state["initial_group_losses"] = current.detach().clamp_min(1e-12)
            state["previous_group_losses"] = current.detach().clamp_min(1e-12)
            state["group_weights"] = torch.ones_like(current)
        temperature = float(config["relobralo_temperature"])
        relative_initial = current.detach() / state["initial_group_losses"]
        relative_previous = current.detach() / state["previous_group_losses"]
        balanced_initial = 2 * torch.softmax(relative_initial / temperature, dim=0)
        balanced_previous = 2 * torch.softmax(relative_previous / temperature, dim=0)
        keep_previous = (
            torch.rand((), device=current.device)
            < float(config["relobralo_lookback_probability"])
        ).to(current.dtype)
        historical = (
            keep_previous * state["group_weights"]
            + (1 - keep_previous) * balanced_initial
        )
        alpha = float(config["relobralo_alpha"])
        state["group_weights"] = alpha * historical + (1 - alpha) * balanced_previous
        state["previous_group_losses"] = current.detach().clamp_min(1e-12)
        return (state["group_weights"] * current).sum() / 2
    if strategy == "gradient_norm":
        if step % 25 == 0 or "group_weights" not in state:
            norms = []
            parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            for group_loss in (boundary, pde):
                gradients = torch.autograd.grad(
                    group_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                squared_norms = [
                    gradient.detach().square().sum()
                    for gradient in gradients
                    if gradient is not None
                ]
                if squared_norms:
                    norm = torch.sqrt(torch.stack(squared_norms).sum())
                else:
                    norm = torch.ones((), device=group_loss.device)
                norms.append(norm.clamp_min(1e-12))
            inverse = 1 / torch.stack(norms)
            state["group_weights"] = 2 * inverse / inverse.sum()
        return (state["group_weights"] * torch.stack([boundary, pde])).sum() / 2
    raise ValueError(f"Unknown loss-balance strategy {strategy!r}")


def pinn_loss(
    equations,
    domains,
    residuals,
    model,
    field_indices,
    sampler,
    config,
    state,
    progress,
    step,
    device,
    adapt_sampler=True,
):
    sample_counts = [int(config["samples"])] * len(equations)
    for index, domain in enumerate(domains):
        if (
            not is_interior(domain)
            and config.get("boundary_samples") is not None
        ):
            sample_counts[index] = int(config["boundary_samples"])
        if not np.isnan(domain["x"]) and not np.isnan(domain["y"]):
            sample_counts[index] = 1
    _, evaluations = compute_equation_losses(
        residuals,
        domains,
        model,
        field_indices,
        sample_counts,
        device,
        sampler=sampler,
        grouped=bool(config["grouped"]),
        penalty=config["penalty"],
        huber_delta=config["huber_delta"],
    )
    equation_losses = []
    for index, (residual, samples) in enumerate(evaluations):
        weights = spatial_weights(index, residual, samples, equations, domains, config)
        equation_losses.append(
            reduce_residual(
                residual,
                penalty=config["penalty"],
                huber_delta=config["huber_delta"],
                weights=weights,
            )
        )
    loss = combine_group_losses(
        equation_losses, domains, model, config, state, progress, step
    )
    if config["integral_continuity_weight"]:
        loss = loss + float(
            config["integral_continuity_weight"]
        ) * integral_continuity_loss(model, field_indices, device)
    if (
        adapt_sampler
        and isinstance(sampler, ResidualAdaptiveSampler)
        and step % int(config["rad_refresh"]) == 0
    ):
        interior_indices = [
            index for index, domain in enumerate(domains) if is_interior(domain)
        ]
        score = torch.stack(
            [
                evaluations[index][0].detach().abs().flatten()
                for index in interior_indices
            ]
        ).mean(dim=0)
        sampler.update(
            domains[interior_indices[0]], evaluations[interior_indices[0]][1], score
        )
    return loss


def finite_difference_operators(field, spacing, order):
    """Return center values, first derivatives, and Laplacian on a grid."""
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
    raise ValueError(f"Unsupported finite-difference order {order}")


def finite_difference_loss(model, field_indices, config, state, device):
    resolution = int(config["fd_resolution"])
    if "fd_coordinates" not in state:
        axis = torch.linspace(0, 1, resolution, device=device)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        state["fd_coordinates"] = (x.reshape(-1, 1), y.reshape(-1, 1))
    x, y = state["fd_coordinates"]
    outputs = model(x, y).reshape(resolution, resolution, -1)
    u = outputs[:, :, field_indices["u"]]
    v = outputs[:, :, field_indices["v"]]
    p = outputs[:, :, field_indices["p"]]
    spacing = 1 / (resolution - 1)
    order = int(config.get("fd_order", 2))
    interior_u, u_x, u_y, u_laplacian = finite_difference_operators(
        u, spacing, order
    )
    interior_v, v_x, v_y, v_laplacian = finite_difference_operators(
        v, spacing, order
    )
    _, p_x, p_y, _ = finite_difference_operators(p, spacing, order)
    formulation = config.get("fd_formulation", "advective")
    if formulation == "advective":
        momentum_u = (
            interior_u * u_x
            + interior_v * u_y
            + p_x
            - 0.01 * u_laplacian
        )
        momentum_v = (
            interior_u * v_x
            + interior_v * v_y
            + p_y
            - 0.01 * v_laplacian
        )
    elif formulation == "conservative":
        _, u_squared_x, _, _ = finite_difference_operators(
            u.square(), spacing, order
        )
        _, uv_x, uv_y, _ = finite_difference_operators(u * v, spacing, order)
        _, _, v_squared_y, _ = finite_difference_operators(
            v.square(), spacing, order
        )
        momentum_u = u_squared_x + uv_y + p_x - 0.01 * u_laplacian
        momentum_v = uv_x + v_squared_y + p_y - 0.01 * v_laplacian
    else:
        raise ValueError(f"Unknown finite-difference formulation {formulation!r}")
    continuity = u_x + v_y
    pde = torch.stack(
        [
            momentum_u.square().mean(),
            momentum_v.square().mean(),
            continuity.square().mean(),
        ]
    ).mean()
    boundary_terms = [
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
    boundary = torch.stack(boundary_terms).mean()
    return 0.5 * (boundary + pde)


def make_sampler(config):
    if config["sampling"] == "rad":
        return ResidualAdaptiveSampler(
            seed=config["seed"], exploration=config["rad_exploration"]
        )
    return CollocationSampler(method=config["sampling"], seed=config["seed"])


def make_optimizer(model, config):
    if config["optimizer"] == "soap":
        return SOAP(model.parameters(), lr=config["lr"])
    return torch.optim.Adam(model.parameters(), lr=config["lr"])


def current_curriculum(config, progress):
    if config["curriculum"] != "re_convection":
        return 100.0, 1.0
    phases = [(10.0, 0.25), (30.0, 0.5), (60.0, 0.75), (100.0, 1.0)]
    return phases[min(len(phases) - 1, int(progress * len(phases)))]


def ghia_error(model, field_indices, device):
    """Compute the sole benchmark ranking metric."""
    metric_model = getattr(model, "_orig_mod", model)
    metric_model.eval()
    with torch.no_grad():
        y = torch.as_tensor(GHIA_U_Y, device=device).view(-1, 1)
        x_mid = torch.full_like(y, 0.5)
        predicted_u = metric_model(x_mid, y)[:, field_indices["u"]]
        x = torch.as_tensor(GHIA_V_X, device=device).view(-1, 1)
        y_mid = torch.full_like(x, 0.5)
        predicted_v = metric_model(x, y_mid)[:, field_indices["v"]]
        reference_u = torch.as_tensor(GHIA_U, device=device)
        reference_v = torch.as_tensor(GHIA_V, device=device)
        relative_u = torch.linalg.vector_norm(
            predicted_u - reference_u
        ) / torch.linalg.vector_norm(reference_u)
        relative_v = torch.linalg.vector_norm(
            predicted_v - reference_v
        ) / torch.linalg.vector_norm(reference_v)
    return (0.5 * (relative_u + relative_v)).item()


def train_one(
    config,
    base_equations,
    variables,
    base_domains,
    device,
    absolute_deadline=None,
):
    set_seed(int(config["seed"]))
    field_indices = {variable: index for index, variable in enumerate(variables)}
    model = build_model(
        config["architecture"],
        in_features=2,
        hidden_features=int(config["hidden_features"]),
        hidden_layers=int(config["hidden_layers"]),
        out_features=len(variables),
        first_omega_0=float(config["first_omega_0"]),
        hidden_omega_0=float(config["hidden_omega_0"]),
        activation=config["activation"],
        fourier_features=int(config["fourier_features"]),
        fourier_sigma=float(config["fourier_sigma"]),
    ).to(device)
    if config["hard_constraints"]:
        model = CavityHardConstraint(model, field_indices).to(device)
    if config["compile_model"]:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requires PyTorch 2.0 or newer")
        model.forward = torch.compile(model.forward, dynamic=False)
        if hasattr(model, "forward_with_derivatives"):
            model.forward_with_derivatives = torch.compile(
                model.forward_with_derivatives, dynamic=False
            )

    optimizer = make_optimizer(model, config)
    exponential_scheduler = None
    if config["lr_schedule"] == "exponential":
        exponential_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(config["lr_gamma"])
        )
    sampler = make_sampler(config)
    state = {}
    equations = base_equations
    domains = base_domains
    residuals = compile_residuals(equations, field_indices, model)
    curriculum_phase = None
    started = time.monotonic()
    # Reserve validation and synchronization time inside the advertised cap.
    deadline = started + max(1.0, float(config["max_seconds"]) - 4.0)
    if absolute_deadline is not None:
        deadline = min(deadline, float(absolute_deadline) - 4.0)
    step = 0
    final_loss = math.nan
    lbfgs = None
    model.train()

    while step < int(config["max_steps"]) and time.monotonic() < deadline:
        progress = progress_fraction(step, started, config)
        reynolds, convection = current_curriculum(config, progress)
        phase = (reynolds, convection)
        if phase != curriculum_phase and config["curriculum"] != "none":
            equations = cavity_equations(reynolds, convection)
            parsed_variables, domains = parse_equations(equations, verbose=False)
            if parsed_variables != variables:
                raise RuntimeError("Curriculum equations changed field order")
            residuals = compile_residuals(equations, field_indices, model)
            curriculum_phase = phase

        use_lbfgs = config["optimizer"] == "adam_lbfgs" and progress >= 0.75
        if use_lbfgs:
            if lbfgs is None:
                lbfgs = torch.optim.LBFGS(
                    model.parameters(),
                    lr=0.5,
                    max_iter=5,
                    max_eval=6,
                    history_size=25,
                    line_search_fn="strong_wolfe",
                )

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                if config["residual_mode"] == "finite_difference":
                    loss = finite_difference_loss(
                        model, field_indices, config, state, device
                    )
                else:
                    loss = pinn_loss(
                        equations,
                        domains,
                        residuals,
                        model,
                        field_indices,
                        sampler,
                        config,
                        state,
                        progress,
                        step,
                        device,
                        adapt_sampler=False,
                    )
                loss.backward()
                return loss

            final_loss = float(lbfgs.step(closure).detach())
        else:
            optimizer.zero_grad(set_to_none=True)
            if config["legacy_exact"]:
                loss = compute_loss(
                    equations,
                    domains,
                    model,
                    field_indices,
                    int(config["samples"]),
                    device,
                    residuals,
                )
            elif config["residual_mode"] == "finite_difference":
                loss = finite_difference_loss(
                    model, field_indices, config, state, device
                )
            else:
                loss = pinn_loss(
                    equations,
                    domains,
                    residuals,
                    model,
                    field_indices,
                    sampler,
                    config,
                    state,
                    progress,
                    step,
                    device,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
            if exponential_scheduler is not None:
                exponential_scheduler.step()
            else:
                configure_learning_rate(optimizer, config, progress)
        step += 1

    if step == 0:
        raise RuntimeError("No training step fit inside the case wall-clock budget")
    score = ghia_error(model, field_indices, device)
    return model, {
        "E_Ghia": score,
        "steps": step,
        "final_training_loss": final_loss,
    }


def config_fingerprint(config, input_file=None):
    digest = hashlib.sha256()
    digest.update(f"benchmark-schema-{RESULT_SCHEMA_VERSION}\0".encode())
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    for source in (Path(__file__), SCRIPT_DIR / "solveanything.py"):
        digest.update(f"\0source:{source.name}\0".encode())
        digest.update(source.read_bytes())
    if input_file is not None:
        digest.update(b"\0problem-file\0")
        digest.update(Path(input_file).read_bytes())
    return digest.hexdigest()[:16]


def result_candidates(results, stages):
    latest_by_id = {}
    for result in results:
        latest_by_id[result.get("id")] = result
    return sorted(
        [
            result
            for result in latest_by_id.values()
            if result.get("status") == "ok"
            and result.get("stage") in stages
            and isinstance(result.get("E_Ghia"), (int, float))
            and math.isfinite(result["E_Ghia"])
        ],
        key=lambda result: result["E_Ghia"],
    )


def resolve_experiment(case, results, defaults):
    config = copy.deepcopy(defaults)
    inherited_from = None
    if case.get("finalist_rank") is not None:
        candidates = result_candidates(results, {"B", "C", "D"})
        rank = int(case["finalist_rank"])
        if len(candidates) < rank:
            raise RuntimeError(
                f"{case['id']} needs finalist rank {rank}, but only "
                f"{len(candidates)} successful B-D results are available"
            )
        parent = candidates[rank - 1]
        config.update(copy.deepcopy(parent["config"]))
        inherited_from = parent["id"]
    elif case.get("inherit_stages"):
        candidates = result_candidates(results, set(case["inherit_stages"]))
        if not candidates:
            raise RuntimeError(
                f"{case['id']} needs a successful result from stages "
                f"{case['inherit_stages']}"
            )
        parent = candidates[0]
        config.update(copy.deepcopy(parent["config"]))
        inherited_from = parent["id"]
    config.update(copy.deepcopy(case.get("config", {})))
    return config, inherited_from


def validate_config(config):
    seconds = float(config["max_seconds"])
    if not 0 < seconds <= MAX_SECONDS_ALLOWED:
        raise ValueError(
            f"max_seconds must be in (0, {MAX_SECONDS_ALLOWED}], got {seconds}"
        )
    if int(config["samples"]) < 1 or int(config["max_steps"]) < 1:
        raise ValueError("samples and max_steps must be positive")
    boundary_samples = config.get("boundary_samples")
    if boundary_samples is not None and int(boundary_samples) < 1:
        raise ValueError("boundary_samples must be positive or null")
    fd_order = int(config.get("fd_order", 2))
    if fd_order not in {2, 4}:
        raise ValueError("fd_order must be 2 or 4")
    minimum_resolution = 5 if fd_order == 2 else 7
    if int(config["fd_resolution"]) < minimum_resolution:
        raise ValueError(
            f"fd_resolution must be at least {minimum_resolution} for "
            f"order {fd_order}"
        )
    if config.get("fd_formulation", "advective") not in {
        "advective",
        "conservative",
    }:
        raise ValueError(
            "fd_formulation must be 'advective' or 'conservative'"
        )


def load_user_config(path, manifest, defaults):
    run_config = {}
    if path is None:
        return manifest, defaults, run_config
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    defaults.update(payload.get("defaults", {}))
    run_config.update(payload.get("run", {}))
    supplied = payload.get("experiments", [])
    if payload.get("replace_manifest", False):
        manifest = []
    by_id = {case["id"]: case for case in manifest}
    for override in supplied:
        case_id = override["id"]
        if case_id not in by_id:
            new_case = _case(
                case_id,
                override.get("stage", "custom"),
                override.get("name", case_id),
            )
            manifest.append(new_case)
            by_id[case_id] = new_case
        target = by_id[case_id]
        for key in (
            "stage",
            "name",
            "enabled",
            "inherit_stages",
            "finalist_rank",
        ):
            if key in override:
                target[key] = copy.deepcopy(override[key])
        config_override = override.get("config", {})
        config_override.update(
            {
                key: value
                for key, value in override.items()
                if key
                not in {
                    "id",
                    "stage",
                    "name",
                    "enabled",
                    "inherit_stages",
                    "finalist_rank",
                    "config",
                }
            }
        )
        target["config"].update(config_override)
    return manifest, defaults, run_config


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
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "metric": {
                "name": "E_Ghia",
                "definition": "mean relative L2 error of u(x=0.5,y) and v(x,y=0.5)",
            },
            "runs": [],
        }
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def upsert_result(results, new_result):
    """Keep one authoritative result per manifest ID."""
    results[:] = [result for result in results if result.get("id") != new_result["id"]]
    results.append(new_result)


def save_results(payload, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    temporary = output_dir / "results.json.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(json_path)
    columns = [
        "id",
        "stage",
        "name",
        "status",
        "E_Ghia",
        "seconds",
        "peak_vram_gb",
        "steps",
        "seed",
        "inherited_from",
        "fingerprint",
        "plot_file",
        "error",
    ]
    with open(output_dir / "results.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in payload["runs"]:
            writer.writerow({column: result.get(column, "") for column in columns})


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


def run_worker(spec_path):
    """Execute one resolved case in an isolated, parent-timed process."""
    with open(spec_path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    result_path = Path(spec["result_path"])
    result = {"status": "error"}
    device = None
    model = None
    try:
        equations, variables, domains, _ = parse_problem_file(
            spec["input_file"], verbose=False
        )
        device = configure_device(spec["device"], float(spec["max_vram_gb"]))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model, training = train_one(
            spec["config"],
            equations,
            variables,
            domains,
            device,
            absolute_deadline=spec["absolute_deadline"],
        )
        axis = torch.linspace(
            0.0,
            1.0,
            FINAL_PLOT_RESOLUTION,
            device=device,
        )
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        model.eval()
        with torch.no_grad():
            final_frame = model(
                x.reshape(-1, 1), y.reshape(-1, 1)
            ).reshape(FINAL_PLOT_RESOLUTION, FINAL_PLOT_RESOLUTION, -1)
        final_frame = final_frame.rot90().cpu().numpy()
        plot_path = Path(spec["plot_path"])
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_plot = plot_path.with_suffix(".tmp.png")
        make_static_plot(final_frame, variables, temporary_plot)
        temporary_plot.replace(plot_path)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result.update(training)
        result["plot_file"] = str(plot_path)
        result["status"] = "ok"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        if device is not None and device.type == "cuda":
            result["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        else:
            result["peak_vram_gb"] = 0.0
        del model
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
            handle.write("\n")
    return 0 if result["status"] == "ok" else 1


def execute_worker(case, config, input_file, output_dir, device_name, max_vram_gb):
    """Launch and hard-limit one case, returning its worker payload and wall time."""
    worker_dir = output_dir / ".workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    token = f"{case['id']}-{config_fingerprint(config, input_file)}"
    spec_path = worker_dir / f"{token}.spec.json"
    result_path = worker_dir / f"{token}.result.json"
    plot_path = output_dir / "plots" / f"{case['id']}.png"

    def cleanup_worker_files():
        for path in (spec_path, result_path):
            if path.exists():
                path.unlink()
        try:
            worker_dir.rmdir()
        except OSError:
            pass

    started = time.monotonic()
    spec = {
        "input_file": str(input_file.resolve()),
        "device": device_name,
        "max_vram_gb": max_vram_gb,
        "config": config,
        "absolute_deadline": started + float(config["max_seconds"]),
        "result_path": str(result_path.resolve()),
        "plot_path": str(plot_path.resolve()),
    }
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle)
        handle.write("\n")
    if result_path.exists():
        result_path.unlink()
    if plot_path.exists():
        plot_path.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-spec",
        str(spec_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        remaining = max(
            0.01,
            float(config["max_seconds"]) - (time.monotonic() - started),
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
        cleanup_worker_files()
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
        detail = (
            stderr.strip() or stdout.strip() or f"worker exited {process.returncode}"
        )
        worker_result = {
            "status": "error",
            "error": detail,
            "peak_vram_gb": None,
        }
    cleanup_worker_files()
    return worker_result, seconds


def selected_cases(manifest, case_ids, stages):
    selected = [case for case in manifest if case.get("enabled", True)]
    if stages:
        selected = [case for case in selected if case["stage"] in stages]
    if case_ids:
        requested = set(case_ids)
        known = {case["id"] for case in manifest}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown case id(s): {', '.join(unknown)}")
        selected = [case for case in selected if case["id"] in requested]
    return selected


def print_manifest(manifest):
    print(f"{'ID':<4} {'Stage':<5} Description")
    for case in manifest:
        suffix = " [disabled]" if not case.get("enabled", True) else ""
        print(f"{case['id']:<4} {case['stage']:<5} {case['name']}{suffix}")
    print(f"\n{sum(case.get('enabled', True) for case in manifest)} enabled cases")


def print_ranking(results):
    stages = {result.get("stage") for result in results}
    successful = result_candidates(results, stages)
    print("\nRanking by E_Ghia (lower is better)")
    print(f"{'Rank':>4} {'ID':<4} {'E_Ghia':>12} {'Seconds':>9} {'VRAM GiB':>9}")
    for rank, result in enumerate(successful, start=1):
        print(
            f"{rank:>4} {result['id']:<4} {result['E_Ghia']:>12.6g} "
            f"{result['seconds']:>9.1f} {result['peak_vram_gb']:>9.3f}"
        )


def run(args):
    if args.worker_spec is not None:
        return run_worker(args.worker_spec)
    manifest = built_in_manifest()
    defaults = copy.deepcopy(DEFAULT_TRAINING)
    # To rerun the original DOE instead, replace the preceding two lines with:
    # manifest = round1_manifest()
    # defaults = copy.deepcopy(ROUND1_DEFAULT_TRAINING)
    manifest, defaults, file_run_config = load_user_config(
        args.config, manifest, defaults
    )
    if args.write_config:
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "replace_manifest": False,
            "run": {
                "input_file": str(DEFAULT_INPUT),
                "output_dir": str(DEFAULT_OUTPUT_DIR),
                "device": "cuda",
                "max_vram_gb": MAX_VRAM_GB_ALLOWED,
            },
            "defaults": defaults,
            "experiments": manifest,
        }
        with open(args.write_config, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {len(manifest)} cases to {args.write_config}")
        return 0
    if args.list:
        print_manifest(manifest)
        return 0

    common_overrides = parse_set_values(args.set_values)
    if args.max_seconds is not None:
        common_overrides["max_seconds"] = args.max_seconds
    if args.samples is not None:
        common_overrides["samples"] = args.samples
    if args.seed is not None:
        common_overrides["seed"] = args.seed
    for case in manifest:
        case["config"].update(common_overrides)

    input_file = Path(
        args.input_file or file_run_config.get("input_file", DEFAULT_INPUT)
    )
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
        raise ValueError(
            f"max_vram_gb must be in (0, {MAX_VRAM_GB_ALLOWED}], got {max_vram_gb}"
        )
    requested_ids = []
    for value in args.case:
        requested_ids.extend(item.strip() for item in value.split(",") if item.strip())
    cases = selected_cases(manifest, requested_ids, set(args.stage))

    if args.dry_run:
        print_manifest(cases)
        print("\nResolved independent cases / overrides for dependent cases:")
        preview = {}
        for case in cases:
            config = copy.deepcopy(defaults)
            config.update(case.get("config", {}))
            preview[case["id"]] = {
                "inherits": case.get("inherit_stages") or case.get("finalist_rank"),
                "config": config,
            }
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    equations, variables, domains, _ = parse_problem_file(input_file, verbose=False)
    if set(variables) != {"u", "v", "p"} or len(equations) != 12:
        raise ValueError(
            "The benchmark expects the 12-equation u/v/p lid-driven-cavity problem"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "results.json"
    payload = load_results(result_file)
    results = payload["runs"]

    for case in cases:
        try:
            config, inherited_from = resolve_experiment(case, results, defaults)
            validate_config(config)
            fingerprint = config_fingerprint(config, input_file)
            if args.resume and any(
                result.get("id") == case["id"]
                and result.get("status") == "ok"
                and result.get("fingerprint") == fingerprint
                for result in results
            ):
                print(f"{case['id']}: already complete (matching configuration)")
                continue
        except Exception as error:
            if args.fail_fast:
                raise
            print(f"{case['id']}: dependency/configuration error: {error}")
            upsert_result(
                results,
                {
                    "id": case["id"],
                    "stage": case["stage"],
                    "name": case["name"],
                    "status": "error",
                    "error": str(error),
                    "E_Ghia": None,
                    "seconds": 0.0,
                    "peak_vram_gb": 0.0,
                },
            )
            save_results(payload, output_dir)
            continue

        print(
            f"{case['id']} [{case['stage']}]: {case['name']} "
            f"(budget {config['max_seconds']:.0f}s, seed {config['seed']})"
        )
        result = {
            "id": case["id"],
            "stage": case["stage"],
            "name": case["name"],
            "status": "error",
            "E_Ghia": None,
            "seconds": None,
            "peak_vram_gb": 0.0,
            "steps": 0,
            "seed": config["seed"],
            "inherited_from": inherited_from,
            "fingerprint": fingerprint,
            "config": config,
        }
        worker_result, seconds = execute_worker(
            case,
            config,
            input_file,
            output_dir,
            device_name,
            max_vram_gb,
        )
        result.update(worker_result)
        result["seconds"] = seconds

        upsert_result(results, result)
        save_results(payload, output_dir)
        if result["status"] == "ok":
            print(
                f"  E_Ghia={result['E_Ghia']:.6g}, {result['seconds']:.1f}s, "
                f"peak={result['peak_vram_gb']:.3f} GiB, steps={result['steps']}"
            )
        else:
            print(f"  ERROR: {result['error']}")
            if args.fail_fast:
                raise RuntimeError(f"{case['id']} failed: {result['error']}")

    print_ranking(results)
    print(f"\nResults: {output_dir / 'results.json'}")
    return 0


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run the follow-up Re=100 lid-driven-cavity PINN benchmark"
    )
    parser.add_argument("--config", type=Path, help="JSON configuration/manifest")
    parser.add_argument(
        "--write-config", type=Path, help="write the built-in manifest and exit"
    )
    parser.add_argument("--input-file", "-i", help="cavity equation file")
    parser.add_argument("--output-dir", help="directory for resumable JSON/CSV results")
    parser.add_argument("--device", help="Torch device (default: cuda)")
    parser.add_argument(
        "--max-vram-gb", type=float, help="GPU allocation cap, at most 8"
    )
    parser.add_argument(
        "--max-seconds", type=float, help="override every selected case, at most 200"
    )
    parser.add_argument("--samples", type=int, help="override collocation count")
    parser.add_argument("--seed", type=int, help="override seed for selected cases")
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        choices=list("ABCDEFGH"),
        help="run a stage; repeatable",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="case ID or comma-separated IDs; repeatable",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="override a training key for all selected cases; repeatable",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the manifest and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show selection/defaults without importing a CUDA context",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip matching successful runs (default: true)",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop at the first failed case"
    )
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(make_parser().parse_args()))
