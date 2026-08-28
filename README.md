# solveanything

`solveanything.py` is a small library for parsing equations and training
coordinate neural networks against their residuals. It supports two-dimensional
steady fields `(x, y)` and two-space/one-time fields `(x, y, t)`, arbitrary
coordinate bounds, and named geometric sampling domains.

The executable entry point is `run_ns_benchmark.py`, a resumable benchmark of
PINN-like approaches on classical incompressible Navier--Stokes problems.

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Pillow and tqdm
- optional FFmpeg or `imageio-ffmpeg` for H.264 `.mp4` artifacts

## Problem files

A problem contains an optional `# Domains` section, a required `# Equations`
section, and an optional analytic `# Solution` section:

```text
# Domains
space_time = box(x=(0, 2 * pi), y=(0, 2 * pi), t=(0, 1))
initial = box(x=(0, 2 * pi), y=(0, 2 * pi), t=0)

# Equations
u = sin(x) * cos(y) @ initial
grad(u, t) + u * grad(u, x) + v * grad(u, y) = 0 @ space_time

# Solution
u = sin(x) * cos(y) * exp(-0.02 * t)
```

The suffix `@ domain_name` selects the points used for that equation. Without
an annotation, the original unit-square domain inference remains available.

The safe geometry language supports:

- `box(x=(xmin, xmax), y=(ymin, ymax), t=(tmin, tmax))`, where any coordinate
  may instead be fixed to one scalar;
- `circle(center=(cx, cy), radius=r, t=(tmin, tmax))` for a boundary;
- `disk(center=(cx, cy), radius=r, ...)` for an interior;
- `difference(outer, hole, ...)` for perforated fluid domains;
- `union(first, second, ...)` for grouped boundaries.

For example, the DFG cylinder interior and no-slip surface are expressed as:

```text
# Domains
fluid = difference(box(x=(0, 2.2), y=(0, 0.41), t=(0, 8)), disk(center=(0.2, 0.2), radius=0.05))
cylinder = circle(center=(0.2, 0.2), radius=0.05, t=(0, 8))

# Equations
u = 0 @ cylinder
v = 0 @ cylinder
```

See [`benchmarks/navier_stokes`](benchmarks/navier_stokes) for complete files.

## Navier--Stokes benchmark

List the built-in physical problems and approaches:

```bash
python3 run_ns_benchmark.py --list
```

Run everything on CUDA:

```bash
python3 run_ns_benchmark.py --device cuda
```

Select individual problems and approaches:

```bash
python3 run_ns_benchmark.py --device cuda \
  --problem cavity_re100,cylinder_re100,taylor_green_re100 \
  --approach siren_low,fourier_mlp
```

Useful development and configuration commands include:

```bash
python3 run_ns_benchmark.py --dry-run
python3 run_ns_benchmark.py --write-config ns_benchmark.json
python3 run_ns_benchmark.py --config ns_benchmark.json --device cuda
python3 run_ns_benchmark.py --problem taylor_green_re100 \
  --approach siren_low --max-steps 5 --max-seconds 30 --device cpu \
  --set samples=16 --set boundary_samples=16
```

The built-in suite contains:

- lid-driven cavity at `Re=100`, `400`, and `1000`, ranked by the mean relative
  `L2` error of the two Ghia centerlines (`E_Ghia`);
- the DFG cylinder cases 2D-1 (`Re=20`, steady) and 2D-3 (`Re=100`, transient),
  ranked by the mean relative error of their standard drag, lift, and pressure
  difference observables (`E_DFG`);
- the analytic Taylor--Green vortex at `Re=100` and `1000`, ranked by the mean
  space-time relative `L2` error of `u` and `v` (`E_TGV`).

References are used only after training:

- [Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991(82)90058-4)
- [DFG cylinder 2D-1](https://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)
- [DFG cylinder 2D-3](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html)
- [Taylor & Green (1937)](https://doi.org/10.1098/rspa.1937.0036)

The default approaches are low-frequency and compact SIRENs, a tanh MLP, a
Fourier-feature MLP, a modified gated MLP, and SIREN sampling/loss-balancing
ablations. The previous structured finite-difference/Fourier winner remains
available for compatible cavity cases.

Each run executes in an isolated process with a hard limit of at most 200
seconds and 8 GiB of allocated CUDA memory. Results are atomically persisted to
`benchmark_results/navier_stokes/results.json` and `results.csv`, so matching
successful runs can be resumed. Final trained models produce:

- one `.png` field plot for `(x, y)` problems;
- one animation over physical time for `(x, y, t)` problems. The runner writes
  H.264 `.mp4` when a system FFmpeg or the optional `imageio-ffmpeg` package is
  available, and otherwise falls back automatically to an animated `.gif`.

Artifacts are stored under
`benchmark_results/navier_stokes/artifacts/<problem>/<approach>.*`.

## Library API

`solveanything.py` deliberately has no CLI. Its reusable entry points include
`parse_problem_file`, `build_model`, `compile_residuals`,
`compute_equation_losses`, `CollocationSampler`, `train_model`, and
`make_static_plot`. The files under [`examples`](examples) remain compact parser
and equation fixtures for direct functions, ODEs, and PDEs.
