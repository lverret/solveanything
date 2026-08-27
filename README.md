# solveanything

`solveanything` takes a set of equations and trains a 2D function that best
satisfies them. Like a physics-informed neural network, it can solve direct
function definitions, boundary-value problems, differential equations, and
coupled systems. The approximating model is a
[SIREN](https://arxiv.org/abs/2006.09661), and mathematical expressions are
validated by a small AST parser before training.

## Requirements

- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow](https://python-pillow.org/)
- [tqdm](https://tqdm.github.io/)

## Problem-file format

Problem files contain an `# Equations` section and may contain a `# Solution`
section with one analytic formula per output field:

```text
# Equations
f = x * y

# Solution
f = x * y
```

The solution section supports `x`, `y`, `pi`, numeric operators, and the
functions `sqrt`, `sin`, `cos`, `exp`, `abs`, and `tanh`. It is used only for
visual comparison and benchmarking; it never participates in training.

The 37 files in [`examples`](examples) cover direct functions, gradients,
ODEs, PDEs, nonlinear equations, coupled fields, and higher-frequency cases.

For SIREN models, pure derivatives in `x` or `y` through fourth order are
propagated exactly through the network layers in one batched pass. Mixed or
higher-order expressions automatically fall back to PyTorch autograd.

## Solve a problem

```bash
python solveanything.py --input_file examples/01_direct_bilinear.txt
```

By default, this writes `out.gif`. If the file provides a `# Solution`, the GIF
shows the evolving approximation in its first row and the analytic solution in
its second row, using the same color scale for each output field.

Use `--no_gif` to train without collecting or exporting animation frames. Run
`python solveanything.py --help` for all model and training options.

## Benchmark a folder

Run all problem files in `examples` with the solver's default SIREN and training
parameters:

```bash
python benchmark.py examples
```

The script resets the Torch seed before each problem, evaluates predictions on
an 81 by 81 grid, and prints one absolute MSE (averaged across the grid and all
output fields) per file in an ASCII table. Problems without a `# Solution`
section are marked `SKIP`; problems that fail are reported as `ERROR` without
preventing the remaining files from running.

To benchmark one file while developing:

```bash
python benchmark.py examples --pattern "01_direct_bilinear.txt"
```

## Re=100 lid-driven-cavity benchmark

`run_lid_driven_cavity_re100_benchmark.py` contains the staged 41-case
PINN/INR design of experiments for
[`examples/37_lid_driven_cavity_re100.txt`](examples/37_lid_driven_cavity_re100.txt).
Run the complete manifest on CUDA with:

```bash
python3 run_lid_driven_cavity_re100_benchmark.py --device cuda
```

Every case runs in an isolated process with a hard wall-clock limit of at most
200 seconds and a PyTorch allocator cap of at most 8 GiB. Results are written
after every case to resumable `results.json` and `results.csv` files under
`benchmark_results/lid_driven_cavity_re100`. Matching successful cases are
skipped on a later invocation unless `--no-resume` is supplied.

The sole ranking metric is

```text
E_Ghia = 0.5 * (relative_L2(u(0.5, y)) + relative_L2(v(x, 0.5)))
```

using the Re=100 centerline data from
[Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991(82)90058-4).
Lower is better. The reference values are used only after training, never in a
loss or stopping rule.

Inspect or select built-in cases and override common settings from the command
line:

```bash
python3 run_lid_driven_cavity_re100_benchmark.py --list
python3 run_lid_driven_cavity_re100_benchmark.py --device cuda --stage A
python3 run_lid_driven_cavity_re100_benchmark.py --device cuda \
  --case B01,B02,B03 --max-seconds 120 --samples 4096
python3 run_lid_driven_cavity_re100_benchmark.py --device cuda \
  --case D05 --set hidden_features=192 --set fourier_sigma=3.0
```

Stages C and D inherit the best available earlier recipe. Stage E expands the
best three B-D cases to seeds 0, 1, and 2, so run the full manifest in order or
reuse an output directory that already contains the prerequisite results.

For a JSON-controlled run, export the complete built-in manifest, edit any
defaults or case-specific `config` objects, and pass it back to the runner:

```bash
python3 run_lid_driven_cavity_re100_benchmark.py \
  --write-config cavity_benchmark.json
python3 run_lid_driven_cavity_re100_benchmark.py \
  --config cavity_benchmark.json --device cuda
```

Use `--dry-run`, `--help`, and `--list` to validate a selection without opening
a CUDA context.
