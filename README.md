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

The 36 files in [`examples`](examples) cover direct functions, gradients,
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
output fields) per file in an ASCII table. Problems that fail are reported as
`ERROR` without preventing the remaining files from running.

To benchmark one file while developing:

```bash
python benchmark.py examples --pattern "01_direct_bilinear.txt"
```
