# Genetic Algorithm Solver — Thomas-Fermi Equation

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python)](https://python.org)
[![Published](https://img.shields.io/badge/Related%20Work-Scientific%20Reports%202023-1565C0?style=flat)](https://www.nature.com/articles/s41598-025-21585-3)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **Genetic Algorithm (GA)** implementation that solves the **Thomas-Fermi equation**, a singular nonlinear second-order boundary-value problem from atomic physics, and compares its accuracy against a naive RK4-based shooting method.

This solver builds on the heuristic optimization approach used in our published work on nonlinear reaction-diffusion systems ([*Scientific Reports*, Nature Portfolio, 2023](https://www.nature.com/articles/s41598-025-21585-3)).

---

## The Problem

The Thomas-Fermi equation models the electron charge density distribution in heavy atoms:

```
y''(x) = y(x)^(3/2) / sqrt(x)
```

with boundary conditions:

```
y(0)  = 1
y(x) -> 0   as x -> infinity
```

This is a **singular, stiff boundary-value problem** — the equation blows up at `x = 0`, and the unknown initial slope `y'(0)` must be found such that the solution decays to zero rather than diverging or crossing zero prematurely.

---

## Approach

Both methods reduce the BVP to an initial-value problem (IVP) by guessing `y'(0) = s` and integrating forward with **RK4**, starting from a small `x₀ > 0` using a local series expansion to avoid the singularity at `x = 0`:

```
y(x)  ≈ 1 + s·x + (4/3)·x^(3/2)
y'(x) ≈ s + 2·x^(1/2)
```

For an incorrect guess of `s`, the trajectory either crosses zero too early (overshoot) or diverges to infinity (undershoot). We define the **escape distance** `E(s)` as how far the trajectory survives before failing — the correct slope maximizes `E(s)` across the full domain.

### Method 1 — Genetic Algorithm
Evolves a population of 40 candidate slopes over 60 generations using tournament selection, arithmetic crossover, Gaussian mutation, and elitism — maximizing `E(s)` directly on the full-resolution RK4 integrator (20,000 steps).

### Method 2 — Naive Bisection Shooting
A hand-coded-style bisection method on a coarser RK4 grid (400 steps), representative of a typical first-pass classical implementation.

---

## Results

![Solution Comparison](results/solution_comparison.png)

| Method | y'(0) | Error vs. reference | Function evaluations |
|---|---|---|---|
| Reference (literature) | −1.5880710226 | — | — |
| Naive Bisection Shooting | −2.0607880599 | 4.73 × 10⁻¹ | 120 |
| **Genetic Algorithm** | **−1.5822587709** | **5.81 × 10⁻³** | 2,440 |

The GA converges to within **5.8 × 10⁻³** of the literature reference value for `y'(0)`, roughly **80× more accurate** than the naive bisection shooting method, which diverges due to its coarse grid and simple interval-halving logic.

![GA Parameter Convergence](results/ga_parameter_convergence.png)

The GA's best estimate stabilizes around generation 5–10 and remains close to the reference value for the remainder of the run, demonstrating robust convergence despite the equation's sensitivity.

---

## Repository Structure

```
genetic-algorithm/
├── ga_solver.py                    # Main script: GA + naive shooting + plots
├── results/
│   ├── solution_comparison.png     # Solution curves + convergence comparison
│   └── ga_parameter_convergence.png
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/saadjamshed256/genetic-algorithm.git
cd genetic-algorithm
pip install -r requirements.txt
python ga_solver.py
```

---

## Related Work

This solver applies the same heuristic optimization philosophy as our published research:

> **Jamshed S. et al.** — *Heuristic computational approach for nonlinear reaction-diffusion kinetics in catalytic systems.*
> Scientific Reports, Nature Portfolio, 2023.
> [Read on nature.com →](https://www.nature.com/articles/s41598-025-21585-3)

---

## Adapting This Solver to Your Own ODE / BVP

This code is structured so the **GA optimizer and integrator are problem-agnostic** — only a handful of pieces need to change to solve a different second-order ODE with different boundary conditions. To adapt it:

### 1. Change the ODE right-hand side

Edit `thomas_fermi_rhs(x, y, yp)`. This defines your system as a pair of first-order ODEs:

```python
def my_rhs(x, y, yp):
    # y'  = yp
    # yp' = f(x, y, yp)   <-- replace with your equation
    return yp, f(x, y, yp)
```

For example, for `y'' + p(x)·y' + q(x)·y = r(x)`:

```python
def my_rhs(x, y, yp):
    return yp, r(x) - p(x)*yp - q(x)*y
```

### 2. Update the boundary conditions

This solver assumes a **shooting method**: one boundary condition is known at `x = 0` (here, `y(0) = 1`), and the unknown is the initial slope `y'(0) = s`, which the GA searches for so that the *other* boundary condition (here, `y(x_max) → 0`) is satisfied.

- If your left boundary condition is `y(0) = A` instead of `1`, change the `1.0` in `series_start()` (or wherever `y0` is set) to `A`.
- If you're solving for a different unknown (e.g. an eigenvalue, a flux, or `y'(0)` has a known value but `y(0)` is unknown), swap which variable the GA searches over — just make sure `slope_bounds` in `GAThomasFermiSolver` reflects a sensible range for that unknown.
- If your right boundary condition is `y(x_max) = B` (not zero), change the failure/escape conditions in `rk4_integrate()` so they check `y_new <= B` (or `>= B`, depending on direction) instead of `<= 0`.

### 3. Remove the singularity workaround if not needed

`series_start()` exists because the Thomas-Fermi equation has a `1/sqrt(x)` singularity at `x = 0`. If your ODE is well-behaved at `x = 0`, you can integrate directly from `x = 0`:

```python
x, y, yp = 0.0, A, slope0   # no series expansion needed
```

### 4. Update the fitness function

`escape_distance()` returns "how far the trajectory survives" — this works well for problems where wrong guesses cause blow-up or premature zero-crossing. For other problems, you may want a more direct fitness:

```python
def fitness(slope0):
    _, y_final, _ = rk4_integrate(slope0, full_output=False)
    return abs(y_final - TARGET_VALUE)   # GA should MINIMIZE this
```

If you switch to a "minimize residual" fitness (rather than "maximize escape distance"), remember to flip the comparisons in `_select_parents()`, `solve()` (elite/worst selection), and `np.argmax` → `np.argmin`.

### 5. PDEs (extending beyond ODEs)

This solver targets a 1D ODE shooting problem. For a PDE (e.g. a heat or reaction-diffusion equation with spatial *and* time dependence), the GA can still be used to optimize **unknown parameters** (boundary fluxes, reaction rates, initial conditions) — but the inner solver needs to become a PDE integrator (e.g. finite differences in space + an ODE integrator in time) rather than a single RK4 sweep. The GA loop structure (`_initial_population`, `_select_parents`, `_crossover`, `_mutate`, `solve`) can be reused as-is; only the fitness function needs to call your PDE solver instead of `rk4_integrate()`.

### Summary of what to change

| Component | Function to edit | Typical change |
|---|---|---|
| The ODE itself | `thomas_fermi_rhs` | Replace with your `f(x, y, y')` |
| Left BC | `series_start` | Change `y(0) = A` |
| Right BC / failure condition | `rk4_integrate` | Change `y_new <= 0` check |
| Singularity handling | `series_start` | Remove if not needed |
| Fitness definition | `escape_distance` | Swap for residual-based fitness if appropriate |
| Search range for unknown | `GAThomasFermiSolver.bounds` | Set to a sensible range for your unknown parameter |

---



```
Jamshed S. et al. (2023). Heuristic computational approach for nonlinear
reaction-diffusion kinetics in catalytic systems. Scientific Reports.
https://www.nature.com/articles/s41598-025-21585-3
```
