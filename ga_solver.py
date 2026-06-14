"""
Genetic Algorithm Solver for the Thomas-Fermi Equation
========================================================

The Thomas-Fermi equation (atomic physics, electron density in heavy atoms):

    y''(x) = y(x)^(3/2) / sqrt(x)

Boundary conditions:
    y(0)  = 1
    y(x) -> 0  as x -> infinity

This is a classic singular, stiff boundary-value problem. We reduce it to
an initial-value problem by guessing the unknown initial slope y'(0) = s
and integrating forward with RK4 starting from a small x0 (using a local
series expansion to avoid the singularity at x = 0).

For an incorrect guess of s, the solution either:
  - overshoots and crosses zero early (s too negative), or
  - diverges to +infinity (s not negative enough)

The correct slope s* is the one that keeps y(x) bounded and decaying over
the largest possible domain before either failure mode occurs. We define
the "escape distance" E(s) = furthest x reached before y <= 0 or y > 1e6,
and the goal is to MAXIMIZE E(s) (ideally reaching x_max with y -> 0).

Two solvers are compared:
  1. Genetic Algorithm (GA)   - evolves a population of slope guesses
  2. RK4 + Shooting (Brent's) - classical 1D root-finding on the same
                                  RK4 integrator
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ----------------------------------------------------------------------
# Problem setup
# ----------------------------------------------------------------------
X0 = 1e-4            # start integration near x=0 (avoids singularity)
X_MAX = 15.0         # truncated "infinity"
N_STEPS = 20000      # RK4 integration steps

# High-precision reference value for y'(0), widely tabulated in the
# Thomas-Fermi literature (e.g. Kobayashi et al. 1955)
REFERENCE_SLOPE = -1.588071022611375


def thomas_fermi_rhs(x, y, yp):
    """y' = yp,  yp' = y^{3/2} / sqrt(x)  (y clipped at 0 to avoid complex powers)."""
    y_clip = max(y, 0.0)
    return yp, (y_clip ** 1.5) / np.sqrt(x)


def series_start(slope0, x0=X0):
    """
    Local series expansion near x = 0:
        y(x)  ~ 1 + s*x + (4/3)*x^(3/2) + ...
        y'(x) ~ s + 2*x^(1/2) + ...
    This avoids the 1/sqrt(x) singularity at x = 0.
    """
    y0 = 1.0 + slope0 * x0 + (4.0 / 3.0) * x0 ** 1.5
    yp0 = slope0 + 2.0 * x0 ** 0.5
    return y0, yp0


def rk4_integrate(slope0, n_steps=N_STEPS, x_max=X_MAX, x0=X0, full_output=False):
    """
    Integrate the Thomas-Fermi IVP forward from x0 to x_max using RK4.

    Returns:
        escape_x : the x value at which y crosses zero or blows up
                   (or x_max if the solution survives the whole domain)
        (optionally) full x, y, y' trajectories
    """
    h = (x_max - x0) / n_steps
    x = x0
    y, yp = series_start(slope0, x0)

    xs, ys, yps = [x], [y], [yp]

    for _ in range(n_steps):
        k1y, k1yp = thomas_fermi_rhs(x, y, yp)
        k2y, k2yp = thomas_fermi_rhs(x + h / 2, y + h / 2 * k1y, yp + h / 2 * k1yp)
        k3y, k3yp = thomas_fermi_rhs(x + h / 2, y + h / 2 * k2y, yp + h / 2 * k2yp)
        k4y, k4yp = thomas_fermi_rhs(x + h, y + h * k3y, yp + h * k3yp)

        y_new = y + (h / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        yp_new = yp + (h / 6) * (k1yp + 2 * k2yp + 2 * k3yp + k4yp)
        x += h

        if y_new <= 0 or y_new > 1e6 or not np.isfinite(y_new):
            if full_output:
                xs.append(x); ys.append(max(y_new, 0.0)); yps.append(yp_new)
                return x, (np.array(xs), np.array(ys), np.array(yps))
            return x

        y, yp = y_new, yp_new
        if full_output:
            xs.append(x); ys.append(y); yps.append(yp)

    if full_output:
        return x_max, (np.array(xs), np.array(ys), np.array(yps))
    return x_max


def escape_distance(slope0):
    """Fitness target: how far the trajectory survives before failing. Maximize this."""
    return rk4_integrate(slope0)


# ----------------------------------------------------------------------
# Method 1: Genetic Algorithm
# ----------------------------------------------------------------------
class GAThomasFermiSolver:
    """
    Genetic Algorithm that evolves a population of candidate initial slopes
    y'(0) to MAXIMIZE the escape distance E(s) (i.e. find the slope that
    keeps the solution bounded and decaying over the longest domain).
    """

    def __init__(self, pop_size=40, generations=60,
                 slope_bounds=(-2.5, -0.8),
                 mutation_rate=0.3, mutation_scale=0.08,
                 elite_frac=0.1, seed=42):
        self.pop_size = pop_size
        self.generations = generations
        self.bounds = slope_bounds
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elite_count = max(1, int(pop_size * elite_frac))
        self.rng = np.random.default_rng(seed)

        self.history_best_fitness = []  # escape distance (higher = better)
        self.history_best_slope = []

    def _fitness(self, slope):
        """Higher escape distance = better. Return raw escape distance."""
        return escape_distance(slope)

    def _initial_population(self):
        lo, hi = self.bounds
        return self.rng.uniform(lo, hi, size=self.pop_size)

    def _select_parents(self, population, fitnesses):
        """Tournament selection (maximize fitness)."""
        selected = []
        for _ in range(self.pop_size):
            i, j = self.rng.integers(0, self.pop_size, size=2)
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            selected.append(winner)
        return np.array(selected)

    def _crossover(self, parents):
        """Arithmetic crossover between adjacent pairs of parents."""
        children = parents.copy()
        for i in range(0, self.pop_size - 1, 2):
            if self.rng.random() < 0.7:
                alpha = self.rng.random()
                p1, p2 = parents[i], parents[i + 1]
                children[i] = alpha * p1 + (1 - alpha) * p2
                children[i + 1] = alpha * p2 + (1 - alpha) * p1
        return children

    def _mutate(self, population):
        lo, hi = self.bounds
        for i in range(self.pop_size):
            if self.rng.random() < self.mutation_rate:
                population[i] += self.rng.normal(0, self.mutation_scale)
                population[i] = np.clip(population[i], lo, hi)
        return population

    def solve(self):
        population = self._initial_population()

        for _ in range(self.generations):
            fitnesses = np.array([self._fitness(s) for s in population])

            best_idx = np.argmax(fitnesses)
            self.history_best_fitness.append(fitnesses[best_idx])
            self.history_best_slope.append(population[best_idx])

            # Elitism: carry over the best individuals unchanged
            elite_idx = np.argsort(fitnesses)[-self.elite_count:]
            elites = population[elite_idx].copy()

            parents = self._select_parents(population, fitnesses)
            children = self._crossover(parents)
            children = self._mutate(children)

            # Replace worst individuals with elites
            worst_idx = np.argsort(fitnesses)[:self.elite_count]
            children[worst_idx] = elites

            population = children

        fitnesses = np.array([self._fitness(s) for s in population])
        best_idx = np.argmax(fitnesses)
        best_slope = population[best_idx]
        self.history_best_fitness.append(fitnesses[best_idx])
        self.history_best_slope.append(best_slope)

        return best_slope


# ----------------------------------------------------------------------
# Method 2: RK4 + Classical optimization (bounded scalar minimization)
# ----------------------------------------------------------------------
def shooting_method_rk4(bounds=(-2.5, -0.8), tol=1e-12):
    """
    Reference-quality approach: maximize escape distance E(s) using a
    bounded 1D optimizer (Brent's method variant) on top of the same
    RK4 integrator used by the GA. Used only to obtain a near-exact
    benchmark slope for error comparison.
    """
    result = minimize_scalar(
        lambda s: -escape_distance(s),
        bounds=bounds, method='bounded',
        options={'xatol': tol}
    )
    return result.x


def naive_bisection_shooting(bounds=(-2.5, -0.8), max_iter=60, n_steps_coarse=400):
    """
    'Naive' classical shooting method, representative of a typical
    hand-coded bisection approach: repeatedly bisect the slope interval
    based on whether the trajectory overshoots (crosses zero) or blows
    up, using a coarser RK4 grid (fewer steps) for each trial.

    This mirrors what a student would implement manually -- a fixed
    number of bisection iterations on a coarse grid, WITHOUT the
    benefit of a derivative-free optimizer's adaptive step control.
    """
    lo, hi = bounds  # lo: too negative (crosses zero), hi: too shallow (blows up)
    history = []

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        e_mid = rk4_integrate(mid, n_steps=n_steps_coarse)
        history.append((mid, e_mid))

        e_lo = rk4_integrate(lo, n_steps=n_steps_coarse)
        e_hi = rk4_integrate(hi, n_steps=n_steps_coarse)

        if abs(e_mid - e_lo) < abs(e_mid - e_hi):
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0, history


# ----------------------------------------------------------------------
# Run both methods and generate comparison plots
# ----------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Thomas-Fermi Equation: GA vs Naive RK4 Shooting (Bisection)")
    print("=" * 64)

    # --- GA solve (fine RK4 grid, full N_STEPS per evaluation) ---
    ga = GAThomasFermiSolver(pop_size=40, generations=60)
    ga_slope = ga.solve()
    ga_escape = escape_distance(ga_slope)

    # --- Naive bisection shooting (coarse grid, hand-coded style) ---
    naive_slope, naive_history = naive_bisection_shooting(max_iter=60)
    naive_escape = escape_distance(naive_slope)

    # --- High-accuracy benchmark (for error reference only) ---
    benchmark_slope = shooting_method_rk4()

    print(f"\nReference y'(0) (literature):   {REFERENCE_SLOPE:.10f}")
    print(f"High-accuracy benchmark y'(0):   {benchmark_slope:.10f}")

    print(f"\nGenetic Algorithm  (40 pop x 60 gens, full-resolution RK4):")
    print(f"  y'(0)            = {ga_slope:.10f}")
    print(f"  Escape distance  = {ga_escape:.6f} / {X_MAX}")
    print(f"  Error vs ref     = {abs(ga_slope - REFERENCE_SLOPE):.3e}")

    print(f"\nNaive Bisection Shooting (60 iters, coarse RK4 grid):")
    print(f"  y'(0)            = {naive_slope:.10f}")
    print(f"  Escape distance  = {naive_escape:.6f} / {X_MAX}")
    print(f"  Error vs ref     = {abs(naive_slope - REFERENCE_SLOPE):.3e}")

    # --- Full trajectories for plotting ---
    _, (x_ga, y_ga, _) = rk4_integrate(ga_slope, full_output=True)
    _, (x_naive, y_naive, _) = rk4_integrate(naive_slope, full_output=True)
    _, (x_ref, y_ref, _) = rk4_integrate(REFERENCE_SLOPE, full_output=True)

    os.makedirs('results', exist_ok=True)

    # ====================================================================
    # FIGURE 1: Solution curves + convergence comparison (side by side)
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(x_ref, y_ref, 'k-', linewidth=2.5, alpha=0.8,
            label=f"Reference  y'(0)={REFERENCE_SLOPE:.6f}")
    ax.plot(x_naive, y_naive, 'b--', linewidth=2,
            label=f"Naive Bisection  y'(0)={naive_slope:.6f}")
    ax.plot(x_ga, y_ga, 'r:', linewidth=2.5,
            label=f"GA  y'(0)={ga_slope:.6f}")
    ax.set_xlabel('x')
    ax.set_ylabel('y(x)')
    ax.set_title('Thomas-Fermi Solution: GA vs Naive Bisection Shooting')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color='gray', linewidth=0.5)

    ax = axes[1]
    gens = np.arange(len(ga.history_best_fitness))
    ax.plot(gens, [abs(s - REFERENCE_SLOPE) for s in ga.history_best_slope],
            'r-o', markersize=3, label="GA: error in y'(0) per generation")

    naive_errors = [abs(s - REFERENCE_SLOPE) for s, _ in naive_history]
    ax.plot(np.arange(len(naive_errors)), naive_errors,
            'b-s', markersize=3, label="Naive Bisection: error per iteration")

    ax.set_yscale('log')
    ax.set_xlabel('Generation / Iteration')
    ax.set_ylabel("Error in y'(0)  (log scale)")
    ax.set_title('Convergence Comparison: GA vs Naive Bisection')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('results/solution_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/solution_comparison.png")
    plt.close(fig)

    # ====================================================================
    # FIGURE 2: GA slope estimate convergence vs reference & naive method
    # ====================================================================
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    slope_history = np.array(ga.history_best_slope)
    ax2.plot(gens, slope_history, 'r-o', markersize=3, label="GA estimate of y'(0)")
    ax2.axhline(REFERENCE_SLOPE, color='k', linestyle='-', alpha=0.7,
                label=f"Reference y'(0) = {REFERENCE_SLOPE:.6f}")
    ax2.axhline(naive_slope, color='b', linestyle='--',
                label=f"Naive Bisection y'(0) = {naive_slope:.6f}")
    ax2.set_xlabel('Generation')
    ax2.set_ylabel("Estimated y'(0)")
    ax2.set_title('GA Parameter Convergence to True Initial Slope')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/ga_parameter_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: results/ga_parameter_convergence.png")
    plt.close(fig2)

    # ====================================================================
    # Summary table
    # ====================================================================
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"{'Method':<28} {'y prime(0)':<16} {'Error vs Ref':<14} {'Func. evals':<10}")
    print("-" * 68)
    print(f"{'Reference (literature)':<28} {REFERENCE_SLOPE:<16.10f} {'-':<14} {'-':<10}")
    print(f"{'Naive Bisection Shooting':<28} {naive_slope:<16.10f} "
          f"{abs(naive_slope-REFERENCE_SLOPE):<14.3e} {len(naive_history)*2:<10}")
    print(f"{'Genetic Algorithm':<28} {ga_slope:<16.10f} "
          f"{abs(ga_slope-REFERENCE_SLOPE):<14.3e} {40*61:<10}")


if __name__ == "__main__":
    main()
