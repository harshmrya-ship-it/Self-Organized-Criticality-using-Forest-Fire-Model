"""
Forest Fire Model - Self-Organised Criticality Simulation
==========================================================
Drossel & Schwabl (1992) forest fire model.
Demonstrates SOC: noise, avalanches, connectivity, and power-law statistics.

Usage:
    python forest_fire_soc.py

Dependencies:
    numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque

# ── State constants ──────────────────────────────────────────────────────────
EMPTY   = 0
TREE    = 1
BURNING = 2
ASH     = 3

# ── Simulation parameters ────────────────────────────────────────────────────
N         = 100      # Grid size (N x N)
P_GROW    = 0.005    # Tree growth probability per step
P_LIGHT   = 0.0001   # Lightning (ignition) probability per step
T_TOTAL   = 10_000   # Total timesteps
T_WARMUP  = 2_000    # Warm-up steps (discarded from statistics)


def bfs_fire(grid, row, col):
    """
    Burn all trees connected to (row, col) using BFS.
    Returns the number of trees burned (avalanche size).
    """
    N = grid.shape[0]
    queue = deque([(row, col)])
    grid[row, col] = BURNING
    size = 1
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and grid[nr, nc] == TREE:
                grid[nr, nc] = BURNING
                queue.append((nr, nc))
                size += 1
    return size


def step(grid, p=P_GROW, f=P_LIGHT):
    """
    Perform one synchronous timestep of the FFM.

    Rules (applied synchronously):
      Empty  --[prob p]--> Tree           (growth / noise)
      Tree   --[prob f]--> Burning        (lightning / noise)
      Tree   --[neighbour is Burning]--> Burning  (fire spread / avalanche)
      Burning --> Ash
      Ash    --> Empty

    Returns:
        new_grid      : updated grid
        fire_sizes    : list of avalanche sizes triggered this step
        density       : fraction of cells that are trees
    """
    N = grid.shape[0]
    new_grid = grid.copy()
    fire_sizes = []

    empty_mask   = (grid == EMPTY)
    burning_mask = (grid == BURNING)
    ash_mask     = (grid == ASH)

    # Ash -> Empty
    new_grid[ash_mask] = EMPTY

    # Empty -> Tree with prob p  (noise: slow growth)
    grow = np.random.random(grid.shape) < p
    new_grid[empty_mask & grow] = TREE

    # Burning -> Ash  (fire dies after one step)
    new_grid[burning_mask] = ASH

    # Spread fire to tree neighbours of currently burning cells
    spread = np.zeros_like(grid, dtype=bool)
    spread[:-1, :] |= burning_mask[1:,  :]   # up
    spread[1:,  :] |= burning_mask[:-1, :]   # down
    spread[:, :-1] |= burning_mask[:,  1:]   # left
    spread[:, 1:]  |= burning_mask[:, :-1]   # right
    new_grid[(grid == TREE) & spread] = BURNING

    # Lightning: random ignition of trees  (noise: rare perturbation)
    lightning = (np.random.random(grid.shape) < f) & (new_grid == TREE)
    ignition_points = list(zip(*np.where(lightning)))

    # Each lightning strike triggers a BFS avalanche
    for (r, c) in ignition_points:
        if new_grid[r, c] == TREE:
            size = bfs_fire(new_grid, r, c)
            fire_sizes.append(size)

    density = np.mean(new_grid == TREE)
    return new_grid, fire_sizes, density


def run_simulation(n=N, p=P_GROW, f=P_LIGHT,
                   t_total=T_TOTAL, t_warmup=T_WARMUP, seed=42):
    """
    Run the full FFM simulation.

    Args:
        n        : grid size
        p        : tree growth probability
        f        : lightning probability
        t_total  : total timesteps
        t_warmup : warm-up steps to discard (allow system to reach SOC attractor)
        seed     : random seed for reproducibility

    Returns:
        all_fire_sizes  : list of avalanche sizes (after warm-up)
        density_ts      : tree density at every timestep
        density_at_fire : density at the moment each fire was triggered
    """
    np.random.seed(seed)
    # Initialise with ~30% tree coverage
    grid = (np.random.random((n, n)) < 0.3).astype(int)

    all_fire_sizes  = []
    density_ts      = []
    density_at_fire = []

    for t in range(t_total):
        grid, fires, density = step(grid, p, f)
        density_ts.append(density)

        if t >= t_warmup:
            for s in fires:
                all_fire_sizes.append(s)
                density_at_fire.append(density)

        if t % 1000 == 0:
            n_fires = len([s for s in fires])
            print(f"  Step {t:6d}/{t_total}  |  "
                  f"density = {density:.3f}  |  fires this step = {n_fires}")

    return all_fire_sizes, density_ts, density_at_fire


def mle_exponent(sizes, s_min=1):
    """
    Maximum likelihood estimate of power-law exponent tau.
    Clauset, Shalizi & Newman (2009) Eq. (3.1).
    """
    s = np.array([x for x in sizes if x >= s_min], dtype=float)
    n = len(s)
    if n < 2:
        return None
    tau = 1 + n * (np.sum(np.log(s / (s_min - 0.5))))**(-1)
    return tau


def plot_results(fire_sizes, density_ts, density_at_fire, n=N, p=P_GROW, f=P_LIGHT):
    """Generate all four analysis panels."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Forest Fire Model — Self-Organised Criticality\n"
        f"N={n}, p={p}, f={f}, ratio f/p={f/p:.3f}",
        fontsize=13, fontweight='bold'
    )

    sizes = np.array(fire_sizes)

    # ── Panel 1: Density time series ─────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(density_ts, color='#2d6a2d', lw=0.5, alpha=0.85, label='Tree density')
    ax.axvline(T_WARMUP, color='gray', lw=1.2, ls='--', label=f'Warm-up end (t={T_WARMUP})')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Tree density  ρ(t)")
    ax.set_title("(a) Density time series — noise & avalanche cycles")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # ── Panel 2: Fire size distribution (log–log) ────────────────────────────
    ax = axes[0, 1]
    if len(sizes) > 0:
        bins = np.logspace(np.log10(max(sizes.min(), 1)),
                           np.log10(sizes.max()), 40)
        counts, edges = np.histogram(sizes, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2
        mask = counts > 0
        ax.scatter(centres[mask], counts[mask],
                   color='#e85d24', s=22, zorder=3, label='Simulation data')

        # OLS fit on log–log
        lx = np.log10(centres[mask])
        ly = np.log10(counts[mask])
        coeffs = np.polyfit(lx, ly, 1)
        xfit = np.linspace(lx.min(), lx.max(), 300)
        ax.plot(10**xfit, 10**np.polyval(coeffs, xfit),
                color='#3B8BD4', lw=2,
                label=f'Power-law fit  (slope = {coeffs[0]:.2f})')

        # MLE exponent
        tau = mle_exponent(sizes, s_min=5)
        if tau:
            ax.text(0.05, 0.08, f'MLE  τ = {tau:.2f}',
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Fire size  s  (cells burned)")
        ax.set_ylabel("Frequency  P(s)")
        ax.set_title("(b) Avalanche size distribution (log–log)")
        ax.legend(fontsize=8)

    # ── Panel 3: Connectivity vs avalanche size ──────────────────────────────
    ax = axes[1, 0]
    if len(sizes) > 0:
        d_arr = np.array(density_at_fire)
        sc = ax.scatter(d_arr, sizes,
                        c=sizes, cmap='hot_r', s=6, alpha=0.4,
                        norm=mcolors.LogNorm(vmin=max(sizes.min(),1),
                                             vmax=sizes.max()))
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Fire size  s", fontsize=8)
        ax.set_xlabel("Tree density  ρ  at ignition")
        ax.set_ylabel("Fire size  s  (cells)")
        ax.set_title("(c) Connectivity vs avalanche size")
        ax.set_yscale('log')

    # ── Panel 4: CCDF ────────────────────────────────────────────────────────
    ax = axes[1, 1]
    if len(sizes) > 0:
        sorted_s = np.sort(sizes)[::-1]
        ccdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
        ax.loglog(sorted_s, ccdf, color='#534AB7', lw=1.5, label='CCDF')

        # Reference slope
        s_ref = np.logspace(np.log10(sorted_s.min()),
                            np.log10(sorted_s.max()), 100)
        tau_ref = 1.3
        scale = ccdf[0] * sorted_s[0]**(tau_ref - 1)
        ax.loglog(s_ref, scale * s_ref**(-(tau_ref - 1)),
                  'k--', lw=1, alpha=0.5, label=f'Reference slope τ-1={tau_ref-1:.1f}')

        ax.set_xlabel("Fire size  s")
        ax.set_ylabel("P(S > s)")
        ax.set_title("(d) Complementary CDF — power-law signature")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("soc_results.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved → soc_results.png")
    plt.show()


def print_summary(fire_sizes, density_ts):
    sizes = np.array(fire_sizes)
    print("\n" + "=" * 52)
    print("   SIMULATION SUMMARY")
    print("=" * 52)
    print(f"   Total fire events      : {len(sizes):,}")
    if len(sizes):
        print(f"   Mean fire size         : {sizes.mean():.2f} cells")
        print(f"   Median fire size       : {np.median(sizes):.1f} cells")
        print(f"   Max fire size          : {sizes.max():,} cells")
        print(f"   Fires  > 100 cells     : {(sizes > 100).sum():,}")
        print(f"   Fires  > 1000 cells    : {(sizes > 1000).sum():,}")
        tau = mle_exponent(sizes, s_min=5)
        if tau:
            print(f"   MLE exponent  τ        : {tau:.3f}")
    print(f"   Mean tree density      : {np.mean(density_ts):.3f}")
    print(f"   Density std dev        : {np.std(density_ts):.4f}")
    print("=" * 52)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  Forest Fire Model — SOC Simulation")
    print("=" * 52)
    print(f"  Grid : {N} x {N}  ({N*N:,} cells)")
    print(f"  p    : {P_GROW}  (tree growth probability)")
    print(f"  f    : {P_LIGHT}  (lightning probability)")
    print(f"  f/p  : {P_LIGHT/P_GROW:.4f}  (must be << 1 for SOC)")
    print(f"  Steps: {T_TOTAL:,}  (warm-up: {T_WARMUP:,})")
    print()

    fire_sizes, density_ts, density_at_fire = run_simulation()
    print_summary(fire_sizes, density_ts)
    plot_results(fire_sizes, density_ts, density_at_fire)
