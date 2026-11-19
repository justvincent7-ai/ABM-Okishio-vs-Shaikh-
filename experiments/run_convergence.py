"""
Monte Carlo convergence validation (N = 1000 replications, RSEM and CV stability, N* = 840).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

from src.calibration import EXPERIMENT_PARAMS, baseline_parameters, init_paired_homogeneous_firms
from src.model import RealCompetitionModel


VARIABLES_META = {
    "share_shaikh": {
        "label": "Shaikh Market Share $\\bar{s}_{\\mathrm{Shaikh}}$",
        "short": "Shaikh Share",
        "unit": "%",
        "scale": 100.0,
        "color": "#C8102E",
    },
    "r_branch": {
        "label": "Branch Profit Rate $\\bar{r}^{\\mathrm{branch}}$",
        "short": "Profit Rate $r^{\\mathrm{branch}}$",
        "unit": "%",
        "scale": 100.0,
        "color": "#1F5A9E",
    },
    "u_industry": {
        "label": "Capacity Utilization $\\bar{u}_{\\mathrm{branch}}$",
        "short": "Utilization $u_{\\mathrm{branch}}$",
        "unit": "%",
        "scale": 100.0,
        "color": "#2E8B57",
    },
    "kappa_industry": {
        "label": "Capital-Output Ratio $\\bar{\\kappa}_{\\mathrm{branch}}$",
        "short": "Capital Ratio $\\bar{\\kappa}_{\\mathrm{branch}}$",
        "unit": "level",
        "scale": 1.0,
        "color": "#7B3294",
    },
    "g_p_leader": {
        "label": "Leader Price Deflation Rate $g_{p^{\\mathrm{leader}}}$",
        "short": "Deflation $g_{p^{\\mathrm{leader}}}$",
        "unit": "%/per.",
        "scale": 100.0,
        "color": "#D95F02",
    },
    "n_firms": {
        "label": "Active Firms $\\overline{|\\mathcal{N}_t|}$",
        "short": "Active Firms $|\\mathcal{N}_t|$",
        "unit": "firms",
        "scale": 1.0,
        "color": "#555555",
    },
}


def run_single_replication(
    seed: int,
    n_steps: int = 120,
    burn_in: int = 40,
    innovation_regime: str = "marx",
    n_firms_init: int = 10,
    D: float | None = None,
    r_star: float | None = None,
    interest_rate: float | None = None,
    model_overrides: dict | None = None,
) -> dict:
    """Run one deterministic replication and return post-burn-in averages over [burn_in + 1, n_steps]."""
    if burn_in >= n_steps:
        raise ValueError(f"burn_in ({burn_in}) must be strictly smaller than n_steps ({n_steps}).")

    extra = dict(model_overrides or {})
    if D is not None:
        extra["D"] = D
    if r_star is not None:
        extra["r_star"] = r_star
    if interest_rate is not None:
        extra["interest_rate"] = interest_rate

    model = RealCompetitionModel(
        **baseline_parameters(
            overrides=extra,
            innovation_regime=innovation_regime,
            seed=seed,
        )
    )

    kappa_0 = 3.280
    if n_firms_init != 10:
        raise ValueError("The paired paper baseline has exactly 10 initial firms.")
    init_paired_homogeneous_firms(model, seed=seed)

    share_shaikh_series = []
    r_branch_series = []
    u_industry_series = []
    kappa_industry_series = []
    n_firms_series = []
    p_leader_burn_in = 1.000
    p_leader_end = 1.000

    for t in range(1, n_steps + 1):
        model.current_period += 1
        firms_t = list(model.agents)
        if not firms_t:
            break

        model.step_0_demand()
        S_t = sum(f.total_capacity for f in firms_t)
        capital_value_t = sum(v.kappa * v.k for f in firms_t for v in f.vintages)

        model.step_1_pricing()
        model.step_2_demand_allocation()
        model.step_3_production_and_accounting()

        p_min_t = min(f.price for f in firms_t)
        total_sales_t = sum(f.sales for f in firms_t)
        shaikh_sales_t = sum(f.sales for f in firms_t if f.rule == "shaikh")
        share_shaikh_t = (shaikh_sales_t / total_sales_t) if total_sales_t > 0.0 else 0.0
        u_ind_t = (total_sales_t / S_t) if S_t > 0.0 else 0.0
        kappa_ind_t = (capital_value_t / S_t) if S_t > 0.0 else kappa_0
        n_firms_t = len(firms_t)

        model.step_4_scrapping()
        model.step_5_innovation_and_investment()
        model.step_6_entry_and_exit(capital_value_t)

        if t == burn_in:
            p_leader_burn_in = p_min_t
        if t > burn_in:
            share_shaikh_series.append(share_shaikh_t)
            r_branch_series.append(model.r_branch)
            u_industry_series.append(u_ind_t)
            kappa_industry_series.append(kappa_ind_t)
            n_firms_series.append(float(n_firms_t))
            p_leader_end = p_min_t

    eff_steps = max(1, len(share_shaikh_series))
    if p_leader_burn_in > 0.0 and p_leader_end > 0.0:
        g_p_leader = -np.log(p_leader_end / p_leader_burn_in) / eff_steps
    else:
        g_p_leader = 0.0

    return {
        "seed": seed,
        "share_shaikh": float(np.mean(share_shaikh_series)),
        "r_branch": float(np.mean(r_branch_series)),
        "u_industry": float(np.mean(u_industry_series)),
        "kappa_industry": float(np.mean(kappa_industry_series)),
        "g_p_leader": float(g_p_leader),
        "n_firms": float(np.mean(n_firms_series)),
        "share_shaikh_final": float(share_shaikh_series[-1]) if share_shaikh_series else np.nan,
    }


def compute_convergence_metrics(df_raw: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Compute cumulative mean, Student-t 95% CI, RSEM_n, CV_n, and relative CV error across n."""
    N = len(df_raw)
    n_grid = np.arange(min_n, N + 1)
    t_crit = stats.t.ppf(0.975, df=n_grid - 1)

    records = []
    for var_name in VARIABLES_META:
        x = df_raw[var_name].to_numpy(dtype=float)
        idx_all = np.arange(1, N + 1, dtype=float)
        csum = np.cumsum(x)
        csum_sq = np.cumsum(x ** 2)

        mu_all = csum / idx_all
        var_all = np.zeros(N, dtype=float)
        var_all[1:] = np.maximum(
            0.0, (csum_sq[1:] - idx_all[1:] * (mu_all[1:] ** 2)) / (idx_all[1:] - 1.0)
        )
        sigma_all = np.sqrt(var_all)

        mu_n = mu_all[min_n - 1 :]
        sigma_n = sigma_all[min_n - 1 :]
        sem_n = sigma_n / np.sqrt(n_grid)
        ci_half = t_crit * sem_n
        rsem_n = np.where(np.abs(mu_n) > 1e-12, sem_n / np.abs(mu_n), 0.0)
        cv_n = np.where(np.abs(mu_n) > 1e-12, sigma_n / np.abs(mu_n), 0.0)
        cv_final = cv_n[-1] if cv_n[-1] > 1e-12 else 1.0
        cv_rel_err = np.abs(cv_n - cv_final) / cv_final

        for i, n_val in enumerate(n_grid):
            records.append(
                {
                    "variable": var_name,
                    "n": int(n_val),
                    "cum_mean": float(mu_n[i]),
                    "cum_std": float(sigma_n[i]),
                    "sem": float(sem_n[i]),
                    "rsem_pct": float(rsem_n[i] * 100.0),
                    "ci95_low": float(mu_n[i] - ci_half[i]),
                    "ci95_high": float(mu_n[i] + ci_half[i]),
                    "ci95_halfwidth": float(ci_half[i]),
                    "cv": float(cv_n[i]),
                    "cv_rel_err_pct": float(cv_rel_err[i] * 100.0),
                }
            )

    return pd.DataFrame(records)


def determine_optimal_n_star(
    df_conv: pd.DataFrame,
    rsem_threshold_pct: float = 1.75,
    cv_stab_threshold_pct: float = 2.0,
) -> tuple[int, dict[str, int]]:
    """Smallest n >= 30 such that all k >= n keep RSEM_k <= threshold and |CV_k/CV_N - 1| <= threshold."""
    n_star_by_var: dict[str, int] = {}
    for var_name, grp in df_conv.groupby("variable"):
        grp = grp.sort_values("n")
        cond = (
            (grp["rsem_pct"] <= rsem_threshold_pct)
            & (grp["cv_rel_err_pct"] <= cv_stab_threshold_pct)
            & (grp["n"] >= 30)
        )
        vals = cond.to_numpy()
        n_arr = grp["n"].to_numpy()
        suffix_ok = np.logical_and.accumulate(vals[::-1])[::-1]
        valid_indices = np.where(suffix_ok)[0]
        n_star_by_var[var_name] = int(n_arr[valid_indices[0]]) if len(valid_indices) > 0 else int(n_arr[-1])

    core_vars = ["share_shaikh", "r_branch", "u_industry", "kappa_industry"]
    n_star_global = max(n_star_by_var[v] for v in core_vars)
    return n_star_global, n_star_by_var


def plot_mc_convergence(
    df_conv: pd.DataFrame,
    n_star_global: int,
    n_star_by_var: dict[str, int],
    burn_in: int,
    n_steps: int,
    output_path: str,
) -> None:
    """Save 300 DPI Monte Carlo convergence figure (cumulative means, RSEM, CV)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
        }
    )

    n_max = int(df_conv["n"].max())
    fig = plt.figure(figsize=(15.0, 13.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.12], hspace=0.34, wspace=0.22)

    core_vars = ["share_shaikh", "r_branch", "u_industry", "kappa_industry"]
    panel_letters = ["(a.1)", "(a.2)", "(a.3)", "(a.4)"]

    for idx, var_name in enumerate(core_vars):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        meta = VARIABLES_META[var_name]
        scale = meta["scale"]
        c = meta["color"]

        sub = df_conv[df_conv["variable"] == var_name].sort_values("n")
        n_vals = sub["n"].to_numpy()
        mean_vals = sub["cum_mean"].to_numpy() * scale
        low_vals = sub["ci95_low"].to_numpy() * scale
        high_vals = sub["ci95_high"].to_numpy() * scale
        final_mean = mean_vals[-1]
        n_star_v = n_star_by_var[var_name]

        ax.fill_between(n_vals, low_vals, high_vals, color=c, alpha=0.20, label="95% CI ($\\hat{\\mu}_n \\pm t_{0.975}\\,\\mathrm{SEM}_n$)")
        ax.plot(n_vals, mean_vals, color=c, linewidth=2.0, label=f"Cumulative mean $\\hat{{\\mu}}_n$ ($\\hat{{\\mu}}_{{{n_max}}} = {final_mean:.2f}$)")
        ax.axhline(final_mean, color="#222222", linestyle=":", linewidth=1.2, label=f"Asymptotic value ($N={n_max}$)")
        ax.axvline(n_star_v, color="#333333", linestyle="--", linewidth=1.4, label=f"Stabilization threshold $N^*_{{{var_name.split('_')[0]}}} = {n_star_v}$")
        ax.set_title(f"{panel_letters[idx]} {meta['label']}", fontweight="bold", pad=7)
        ax.set_xlabel("Cumulative replications $n$")
        ax.set_ylabel(f"Mean ({meta['unit']})")
        ax.set_xlim(10, n_max)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.92)

    ax_sem = fig.add_subplot(gs[2, 0])
    for var_name in core_vars + ["g_p_leader", "n_firms"]:
        meta = VARIABLES_META[var_name]
        sub = df_conv[df_conv["variable"] == var_name].sort_values("n")
        ax_sem.plot(sub["n"], sub["rsem_pct"], color=meta["color"], linewidth=1.8, label=meta["short"])
    ax_sem.axhline(1.75, color="#B22222", linestyle=":", linewidth=1.5, label="Precision threshold RSEM $= 1.75\\%$")
    ax_sem.axvline(n_star_global, color="#111111", linestyle="--", linewidth=1.8, label=f"Recommended global threshold $N^* = {n_star_global}$")
    ax_sem.set_yscale("log")
    ax_sem.set_xscale("log")
    ax_sem.set_title("(b.1) Relative Standard Error decay $\\mathrm{RSEM}_n = \\mathrm{SEM}_n / |\\hat{\\mu}_n|$ (log-log scale)", fontweight="bold", pad=7)
    ax_sem.set_xlabel("Cumulative replications $n$ (log)")
    ax_sem.set_ylabel("Relative Standard Error RSEM (%) [log]")
    ax_sem.set_xlim(10, n_max)
    ax_sem.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_sem.legend(loc="upper right", ncol=2, framealpha=0.92)

    ax_cv = fig.add_subplot(gs[2, 1])
    for var_name in core_vars + ["g_p_leader", "n_firms"]:
        meta = VARIABLES_META[var_name]
        sub = df_conv[df_conv["variable"] == var_name].sort_values("n")
        cv_vals = sub["cv"].to_numpy()
        cv_norm = (cv_vals / cv_vals[-1]) if cv_vals[-1] > 0 else np.ones_like(cv_vals)
        ax_cv.plot(sub["n"], cv_norm, color=meta["color"], linewidth=1.8, label=f"{meta['short']} ($\\mathrm{{CV}}_{{{n_max}}}={cv_vals[-1]:.3f}$)")
    ax_cv.axhspan(0.98, 1.02, color="#2E8B57", alpha=0.14, label=f"Stabilization band $\\pm 2\\%$ around $\\mathrm{{CV}}_{{{n_max}}}$")
    ax_cv.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2)
    ax_cv.axvline(n_star_global, color="#111111", linestyle="--", linewidth=1.8, label=f"Recommended global threshold $N^* = {n_star_global}$")
    ax_cv.set_title(f"(b.2) Relative stabilization of Coefficient of Variation $\\mathrm{{CV}}_n / \\mathrm{{CV}}_{{{n_max}}}$", fontweight="bold", pad=7)
    ax_cv.set_xlabel("Cumulative replications $n$")
    ax_cv.set_ylabel(f"Ratio $\\mathrm{{CV}}_n / \\mathrm{{CV}}_{{{n_max}}}$")
    ax_cv.set_xlim(10, n_max)
    ax_cv.set_ylim(0.82, 1.18)
    ax_cv.grid(True, linestyle="--", alpha=0.3)
    ax_cv.legend(loc="upper right", ncol=2, framealpha=0.92)

    fig.suptitle(
        f"Monte Carlo Convergence Analysis ($N = {n_max}$ replications, horizon $T = {n_steps}$, burn-in $T_0 = {burn_in}$, Marx-biased regime)",
        fontsize=13,
        fontweight="bold",
        y=0.94,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_monte_carlo_validation(
    n_replications: int = 1000,
    n_steps: int = 120,
    burn_in: int = 40,
    innovation_regime: str = "marx",
    n_jobs: int = -1,
    results_dir: str = str(ROOT / "results"),
    figures_dir: str = str(ROOT / "figures"),
) -> tuple[pd.DataFrame, pd.DataFrame, int, dict[str, int]]:
    """Run parallel Monte Carlo convergence diagnostics and return raw/metrics frames and N*."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"=== MONTE CARLO CONVERGENCE ANALYSIS (N = {n_replications} runs) ===")
    print(f"Parameters: T = {n_steps}, burn-in T_0 = {burn_in}, regime = '{innovation_regime}', n_jobs = {n_jobs}")
    t0 = time.perf_counter()

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_single_replication)(
            seed=s,
            n_steps=n_steps,
            burn_in=burn_in,
            innovation_regime=innovation_regime,
        )
        for s in range(n_replications)
    )
    elapsed = time.perf_counter() - t0
    print(f"Parallel batch finished in {elapsed:.2f} s ({elapsed / n_replications * 1000:.2f} ms / run).")

    df_raw = pd.DataFrame(results).sort_values("seed").reset_index(drop=True)
    raw_csv_path = os.path.join(results_dir, f"mc_baseline_raw_{n_replications}.csv")
    df_raw.to_csv(raw_csv_path, index=False)

    df_conv = compute_convergence_metrics(df_raw, min_n=10)
    conv_csv_path = os.path.join(results_dir, f"mc_convergence_metrics_{n_replications}.csv")
    df_conv.to_csv(conv_csv_path, index=False)

    n_star_global, n_star_by_var = determine_optimal_n_star(
        df_conv, rsem_threshold_pct=1.75, cv_stab_threshold_pct=2.0
    )

    fig_path = os.path.join(figures_dir, "fig_mc_convergence.png")
    plot_mc_convergence(
        df_conv=df_conv,
        n_star_global=n_star_global,
        n_star_by_var=n_star_by_var,
        burn_in=burn_in,
        n_steps=n_steps,
        output_path=fig_path,
    )

    print(f"\n=== MONTE CARLO CONVERGENCE SUMMARY (N = {n_replications}, t in [{burn_in + 1}, {n_steps}]) ===")
    summary_rows = []
    for var_name, meta in VARIABLES_META.items():
        sub_final = df_conv[(df_conv["variable"] == var_name) & (df_conv["n"] == n_replications)].iloc[0]
        scale = meta["scale"]
        summary_rows.append(
            {
                "Variable": var_name,
                f"Mean (N={n_replications})": f"{sub_final['cum_mean'] * scale:.4f} {meta['unit']}",
                "95% CI": f"[{sub_final['ci95_low'] * scale:.4f}, {sub_final['ci95_high'] * scale:.4f}]",
                "SEM": f"{sub_final['sem'] * scale:.4f}",
                "RSEM (%)": f"{sub_final['rsem_pct']:.2f}%",
                "CV": f"{sub_final['cv']:.4f}",
                "N* (RSEM<=1.75%, dCV<=2%)": n_star_by_var[var_name],
            }
        )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n-> Global stabilization threshold across core indicators: N* = {n_star_global}")
    print(f"-> Raw CSV: {raw_csv_path}")
    print(f"-> Convergence metrics CSV: {conv_csv_path}")
    print(f"-> Figure: {fig_path}")
    return df_raw, df_conv, n_star_global, n_star_by_var


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Monte Carlo RSEM & CV convergence validation.")
    parser.add_argument("--n-reps", type=int, default=int(EXPERIMENT_PARAMS["n_replications"]))
    parser.add_argument("--n-steps", type=int, default=int(EXPERIMENT_PARAMS["n_steps"]))
    parser.add_argument("--burn-in", type=int, default=int(EXPERIMENT_PARAMS["burn_in"]))
    parser.add_argument("--regime", type=str, default="marx", choices=("neutral", "marx"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--figures-dir", type=str, default=str(ROOT / "figures"))
    args = parser.parse_args()

    run_monte_carlo_validation(
        n_replications=args.n_reps,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        innovation_regime=args.regime,
        n_jobs=args.n_jobs,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
