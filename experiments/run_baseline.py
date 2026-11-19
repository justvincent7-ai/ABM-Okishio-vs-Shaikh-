"""
Master baseline experiment (N = 1000 paired runs/regime, Neutral vs Marx-biased, T = 120),
vintage prime-cost feasibility audit (ucs_v > 0), staircase robustness, and figures.
"""

from __future__ import annotations

import argparse
import json
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.calibration import (
    BASELINE_PARAMS,
    EXPERIMENT_PARAMS,
    INITIAL_FIRM_PARAMS,
    build_baseline_model,
    collect_provenance,
    init_paired_homogeneous_firms,
    init_paired_staircase_firms,
)
from src.model import run_detailed_simulation


def verify_initial_symmetry(model) -> None:
    """Assert exact identity of initial endowments and age distributions between rules."""
    firms = list(model.agents)
    sh = [f for f in firms if f.rule == "shaikh"]
    ok = [f for f in firms if f.rule == "okishio"]
    assert len(sh) == 5 and len(ok) == 5
    for attr_fn in (
        lambda f: round(f.mean_uc, 12),
        lambda f: round(f.mean_kappa, 12),
        lambda f: round(f.price, 12),
        lambda f: round(f.total_capacity, 12),
        lambda f: round(f.budget, 12),
        lambda f: round(f.target_share, 12),
        lambda f: round(f.realized_share, 12),
        lambda f: round(f.u, 12),
        lambda f: f.vintages[0].birth_period,
    ):
        assert sorted(attr_fn(f) for f in sh) == sorted(attr_fn(f) for f in ok)


def simulate_one_run(
    seed: int,
    regime: str,
    n_steps: int = EXPERIMENT_PARAMS["n_steps"],
    staircase_robustness: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run one simulation replication, recording branch metrics and vintage feasibility."""
    model = build_baseline_model(innovation_regime=regime, seed=seed)
    assert model.enable_monopoly_pricing is False
    assert model.evolutionary_entry is False
    assert model.enable_low_profit_exit is False

    if staircase_robustness:
        init_paired_staircase_firms(model, seed=seed)
    else:
        init_paired_homogeneous_firms(model, seed=seed)
        verify_initial_symmetry(model)

    dep_denom = model.T * model.u_n
    records = []
    ucs_neg_vintages = 0
    ucs_total_vintages = 0
    ucs_neg_steps = 0
    ucs_min_val = float("inf")
    first_neg_t = None
    first_neg_rule = None
    ucs_neg_shaikh_vintages = 0
    ucs_neg_okishio_vintages = 0

    for t in range(1, n_steps + 1):
        model.current_period += 1
        firms_t = list(model.agents)
        if not firms_t:
            break

        model.step_0_demand()
        S_t = sum(f.total_capacity for f in firms_t)
        is_shortage = float(S_t < model.D)
        capital_value_t = sum(v.kappa * v.k for f in firms_t for v in f.vintages)

        model.step_1_pricing()
        model.step_2_demand_allocation()
        model.step_3_production_and_accounting()

        model.check_invariants(check_prime_costs=False, check_realized_shares=True)
        step_has_neg = False
        for f in firms_t:
            for v in f.vintages:
                ucs_v = v.uc - v.kappa / dep_denom
                ucs_total_vintages += 1
                if ucs_v < ucs_min_val:
                    ucs_min_val = ucs_v
                if ucs_v < -1e-12:
                    ucs_neg_vintages += 1
                    step_has_neg = True
                    if f.rule == "shaikh":
                        ucs_neg_shaikh_vintages += 1
                    else:
                        ucs_neg_okishio_vintages += 1
                    if first_neg_t is None:
                        first_neg_t = t
                        first_neg_rule = f.rule
        if step_has_neg:
            ucs_neg_steps += 1

        Q_t = sum(f.sales for f in firms_t)
        prices = np.array([f.price for f in firms_t], dtype=float)
        sales = np.array([f.sales for f in firms_t], dtype=float)
        p_posted_mean = float(np.mean(prices))
        sd_posted = float(np.std(prices, ddof=0))
        cv_posted = (sd_posted / p_posted_mean * 100.0) if p_posted_mean > 0 else 0.0
        spread_pct_posted = (
            ((float(np.max(prices)) - float(np.min(prices))) / float(np.min(prices)) * 100.0)
            if np.min(prices) > 0
            else 0.0
        )
        p_leader_t = float(np.min(prices))

        sellers_mask = sales > 1e-12
        n_sellers = int(np.sum(sellers_mask))
        if Q_t > 0.0:
            weights = sales / Q_t
            p_avg_t = float(np.sum(weights * prices))
            var_sales = float(np.sum(weights * (prices - p_avg_t) ** 2))
            sd_sales = float(np.sqrt(max(0.0, var_sales)))
            cv_sales = (sd_sales / p_avg_t * 100.0) if p_avg_t > 0 else 0.0
        else:
            p_avg_t = p_leader_t
            sd_sales = 0.0
            cv_sales = 0.0

        uc_branch_t = (
            sum(f.mean_uc * f.total_capacity for f in firms_t) / S_t
            if S_t > 0.0
            else INITIAL_FIRM_PARAMS["uc_0"]
        )
        kappa_branch_t = (
            capital_value_t / S_t if S_t > 0.0 else INITIAL_FIRM_PARAMS["kappa_0"]
        )
        u_branch_t = (Q_t / S_t * 100.0) if S_t > 0.0 else 0.0

        sh_firms = [f for f in firms_t if f.rule == "shaikh"]
        ok_firms = [f for f in firms_t if f.rule == "okishio"]
        share_shaikh_t = (sum(f.sales for f in sh_firms) / Q_t * 100.0) if Q_t > 0.0 else 0.0
        share_okishio_t = (sum(f.sales for f in ok_firms) / Q_t * 100.0) if Q_t > 0.0 else 0.0

        def rule_means(grp):
            if not grp:
                return np.nan, np.nan, np.nan, np.nan
            rs = []
            for f in grp:
                cap_i = sum(v.kappa * v.k for v in f.vintages)
                rs.append((f.profit / cap_i * 100.0) if cap_i > 0 else np.nan)
            return (
                float(np.nanmean(rs)),
                float(np.mean([f.mean_kappa for f in grp])),
                float(np.mean([f.mean_uc for f in grp])),
                float(np.mean([f.price for f in grp])),
            )

        r_sh, kap_sh, uc_sh, p_sh = rule_means(sh_firms)
        r_ok, kap_ok, uc_ok, p_ok = rule_means(ok_firms)

        model.step_4_scrapping()
        model.step_5_innovation_and_investment()
        model.step_6_entry_and_exit(capital_value_t)

        n_exits_marg = len(model.last_exits_marginalized)
        n_exits_exh = len(model.last_exits_exhausted)

        records.append(
            {
                "seed": seed,
                "regime": regime,
                "t": t,
                "D_t": model.D,
                "D_trend_t": model.D_0 * ((1.0 + model.g_D) ** t),
                "r_star": model.r_star * 100.0,
                "is_shortage": is_shortage,
                "share_shaikh": share_shaikh_t,
                "share_okishio": share_okishio_t,
                "p_avg": p_avg_t,
                "p_posted_mean": p_posted_mean,
                "p_leader": p_leader_t,
                "uc_branch": uc_branch_t,
                "kappa_branch": kappa_branch_t,
                "r_branch": model.r_branch * 100.0,
                "u_branch": u_branch_t,
                "r_shaikh": r_sh,
                "r_okishio": r_ok,
                "kappa_shaikh": kap_sh,
                "kappa_okishio": kap_ok,
                "uc_shaikh": uc_sh,
                "uc_okishio": uc_ok,
                "p_shaikh": p_sh,
                "p_okishio": p_ok,
                "n_firms": len(firms_t),
                "n_shaikh": len(sh_firms),
                "n_okishio": len(ok_firms),
                "n_sellers": n_sellers,
                "n_entries": model.last_entries,
                "n_exits_marg": n_exits_marg,
                "n_exits_exh": n_exits_exh,
                "n_exits_total": n_exits_marg + n_exits_exh,
                "sd_posted": sd_posted,
                "cv_posted": cv_posted,
                "spread_pct_posted": spread_pct_posted,
                "sd_sales": sd_sales,
                "cv_sales": cv_sales,
            }
        )

    audit = {
        "seed": seed,
        "regime": regime,
        "ucs_total_vintages": ucs_total_vintages,
        "ucs_neg_vintages": ucs_neg_vintages,
        "ucs_neg_shaikh_vintages": ucs_neg_shaikh_vintages,
        "ucs_neg_okishio_vintages": ucs_neg_okishio_vintages,
        "ucs_neg_steps": ucs_neg_steps,
        "ucs_min": float(ucs_min_val),
        "first_neg_t": first_neg_t,
        "first_neg_rule": first_neg_rule,
    }
    return pd.DataFrame(records), audit


def summarize_experiment(df_all: pd.DataFrame, n_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute primary summary stats over finite-horizon windows and price dispersion decomposition."""
    summary_rows = []
    disp_rows = []
    for reg in ("neutral", "marx"):
        sub = df_all[df_all["regime"] == reg]
        sub_end = sub[sub["t"] == n_steps]
        late_window = max(1, min(80, n_steps - 40)) if n_steps > 40 else n_steps
        sub_late = sub[sub["t"] > max(0, n_steps - late_window)]
        late_start = max(1, n_steps - late_window + 1)
        per_run_cum = sub.groupby("seed")[
            ["n_entries", "n_exits_total", "n_exits_marg", "n_exits_exh", "is_shortage"]
        ].sum()
        summary_rows.append(
            {
                "regime": reg,
                f"share_shaikh_t1_{n_steps}": sub["share_shaikh"].mean(),
                f"share_shaikh_t{late_start}_{n_steps}": sub_late["share_shaikh"].mean(),
                f"share_shaikh_t{n_steps}": sub_end["share_shaikh"].mean(),
                f"shaikh_win_rate_t{n_steps}_pct": (sub_end["share_shaikh"] > 50.0).mean() * 100.0,
                f"p_avg_t{n_steps}": sub_end["p_avg"].mean(),
                f"p_posted_t{n_steps}": sub_end["p_posted_mean"].mean(),
                f"p_leader_t{n_steps}": sub_end["p_leader"].mean(),
                f"uc_branch_t{n_steps}": sub_end["uc_branch"].mean(),
                f"uc_shaikh_t{n_steps}": sub_end["uc_shaikh"].mean(),
                f"uc_okishio_t{n_steps}": sub_end["uc_okishio"].mean(),
                f"kappa_branch_t{n_steps}": sub_end["kappa_branch"].mean(),
                f"kappa_shaikh_t{n_steps}": sub_end["kappa_shaikh"].mean(),
                f"kappa_okishio_t{n_steps}": sub_end["kappa_okishio"].mean(),
                f"r_branch_t{late_start}_{n_steps}_pct": sub_late["r_branch"].mean(),
                f"r_branch_t{n_steps}_pct": sub_end["r_branch"].mean(),
                f"u_branch_t{late_start}_{n_steps}_pct": sub_late["u_branch"].mean(),
                f"u_branch_t{n_steps}_pct": sub_end["u_branch"].mean(),
                f"D_end_t{n_steps}": sub_end["D_t"].mean(),
                "D_boost_vs_trend_pct": ((sub_end["D_t"] / sub_end["D_trend_t"] - 1.0).mean()) * 100.0,
                f"n_firms_t{n_steps}": sub_end["n_firms"].mean(),
                "n_firms_in_4_6_pct": ((sub_end["n_firms"] >= 4) & (sub_end["n_firms"] <= 6)).mean() * 100.0,
                f"n_shaikh_t{n_steps}": sub_end["n_shaikh"].mean(),
                f"n_okishio_t{n_steps}": sub_end["n_okishio"].mean(),
                "shortage_years_mean": per_run_cum["is_shortage"].mean(),
                "cum_entries_mean": per_run_cum["n_entries"].mean(),
                "cum_exits_total_mean": per_run_cum["n_exits_total"].mean(),
                "cum_exits_marg_mean": per_run_cum["n_exits_marg"].mean(),
                "cum_exits_exh_mean": per_run_cum["n_exits_exh"].mean(),
            }
        )

        for label, mask in (
            (f"All periods (t=1..{n_steps})", sub["n_sellers"] >= 0),
            ("Multi-seller periods (n_sellers >= 2)", sub["n_sellers"] >= 2),
        ):
            s_sub = sub[mask]
            disp_rows.append(
                {
                    "regime": reg,
                    "sample": label,
                    "n_obs": len(s_sub),
                    "pct_of_periods": len(s_sub) / len(sub) * 100.0,
                    "sd_posted_mean": s_sub["sd_posted"].mean(),
                    "cv_posted_mean_pct": s_sub["cv_posted"].mean(),
                    "spread_pct_posted_mean": s_sub["spread_pct_posted"].mean(),
                    "sd_sales_mean": s_sub["sd_sales"].mean(),
                    "cv_sales_mean_pct": s_sub["cv_sales"].mean(),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(disp_rows)


def plot_paper_figures(
    df_all: pd.DataFrame,
    df_rob: pd.DataFrame | None,
    n_reps: int,
    n_steps: int,
    figures_dir: str = "figures",
) -> None:
    """Generate all baseline publication figures."""
    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9.8,
            "axes.titlesize": 10.6,
            "axes.labelsize": 9.8,
            "legend.fontsize": 8.4,
        }
    )
    c_neut, c_marx = "#1F5A9E", "#C8102E"

    def regime_agg(df, reg):
        g = df[df["regime"] == reg].groupby("t")
        return (
            g.mean(numeric_only=True),
            g.quantile(0.10, numeric_only=True),
            g.quantile(0.90, numeric_only=True),
        )

    # 1. Main Neutral vs Marx Comparison (8 panels)
    fig, axes = plt.subplots(4, 2, figsize=(14.0, 15.8), sharex=True)
    df_all = df_all.sort_values(["regime", "seed", "t"]).copy()
    df_all["cum_entries"] = df_all.groupby(["regime", "seed"])["n_entries"].cumsum()
    df_all["cum_exits"] = df_all.groupby(["regime", "seed"])["n_exits_total"].cumsum()

    r_star_val = float(df_all["r_star"].iloc[0])
    specs = [
        ("share_shaikh", "(a) Shaikh Market Share $s_{\\mathrm{Shaikh}}(t)$", "Market share (%)", 50.0, "50% Parity"),
        ("p_avg", "(b) Sales-Weighted Mean Price $\\bar{p}(t)$ and Leader Price", "Price", 1.000, "Initial $p_0 = 1.000$"),
        ("uc_branch", "(c) Mean Complete Unit Cost $\\overline{uc}_{\\mathrm{branch}}(t)$", "Unit cost $\\overline{uc}$", 0.672, "Initial $uc_0 = 0.672$"),
        ("kappa_branch", "(d) Mean Capital-Output Ratio $\\overline{\\kappa}_{\\mathrm{branch}}(t)$", "Capital-output ratio $\\overline{\\kappa}$", 3.280, "Initial $\\kappa_0 = 3.280$"),
        ("r_branch", "(e) Realized Branch Profit Rate $r_t^{\\mathrm{branch}}$ (%)", "Profit rate (%)", r_star_val, f"General profit rate $r^* = {r_star_val:.1f}\\%$"),
        ("u_branch", "(f) Aggregate Capacity Utilization $u_{\\mathrm{branch}}(t)$ (%)", "Utilization (%)", 82.0, "Normal target $u_n = 82\\%$"),
        ("n_firms", "(g) Active Firm Population $N_t$", "Active firms $N_t$", None, None),
        ("cum_exits", "(h) Cumulative Firm Entries & Exits", "Cumulative count", None, None),
    ]
    n_mean, n_q10, n_q90 = regime_agg(df_all, "neutral")
    m_mean, m_q10, m_q90 = regime_agg(df_all, "marx")
    t_vals = n_mean.index.to_numpy()

    for ax, (col, title, ylabel, ref_val, ref_lbl) in zip(axes.flat, specs):
        for label, colr, mn, q1, q9 in (
            ("Neutral ($\\mathcal{I}_{\\mathrm{neutral}}$)", c_neut, n_mean, n_q10, n_q90),
            ("Marx-biased ($\\mathcal{I}_{\\mathrm{Marx}}$)", c_marx, m_mean, m_q10, m_q90),
        ):
            if col == "cum_exits":
                reg_short = "neutral" if colr == c_neut else "marx"
                ax.plot(t_vals, mn["cum_exits"], color=colr, lw=2.2, label=f"Exits ({reg_short})")
                ax.plot(t_vals, mn["cum_entries"], color=colr, ls="--", lw=1.8, label=f"Entries ({reg_short})")
            else:
                ax.plot(t_vals, mn[col], color=colr, lw=2.2, label=label)
                ax.fill_between(t_vals, q1[col], q9[col], color=colr, alpha=0.14)
                if col == "p_avg":
                    reg_short = "neutral" if colr == c_neut else "marx"
                    ax.plot(t_vals, mn["p_leader"], color=colr, ls=":", lw=1.4, label=f"$p^{{\\mathrm{{leader}}}}$ ({reg_short})")
        if ref_val is not None:
            ax.axhline(ref_val, color="#2E8B57", ls="--", lw=1.4, label=ref_lbl)
        ax.set_title(title, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)

    axes[3, 0].set_xlabel("Period $t$")
    axes[3, 1].set_xlabel("Period $t$")
    axes[3, 1].set_xlim(1, n_steps)
    fig.suptitle(
        f"Core Paper Baseline (Paired Homogeneous Init, Core Rules Only, N = {n_reps}, T = {n_steps})",
        fontsize=12.5,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(figures_dir, "fig_paper_neutral_vs_marx.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Disaggregated Neutral & Marx (Shaikh vs Okishio)
    for reg, reg_title, fname in (
        ("neutral", "Neutral Regime", "fig_paper_disaggregated_neutral.png"),
        ("marx", "Marx-Biased Regime", "fig_paper_disaggregated_marx.png"),
    ):
        mn, q1, q9 = regime_agg(df_all, reg)
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), sharex=True)
        d_panels = [
            ("r", "(a) Mean Realized Profit Rate (%)", "Profit rate (%)"),
            ("kappa", "(b) Mean Capital-Output Ratio $\\overline{\\kappa}$", "Capital-output ratio $\\overline{\\kappa}$"),
            ("uc", "(c) Mean Complete Unit Cost $\\overline{uc}$", "Complete unit cost $\\overline{uc}$"),
            ("p", "(d) Mean Posted Price $p$", "Posted price $p$"),
        ]
        for ax, (prefix, title, ylabel) in zip(axes.flat, d_panels):
            for rule, lbl, colr in (("shaikh", "Shaikh firms", "#1f77b4"), ("okishio", "Okishio firms", "#d62728")):
                cname = f"{prefix}_{rule}"
                ax.plot(t_vals, mn[cname], color=colr, lw=2.0, label=f"{lbl} (mean)")
                ax.fill_between(t_vals, q1[cname], q9[cname], color=colr, alpha=0.18, label=f"{lbl} (10th-90th pct)")
            ax.set_title(title, fontweight="bold", pad=6)
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.3)
            ax.legend(loc="best", fontsize=8.0, framealpha=0.9)
        axes[1, 0].set_xlabel("Period $t$")
        axes[1, 1].set_xlabel("Period $t$")
        axes[1, 0].set_xlim(1, n_steps)
        fig.suptitle(
            f"Disaggregated Dynamics: Shaikh vs Okishio — {reg_title} (N = {n_reps}, T = {n_steps})",
            fontsize=12.5,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(figures_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 3. Price Dispersion Secondary Metrics
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), sharex=True)
    for col_idx, (reg, reg_lbl, colr) in enumerate((("neutral", "Neutral", c_neut), ("marx", "Marx-Biased", c_marx))):
        sub = df_all[df_all["regime"] == reg]
        sub_multi = sub[sub["n_sellers"] >= 2]
        g_all = sub.groupby("t").mean(numeric_only=True)
        g_mul = sub_multi.groupby("t").mean(numeric_only=True)

        ax_sd = axes[0, col_idx]
        ax_sd.plot(g_all.index, g_all["sd_posted"], color=colr, lw=2.0, label="Posted prices SD (all periods)")
        ax_sd.plot(g_all.index, g_all["sd_sales"], color=colr, ls="--", lw=1.8, label="Sales-weighted SD (all periods)")
        ax_sd.plot(g_mul.index, g_mul["sd_posted"], color="#111111", ls=":", lw=1.6, label="Posted SD ($N_{\\mathrm{sellers}} \\geq 2$)")
        ax_sd.set_title(f"{reg_lbl}: Price Dispersion (Std. Dev.)", fontweight="bold")
        ax_sd.set_ylabel("Standard deviation")
        ax_sd.grid(True, ls="--", alpha=0.3)
        ax_sd.legend(loc="best", fontsize=8.0)

        ax_cv = axes[1, col_idx]
        ax_cv.plot(g_all.index, g_all["cv_posted"], color=colr, lw=2.0, label="Posted prices CV (%) [all]")
        ax_cv.plot(g_all.index, g_all["cv_sales"], color=colr, ls="--", lw=1.8, label="Sales-weighted CV (%) [all]")
        ax_cv.plot(g_mul.index, g_mul["spread_pct_posted"], color="#2E8B57", ls="-.", lw=1.5, label="Max-min spread (%) [$N_{\\mathrm{sellers}} \\geq 2$]")
        ax_cv.set_title(f"{reg_lbl}: Relative Dispersion (CV & Max-Min Spread)", fontweight="bold")
        ax_cv.set_xlabel("Period $t$")
        ax_cv.set_ylabel("Percent (%)")
        ax_cv.set_xlim(1, n_steps)
        ax_cv.grid(True, ls="--", alpha=0.3)
        ax_cv.legend(loc="best", fontsize=8.0)

    fig.suptitle(
        f"Secondary Metric: Price Dispersion (Posted vs Sales-Weighted, N = {n_reps}, T = {n_steps})",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(figures_dir, "fig_paper_price_dispersion.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Seed 42 Illustrative Trajectory (Neutral vs Marx)
    seed_ill = int(EXPERIMENT_PARAMS["illustrative_seed"])
    s42 = df_all[df_all["seed"] == seed_ill]
    if not s42.empty:
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex=True)
        for col_idx, (reg, rtitle) in enumerate((("neutral", f"Neutral Regime (Seed {seed_ill})"), ("marx", f"Marx-Biased Regime (Seed {seed_ill})"))):
            ss = s42[s42["regime"] == reg]
            axes[0, col_idx].plot(ss["t"], ss["p_avg"], color="#111111", lw=2.0, label="Weighted mean price $\\bar{p}(t)$")
            axes[0, col_idx].plot(ss["t"], ss["uc_shaikh"], color="#1f77b4", ls="--", lw=1.6, label="$\\overline{uc}_{\\mathrm{Shaikh}}$")
            axes[0, col_idx].plot(ss["t"], ss["uc_okishio"], color="#d62728", ls=":", lw=1.6, label="$\\overline{uc}_{\\mathrm{Okishio}}$")
            axes[0, col_idx].set_title(f"{rtitle}: Price & Unit Costs", fontweight="bold")
            axes[0, col_idx].set_ylabel("Price / Cost")
            axes[0, col_idx].grid(True, ls="--", alpha=0.3)
            axes[0, col_idx].legend(loc="best", fontsize=8.0)

            axes[1, col_idx].plot(ss["t"], ss["share_shaikh"], color="#1f77b4", lw=2.0, label="Shaikh share (%)")
            ax_r = axes[1, col_idx].twinx()
            ax_r.plot(ss["t"], ss["n_firms"], color="#555555", ls="--", lw=1.5, label="Active firms $N_t$")
            axes[1, col_idx].axhline(50.0, color="#2E8B57", ls=":", lw=1.2)
            axes[1, col_idx].set_title(f"{rtitle}: Market Share & Population", fontweight="bold")
            axes[1, col_idx].set_xlabel("Period $t$")
            axes[1, col_idx].set_ylabel("Shaikh share (%)")
            ax_r.set_ylabel("Active firms $N_t$")
            axes[1, col_idx].set_xlim(1, n_steps)
            axes[1, col_idx].grid(True, ls="--", alpha=0.3)
        fig.suptitle(
            f"Illustrative Single Replication (Seed {seed_ill}, Homogeneous Core Baseline, T = {n_steps})",
            fontsize=12.5,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(figures_dir, "fig_paper_seed42_illustration.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 5. Price Trajectories Figure (Panel a: seed 42 firm prices & floors; Panel b: ensemble regimes)
    df_firms, df_agg = run_detailed_simulation(
        n_steps=n_steps, innovation_regime="marx", seed=seed_ill
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 9.2), sharex=True)
    for _, grp in df_firms.groupby("firm_id"):
        grp = grp.sort_values("t")
        rule = grp["rule"].iloc[0]
        c = c_marx if rule == "shaikh" else c_neut
        ls = "-" if rule == "shaikh" else "--"
        ax1.plot(grp["t"], grp["p_i"], color=c, linestyle=ls, linewidth=1.0, alpha=0.45, zorder=2)
        if grp["exited"].iloc[-1]:
            ax1.scatter(grp["t"].iloc[-1], grp["p_i"].iloc[-1], color=c, marker="x", s=32, alpha=0.85, zorder=4)
    ax1.plot(df_agg["t"], df_agg["p_leader_in"], color="#111111", linewidth=2.3, label="Leader price $p^{\\mathrm{leader}}(t)$", zorder=5)
    ax1.plot(df_agg["t"], df_agg["p_avg_t"], color="#444444", linestyle="-.", linewidth=1.9, label="Weighted mean price $\\bar{p}(t)$", zorder=5)
    ax1.plot(df_agg["t"], df_agg["uc_shaikh"], color=c_marx, linestyle=":", linewidth=2.1, label="Floor $p^{\\mathrm{repro}}_{\\mathrm{Shaikh}}(t) = \\overline{uc}_{\\mathrm{Shaikh}}(t)$", zorder=5)
    ax1.plot(df_agg["t"], df_agg["uc_okishio"], color=c_neut, linestyle=":", linewidth=2.1, label="Floor $p^{\\mathrm{repro}}_{\\mathrm{Okishio}}(t) = \\overline{uc}_{\\mathrm{Okishio}}(t)$", zorder=5)
    custom_a = [
        Line2D([0], [0], color=c_marx, lw=1.2, ls="-", alpha=0.65, label="Individual prices $p_i(t)$ — Shaikh"),
        Line2D([0], [0], color=c_neut, lw=1.2, ls="--", alpha=0.65, label="Individual prices $p_i(t)$ — Okishio"),
        Line2D([0], [0], color="gray", marker="x", ls="None", ms=6, label="Firm exit"),
    ]
    handles_a, _ = ax1.get_legend_handles_labels()
    ax1.set_title(f"(a) Individual Price Trajectories $p_i(t)$ and Reproduction Floors (Marx-Biased, seed={seed_ill})", fontweight="bold", pad=8)
    ax1.set_ylabel("Price and cost floor $p^{\\mathrm{repro}}$")
    ax1.legend(handles=custom_a + handles_a, loc="upper right", ncol=2, framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.3)

    for reg, col, lbl in (
        ("neutral", c_neut, "Neutral Regime ($\\mathcal{I}_{\\mathrm{neutral}}$)"),
        ("marx", c_marx, "Marx-Biased Regime ($\\mathcal{I}_{\\mathrm{Marx}}$)"),
    ):
        sub = df_all[df_all["regime"] == reg]
        grp = sub.groupby("t")
        tv = grp["t"].first().values
        ax2.plot(tv, grp["p_avg"].mean().values, color=col, linewidth=2.3, label=f"Mean price $\\bar{{p}}(t)$ — {lbl}", zorder=4)
        ax2.fill_between(tv, grp["p_avg"].quantile(0.10).values, grp["p_avg"].quantile(0.90).values, color=col, alpha=0.14, zorder=2)
        ax2.plot(tv, grp["p_leader"].mean().values, color=col, linestyle=":", linewidth=1.8, label=f"Leader price $p^{{\\mathrm{{leader}}}}(t)$ — {lbl}", zorder=3)
    ax2.axhline(1.000, color="#2E8B57", linestyle="--", linewidth=1.4, label="Initial price $p_0 = 1.000$", zorder=2)
    ax2.set_title(f"(b) Price Trajectory Comparison: Neutral vs Marx-Biased (Means and 10th-90th pct bands, N = {n_reps})", fontweight="bold", pad=8)
    ax2.set_xlabel("Period $t$")
    ax2.set_ylabel("Price")
    ax2.set_xlim(1, n_steps)
    ax2.legend(loc="best", ncol=2, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_price_trajectories.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 6. Separate Staircase Robustness Figure
    if df_rob is not None and not df_rob.empty:
        r_n_m, r_n_q1, r_n_q9 = regime_agg(df_rob, "neutral")
        r_m_m, r_m_q1, r_m_q9 = regime_agg(df_rob, "marx")
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), sharex=True)
        for ax, (col, title, ylabel) in zip(
            axes.flat,
            [
                ("share_shaikh", "(a) Shaikh Market Share (%)", "Market share (%)"),
                ("p_avg", "(b) Sales-Weighted Mean Price", "Price"),
                ("kappa_branch", "(c) Mean Capital-Output Ratio", "$\\overline{\\kappa}$"),
                ("n_firms", "(d) Active Firm Population $N_t$", "Active firms"),
            ],
        ):
            ax.plot(t_vals, r_n_m[col], color=c_neut, lw=2.0, label="Neutral (Staircase)")
            ax.fill_between(t_vals, r_n_q1[col], r_n_q9[col], color=c_neut, alpha=0.14)
            ax.plot(t_vals, r_m_m[col], color=c_marx, lw=2.0, label="Marx-Biased (Staircase)")
            ax.fill_between(t_vals, r_m_q1[col], r_m_q9[col], color=c_marx, alpha=0.14)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.3)
            ax.legend(loc="best", fontsize=8.0)
        axes[1, 0].set_xlabel("Period $t$")
        axes[1, 1].set_xlabel("Period $t$")
        axes[1, 0].set_xlim(1, n_steps)
        fig.suptitle(
            f"Separate Robustness Check: 5-Step Paired Cost Ladder (N = {n_reps}, T = {n_steps})",
            fontsize=12.5,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(figures_dir, "fig_robustness_staircase.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 7. Endogenous Demand Expansion D(t) vs Trend D_trend(t)
    df_all["D_boost_pct"] = (df_all["D_t"] / df_all["D_trend_t"] - 1.0) * 100.0
    n_dm, n_dq1, n_dq9 = regime_agg(df_all, "neutral")
    m_dm, m_dq1, m_dq9 = regime_agg(df_all, "marx")
    s42_n_d = df_all[(df_all["seed"] == seed_ill) & (df_all["regime"] == "neutral")].sort_values("t")
    s42_m_d = df_all[(df_all["seed"] == seed_ill) & (df_all["regime"] == "marx")].sort_values("t")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.4), sharex=True)
    ax1.plot(t_vals, n_dm["D_trend_t"], color="#555555", ls="--", lw=1.8, label="Exogenous trend $D_0 (1 + g_D)^t$")
    ax1.plot(t_vals, n_dm["D_t"], color=c_neut, lw=1.8, alpha=0.65, label="Neutral mean $D(t)$")
    ax1.fill_between(t_vals, n_dq1["D_t"], n_dq9["D_t"], color=c_neut, alpha=0.14, label="Neutral 10th-90th pct")
    if not s42_n_d.empty:
        ax1.plot(s42_n_d["t"], s42_n_d["D_t"], color=c_neut, lw=1.45, ls="-", label=f"Neutral seed {seed_ill}")
    ax1.plot(t_vals, m_dm["D_t"], color=c_marx, lw=1.8, alpha=0.65, label="Marx-Biased mean $D(t)$")
    ax1.fill_between(t_vals, m_dq1["D_t"], m_dq9["D_t"], color=c_marx, alpha=0.14, label="Marx-Biased 10th-90th pct")
    if not s42_m_d.empty:
        ax1.plot(s42_m_d["t"], s42_m_d["D_t"], color=c_marx, lw=1.45, ls="-", label=f"Marx-Biased seed {seed_ill}")
    ax1.set_title("(a) Endogenous Demand $D(t)$ vs Trend", fontweight="bold")
    ax1.set_xlabel("Year $t$")
    ax1.set_ylabel("Aggregate demand volume")
    ax1.grid(True, ls="--", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=7.6)

    ax2.plot(t_vals, n_dm["D_boost_pct"], color=c_neut, lw=1.9, alpha=0.65, label="Neutral mean (%)")
    ax2.fill_between(t_vals, n_dq1["D_boost_pct"], n_dq9["D_boost_pct"], color=c_neut, alpha=0.14, label="Neutral 10th-90th pct")
    if not s42_n_d.empty:
        ax2.plot(s42_n_d["t"], s42_n_d["D_boost_pct"], color=c_neut, lw=1.45, label=f"Neutral seed {seed_ill} (endogenous cycles)")
    ax2.plot(t_vals, m_dm["D_boost_pct"], color=c_marx, lw=1.9, alpha=0.65, label="Marx-Biased mean (%)")
    ax2.fill_between(t_vals, m_dq1["D_boost_pct"], m_dq9["D_boost_pct"], color=c_marx, alpha=0.14, label="Marx-Biased 10th-90th pct")
    if not s42_m_d.empty:
        ax2.plot(s42_m_d["t"], s42_m_d["D_boost_pct"], color=c_marx, lw=1.45, label=f"Marx-Biased seed {seed_ill} (endogenous cycles)")
    ax2.axhline(0.0, color="#555555", ls=":", lw=1.3, label="Trend benchmark (0%)")
    ax2.set_title("(b) Adaptive Price-Induced Demand Cycles $(\\tilde{p}(t-1) / p_{\\min}(t-1))^\\eta - 1$ (%)", fontweight="bold")
    ax2.set_xlabel("Year $t$")
    ax2.set_ylabel("Deviation from trend (%)")
    ax2.set_xlim(1, n_steps)
    ax2.grid(True, ls="--", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=7.5, ncol=2)

    fig.suptitle(
        f"Endogenous Demand Dynamics (Adaptive Price Memory $\\alpha_p = {BASELINE_PARAMS['alpha_p']}, \\eta = {BASELINE_PARAMS['eta']}$, N = {n_reps}/regime)",
        fontsize=12.0,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(figures_dir, "fig_paper_endogenous_demand.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired Neutral vs Marx-biased baseline Monte Carlo.")
    parser.add_argument("--n-reps", type=int, default=int(EXPERIMENT_PARAMS["n_replications"]), help="Replications per regime.")
    parser.add_argument("--n-steps", type=int, default=int(EXPERIMENT_PARAMS["n_steps"]), help="Annual simulation horizon.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel worker count for joblib.")
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--figures-dir", type=str, default=str(ROOT / "figures"))
    parser.add_argument("--skip-staircase", action="store_true", help="Skip separate 5-step staircase robustness batch.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip saving PNG figures.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    n_reps = int(args.n_reps)
    n_steps = int(args.n_steps)
    seeds = list(range(n_reps))

    t0 = time.perf_counter()
    print(f"=== RUNNING CORE PAPER BASELINE (Homogeneous Paired Init, N={n_reps}/regime, T={n_steps}) ===")
    res_n = Parallel(n_jobs=args.n_jobs)(delayed(simulate_one_run)(s, "neutral", n_steps, False) for s in seeds)
    res_m = Parallel(n_jobs=args.n_jobs)(delayed(simulate_one_run)(s, "marx", n_steps, False) for s in seeds)

    df_all = pd.concat([r[0] for r in res_n + res_m], ignore_index=True)
    df_audit = pd.DataFrame([r[1] for r in res_n + res_m])

    df_all.to_csv(os.path.join(args.results_dir, f"paper_baseline_raw_{n_reps}.csv"), index=False)
    df_audit.to_csv(os.path.join(args.results_dir, "paper_ucs_feasibility_audit.csv"), index=False)

    df_sum, df_disp = summarize_experiment(df_all, n_steps)
    df_sum.to_csv(os.path.join(args.results_dir, "paper_baseline_summary.csv"), index=False)
    df_disp.to_csv(os.path.join(args.results_dir, "paper_price_dispersion_summary.csv"), index=False)

    df_rob = None
    df_rob_sum = None
    if not args.skip_staircase:
        print(f"=== RUNNING SEPARATE STAIRCASE ROBUSTNESS (5 Matched Steps, N={n_reps}/regime, T={n_steps}) ===")
        rob_n = Parallel(n_jobs=args.n_jobs)(delayed(simulate_one_run)(s, "neutral", n_steps, True) for s in seeds)
        rob_m = Parallel(n_jobs=args.n_jobs)(delayed(simulate_one_run)(s, "marx", n_steps, True) for s in seeds)
        df_rob = pd.concat([r[0] for r in rob_n + rob_m], ignore_index=True)
        df_rob.to_csv(os.path.join(args.results_dir, f"robustness_staircase_raw_{n_reps}.csv"), index=False)
        df_rob_sum, _ = summarize_experiment(df_rob, n_steps)
        df_rob_sum.to_csv(os.path.join(args.results_dir, "robustness_staircase_summary.csv"), index=False)

    if not args.skip_figures:
        plot_paper_figures(df_all, df_rob, n_reps, n_steps, figures_dir=args.figures_dir)

    feas_summary = {}
    any_ucs_violation = bool((df_audit["ucs_neg_vintages"] > 0).any())
    for reg in ("neutral", "marx"):
        sub_a = df_audit[df_audit["regime"] == reg]
        n_fail_runs = int((sub_a["ucs_neg_vintages"] > 0).sum())
        feas_summary[reg] = {
            "runs_with_negative_ucs": n_fail_runs,
            "pct_runs_failed": n_fail_runs / n_reps * 100.0,
            "total_negative_vintage_obs": int(sub_a["ucs_neg_vintages"].sum()),
            "negative_shaikh_vintage_obs": int(sub_a["ucs_neg_shaikh_vintages"].sum()),
            "negative_okishio_vintage_obs": int(sub_a["ucs_neg_okishio_vintages"].sum()),
            "total_vintage_obs": int(sub_a["ucs_total_vintages"].sum()),
            "min_ucs_observed": float(sub_a["ucs_min"].min()),
            "earliest_negative_t": int(sub_a["first_neg_t"].dropna().min()) if sub_a["first_neg_t"].notna().any() else None,
        }

    src_files = [
        str(ROOT / "src" / "calibration.py"),
        str(ROOT / "src" / "innovation.py"),
        str(ROOT / "src" / "agents.py"),
        str(ROOT / "src" / "model.py"),
        str(ROOT / "experiments" / "run_baseline.py"),
    ]
    manifest = collect_provenance(src_files)
    manifest["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    manifest["baseline_economically_validated"] = not any_ucs_violation
    manifest["feasibility_audit"] = feas_summary
    with open(os.path.join(args.results_dir, "paper_baseline_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n=== MAIN BASELINE SUMMARY (t=1..{n_steps} and t={n_steps}, N={n_reps}/regime) ===")
    print(df_sum.to_string(index=False))
    print("\n=== SECONDARY: PRICE DISPERSION SUMMARY ===")
    print(df_disp.to_string(index=False))
    print("\n=== FEASIBILITY AUDIT: ucs_v = uc_v - kappa_v/(T*u_n) >= 0 ===")
    print(json.dumps(feas_summary, indent=2))
    print(f"-> baseline_economically_validated = {not any_ucs_violation}")
    if df_rob_sum is not None:
        print("\n=== SEPARATE ROBUSTNESS SUMMARY (5-Step Staircase) ===")
        print(df_rob_sum.to_string(index=False))
    print(f"\nCompleted in {time.perf_counter() - t0:.2f} s.")


if __name__ == "__main__":
    main()
