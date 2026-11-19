"""
Two-stage sensitivity analysis and reporting:
1) Global Morris elementary-effects screening across all 27 exogenous parameters
   (560 configs x 12 seeds x 2 regimes = 13,440 runs).
2) Univariate 5-point sweeps + 4 structural stress tests on 64 fresh out-of-sample seeds
   (24 configs x 64 seeds x 2 regimes = 3,072 runs).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as pkg_version
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr, t as student_t

from src.calibration import (
    MORRIS_FACTORS as FACTORS,
    baseline_parameters,
    init_paired_homogeneous_firms,
)
from src.model import RealCompetitionModel


NAMES = [f[0] for f in FACTORS]
BASE = {name: base for name, _lo, _hi, base in FACTORS}
METRICS = [
    "share_final",
    "share_post",
    "profit_post",
    "price_final",
    "kappa_final",
    "util_post",
    "shortage_pct",
    "entries",
    "firms_final",
    "invalid_prime",
    "min_prime",
    "extinct",
]

LABELS = {
    "D": "Demande initiale D_0", "g_D": "Croissance tendancielle g_D",
    "eta": "Élasticité-prix η", "alpha_p": "Mémoire adaptative α_p",
    "u_n": "Utilisation normale u_n", "T": "Durée de vie T (ans)",
    "gamma": "Hausse sous-capacité γ", "delta_cost": "Sous-cotation coût δ_cost",
    "eps_0": "Sous-cotation de base ε_0", "mu_z": "Hausse en pénurie μ_z",
    "sigma_z": "Dispersion pénurie σ_z", "lam": "Réallocation demande λ",
    "beta_r": "Sensibilité profit D&L β_r", "beta_u": "Sensibilité utilisation D&L β_u",
    "R": "Amplitude innovations R", "ellipse_ratio": "Rapport des axes (a/b)",
    "marx_theta": "Angle de l'ellipse θ", "marx_mu_uc": "Dérive coût premier μ_ucs",
    "marx_mu_kappa": "Dérive capital μ_κ", "gamma_thresh": "Seuil consolidation γ_thresh",
    "r_star": "Taux profit général r*", "psi": "Vitesse d'entrée ψ",
    "f_0": "Part cible entrants f_0", "K_0": "Capacité base entrants K_0",
    "B_0": "Budget base entrants B_0", "s_min": "Seuil de sortie s_min",
    "tau_grace": "Durée de grâce τ_grace",
}
OUTPUT_LABELS = {
    "share_final": "Part Shaikh finale (%)", "share_post": "Part Shaikh moyenne (%)",
    "profit_post": "Profit réalisé moyen (%)", "price_final": "Prix final",
    "kappa_final": "Intensité capitalistique finale", "util_post": "Utilisation moyenne (%)",
    "entries": "Entrées cumulées", "invalid_prime": "Présence de coûts premiers négatifs",
}
FIG_LABELS = {
    "D": "Initial demand D_0", "g_D": "Trend demand growth g_D",
    "eta": "Price elasticity η", "alpha_p": "Adaptive price memory α_p",
    "u_n": "Normal utilization u_n", "T": "Service life T (years)",
    "gamma": "Shortfall markup γ", "delta_cost": "Cost undercutting δ_cost",
    "eps_0": "Baseline undercutting ε_0", "mu_z": "Shortage markup mean μ_z",
    "sigma_z": "Shortage markup std σ_z", "lam": "Demand reallocation λ",
    "beta_r": "Profit accumulation β_r", "beta_u": "Utilization accumulation β_u",
    "R": "Innovation radius R", "ellipse_ratio": "Ellipse axis ratio (a/b)",
    "marx_theta": "Ellipse rotation angle θ", "marx_mu_uc": "Prime-cost drift μ_ucs",
    "marx_mu_kappa": "Capital-using drift μ_κ", "gamma_thresh": "Consolidation threshold γ_thresh",
    "r_star": "General profit rate r*", "psi": "Entry speed ψ",
    "f_0": "Entrant target share f_0", "K_0": "Entrant base capacity K_0",
    "B_0": "Entrant base budget B_0", "s_min": "Exit share floor s_min",
    "tau_grace": "Grace period τ_grace",
}
FIG_OUTPUT_LABELS = {
    "share_final": "Final Shaikh share (%)",
    "share_post": "Post-burn-in Shaikh share (%)",
    "profit_post": "Realized profit rate (%)",
    "price_final": "Final mean price",
    "kappa_final": "Final capital-output ratio",
    "util_post": "Mean utilization (%)",
    "entries": "Cumulative entries",
    "invalid_prime": "Negative prime-cost share",
}
REGIME_LABELS = {"neutral": "Neutral", "marx": "Marx-biased"}
COLORS = {"neutral": "#245C9C", "marx": "#B51E35"}


def decode(values: dict[str, float]) -> dict:
    """Decode sampled Morris factors into model constructor overrides."""
    params = dict(values)
    ratio = params.pop("ellipse_ratio")
    scale = params["R"] / BASE["R"]
    params["marx_a"] = 0.020 * scale * np.sqrt(ratio / 4.0)
    params["marx_b"] = 0.005 * scale / np.sqrt(ratio / 4.0)
    for name in ("T", "tau_grace"):
        params[name] = int(np.rint(params[name]))
    return params


def simulate(
    values: dict[str, float],
    regime: str,
    seed: int,
    n_steps: int = 120,
    reverse: bool = False,
    normal_price: bool = False,
) -> dict[str, float]:
    """Run one configuration and return summary metrics across horizons and feasibility audit."""
    params = decode(values)
    m = RealCompetitionModel(
        **baseline_parameters(**params, innovation_regime=regime, seed=int(seed))
    )
    capacity = m.D / m.u_n / 10.0
    price0 = 0.672 + 0.1 * 3.28 / m.u_n if normal_price else 1.0
    init_paired_homogeneous_firms(m, seed=seed)
    for firm in m.agents:
        firm.price = price0
        firm.budget = 30.0
        firm.u = m.u_n
        firm.u_pre = m.u_n
        firm.r_bar_pre = firm.normal_profit_rate()
        firm.r_real = (
            firm.u * ((firm.price - firm.mean_uc) / firm.mean_kappa)
            if (firm.total_capacity > 0.0 and firm.mean_kappa > 0.0)
            else 0.0
        )
        firm.r_real_pre = firm.r_real
        for vintage in firm.vintages:
            vintage.k_init = vintage.k = capacity
        firm.k_surv = firm.total_capacity
        firm.k_pre = firm.total_capacity
        if reverse:
            firm.rule = "okishio" if firm.rule == "shaikh" else "shaikh"

    series = []
    min_prime = float("inf")
    n_entries = 0
    any_extinction = False
    for _period in range(1, n_steps + 1):
        m.current_period += 1
        firms = list(m.agents)
        if not firms:
            any_extinction = True
            series.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            continue
        m.step_0_demand()
        cap = sum(f.total_capacity for f in firms)
        capital = sum(v.kappa * v.k for f in firms for v in f.vintages)
        vintages = [v for f in firms for v in f.vintages]
        if vintages:
            min_prime = min(min_prime, min(v.uc - v.kappa / (m.T * m.u_n) for v in vintages))
        m.step_1_pricing()
        m.step_2_demand_allocation()
        m.step_3_production_and_accounting()
        sales = sum(f.sales for f in firms)
        share = sum(f.sales for f in firms if f.rule == "shaikh") / sales if sales else 0.0
        price = sum(f.price * f.sales for f in firms) / sales if sales else min(f.price for f in firms)
        kappa = capital / cap if cap else 0.0
        util = sales / cap if cap else 0.0
        n_firms = len(firms)
        shortage = float(cap < m.D)
        m.step_4_scrapping()
        m.step_5_innovation_and_investment()
        m.step_6_entry_and_exit(capital)
        n_entries += m.last_entries
        for firm in m.agents:
            for v in firm.vintages:
                min_prime = min(min_prime, v.uc - v.kappa / (m.T * m.u_n))
        series.append([share, m.r_branch, price, kappa, util, shortage, n_firms])

    a = np.array(series)
    burn_in = min(40, max(0, n_steps - 1))
    post = a[burn_in:]
    return dict(
        share_final=100.0 * a[-1, 0],
        share_post=100.0 * post[:, 0].mean(),
        profit_post=100.0 * post[:, 1].mean(),
        price_final=a[-1, 2],
        kappa_final=a[-1, 3],
        util_post=100.0 * post[:, 4].mean(),
        shortage_pct=100.0 * a[:, 5].mean(),
        entries=n_entries,
        firms_final=a[-1, 6],
        invalid_prime=float(min_prime < -1e-10),
        min_prime=min_prime,
        extinct=float(any_extinction),
    )


def morris_design(trajectories: int = 20, levels: int = 6, seed: int = 20261117) -> pd.DataFrame:
    """Generate randomized OAT Morris trajectories across the 27-dimensional factor space."""
    assert levels % 2 == 0
    rng = np.random.default_rng(seed)
    delta = levels / (2 * (levels - 1))
    rows = []
    for trajectory in range(trajectories):
        signs = rng.choice([-1, 1], len(NAMES))
        x = rng.integers(0, levels // 2, len(NAMES)) / (levels - 1)
        x += (signs < 0) * delta
        order = rng.permutation(len(NAMES))
        for step in range(len(NAMES) + 1):
            row = dict(
                config_id=trajectory * (len(NAMES) + 1) + step,
                trajectory=trajectory,
                step=step,
                changed="" if step == 0 else NAMES[order[step - 1]],
                signed_delta=0.0 if step == 0 else signs[order[step - 1]] * delta,
            )
            for j, (name, lo, hi, _base) in enumerate(FACTORS):
                row[name] = lo + x[j] * (hi - lo)
                row[f"z_{name}"] = x[j]
            rows.append(row)
            if step < len(NAMES):
                j = order[step]
                x[j] += signs[j] * delta
    return pd.DataFrame(rows)


def evaluate_config(row: dict, seeds: list[int]) -> list[dict]:
    """Evaluate a single design point across common seeds and both regimes."""
    values = {name: row[name] for name in NAMES}
    records = []
    for seed in seeds:
        for regime in ("neutral", "marx"):
            try:
                result = simulate(
                    values,
                    regime,
                    seed,
                    n_steps=int(row.get("n_steps", 120)),
                    reverse=bool(row.get("reverse", False)),
                    normal_price=bool(row.get("normal_price", False)),
                )
                records.append(dict(config_id=row["config_id"], seed=seed, regime=regime, error="", **result))
            except Exception as exc:
                records.append(dict(config_id=row["config_id"], seed=seed, regime=regime, error=f"{type(exc).__name__}: {exc}"))
    return records


def run_batch(design: pd.DataFrame, seeds: list[int], output: Path, prefix: str, jobs: int = 4) -> pd.DataFrame:
    """Execute resumable chunked parallel batch and assemble validated raw outputs."""
    chunks = output / f"{prefix}_chunks"
    chunks.mkdir(exist_ok=True)
    todo = [
        row
        for row in design.to_dict("records")
        if not (chunks / f"{int(row['config_id']):05d}.csv").exists()
    ]
    started = time.perf_counter()
    if todo:
        with parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=1):
            results = Parallel(return_as="generator_unordered")(
                delayed(evaluate_config)(row, seeds) for row in todo
            )
            for i, records in enumerate(results, 1):
                cid = int(records[0]["config_id"])
                pd.DataFrame(records).to_csv(chunks / f"{cid:05d}.csv", index=False)
                if i == 1 or i % 10 == 0 or i == len(todo):
                    print(f"{prefix}: {i}/{len(todo)} configurations; {time.perf_counter() - started:.1f}s", flush=True)
    raw = pd.concat(
        [pd.read_csv(chunks / f"{int(cid):05d}.csv", keep_default_na=False) for cid in design.config_id],
        ignore_index=True,
    )
    raw.to_csv(output / f"{prefix}_raw.csv", index=False)
    failures = raw[raw.error != ""]
    failures.to_csv(output / f"{prefix}_failures.csv", index=False)
    if len(failures):
        raise RuntimeError(f"{len(failures)} failures retained in {prefix}_failures.csv")
    assert len(raw) == len(design) * len(seeds) * 2
    assert np.isfinite(raw[METRICS].to_numpy(dtype=float)).all()
    return raw


def elementary_effects(design: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Compute paired seed-wise Morris elementary effects per normalized input unit."""
    index = raw.set_index(["config_id", "seed", "regime"])
    records = []
    for row in design[design.step > 0].to_dict("records"):
        cid = int(row["config_id"])
        for regime in ("neutral", "marx"):
            for seed in sorted(raw.seed.unique()):
                change = (
                    index.loc[(cid, seed, regime), METRICS].astype(float)
                    - index.loc[(cid - 1, seed, regime), METRICS].astype(float)
                ) / row["signed_delta"]
                records.append(
                    dict(
                        trajectory=int(row["trajectory"]),
                        parameter=row["changed"],
                        regime=regime,
                        seed=seed,
                        **change.to_dict(),
                    )
                )
    return pd.DataFrame(records)


def summarize_effects(effects: pd.DataFrame, bootstraps: int = 1000) -> pd.DataFrame:
    """Compute mu, mu*, sigma, and two-way bootstrap CIs across trajectories and seeds."""
    rng = np.random.default_rng(20261118)
    rows = []
    for (regime, parameter), group in effects.groupby(["regime", "parameter"]):
        for metric in METRICS:
            matrix = group.pivot(index="trajectory", columns="seed", values=metric).to_numpy()
            ee = matrix.mean(axis=1)
            tr = rng.integers(0, matrix.shape[0], (bootstraps, matrix.shape[0]))
            sd = rng.integers(0, matrix.shape[1], (bootstraps, matrix.shape[1]))
            boot = np.abs(matrix[tr[:, :, None], sd[:, None, :]].mean(axis=2)).mean(axis=1)
            half = matrix[:, : matrix.shape[1] // 2].mean(axis=1)
            mean_noise = (matrix.std(axis=1, ddof=1) / np.sqrt(matrix.shape[1])).mean()
            rows.append(
                dict(
                    regime=regime,
                    parameter=parameter,
                    metric=metric,
                    mu=ee.mean(),
                    mu_star=np.abs(ee).mean(),
                    sigma=ee.std(ddof=1),
                    ci_low=np.quantile(boot, 0.025),
                    ci_high=np.quantile(boot, 0.975),
                    mu_star_half_seeds=np.abs(half).mean(),
                    mu_star_half_trajectories=np.abs(ee[: len(ee) // 2]).mean(),
                    mean_ee_mc_se=mean_noise,
                )
            )
    return pd.DataFrame(rows)


def analyze(output: Path) -> pd.DataFrame:
    """Compute Morris indices, rank stability diagnostics, and configuration aggregates."""
    design = pd.read_csv(output / "morris_design.csv")
    raw = pd.read_csv(output / "morris_raw.csv", keep_default_na=False)
    effects = elementary_effects(design, raw)
    effects.to_csv(output / "elementary_effects.csv", index=False)
    summary = summarize_effects(effects)
    summary.to_csv(output / "morris_indices.csv", index=False)
    stability = []
    for (regime, metric), group in summary.groupby(["regime", "metric"]):
        for alternate in ("mu_star_half_seeds", "mu_star_half_trajectories"):
            corr = spearmanr(group.mu_star, group[alternate]).statistic if group.mu_star.std() else np.nan
            stability.append(dict(regime=regime, metric=metric, comparison=alternate, rank_spearman=corr))
    pd.DataFrame(stability).to_csv(output / "rank_stability.csv", index=False)
    raw.groupby(["config_id", "regime"])[METRICS].agg(["mean", "std"]).to_csv(output / "configuration_summary.csv")
    return summary


def followup_design(summary: pd.DataFrame) -> pd.DataFrame:
    """Construct Stage 2 univariate 5-point sweeps + 4 structural stress tests."""
    top = summary[(summary.regime == "marx") & (summary.metric == "share_final")].nlargest(4, "mu_star").parameter.tolist()
    rows = []
    for parameter in top:
        _, lo, hi, base = next(f for f in FACTORS if f[0] == parameter)
        for value in sorted({lo, lo + 0.25 * (hi - lo), base, lo + 0.75 * (hi - lo), hi}):
            rows.append(
                dict(
                    **(BASE | {parameter: value}),
                    parameter=parameter,
                    value=value,
                    scenario="curve",
                    n_steps=120,
                    reverse=False,
                    normal_price=False,
                )
            )
    for scenario, kwargs in (
        ("baseline", {}),
        ("reversed_types", {"reverse": True}),
        ("horizon_300", {"n_steps": 300}),
        ("initial_normal_return_10pct", {"normal_price": True}),
    ):
        rows.append(
            dict(
                **BASE,
                parameter="scenario",
                value=0.0,
                scenario=scenario,
                **(dict(n_steps=120, reverse=False, normal_price=False) | kwargs),
            )
        )
    for i, row in enumerate(rows):
        row["config_id"] = i
    return pd.DataFrame(rows)


def archive(output: Path) -> None:
    """Archive source snapshot, pinned package list, and experimental protocol manifest."""
    source = output / "source"
    source.mkdir(exist_ok=True)
    rel_names = [
        "src/calibration.py",
        "src/innovation.py",
        "src/agents.py",
        "src/model.py",
        "experiments/run_morris.py",
        "experiments/run_convergence.py",
        "tests/test_invariants.py",
    ]
    hashes = {}
    for rel in rel_names:
        src_path = ROOT / rel
        if src_path.exists():
            dst = source / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            hashes[rel] = hashlib.sha256(dst.read_bytes()).hexdigest()

    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "mesa", "joblib", "pytest"):
        try:
            packages[name] = pkg_version(name)
        except Exception:
            packages[name] = "not-installed"
    (output / "requirements.txt").write_text(
        "\n".join(f"{k}=={v}" for k, v in packages.items()) + "\n", encoding="utf-8"
    )
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_head = "unknown"

    manifest = dict(
        status="prepared",
        started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,
        executable=sys.executable,
        dependencies=packages,
        git_head=git_head,
        source_sha256=hashes,
        model_defaults={k: v.default for k, v in inspect.signature(RealCompetitionModel).parameters.items()},
        factors=[dict(name=n, low=lo, high=hi, baseline=b) for n, lo, hi, b in FACTORS],
        protocol=dict(
            method="random Morris OAT",
            levels=6,
            delta=0.6,
            trajectories=20,
            seeds=list(range(12)),
            regimes=["neutral", "marx"],
            n_steps=120,
            burn_in=40,
            followup_seeds=list(range(1000, 1064)),
        ),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def interval(values) -> tuple[float, float, float]:
    """Mean and two-sided 95% Student-t confidence interval."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    half = float(student_t.ppf(0.975, n - 1) * arr.std(ddof=1) / np.sqrt(n))
    return mean, mean - half, mean + half


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Format DataFrame as GitHub-flavored Markdown table."""
    def fmt(value):
        return f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(fmt(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def summarize_followup(output: Path):
    """Compute Stage 2 summary tables and paired within-seed contrasts."""
    design = pd.read_csv(output / "followup_design.csv")
    raw = pd.read_csv(output / "followup_raw.csv", keep_default_na=False)
    rows, contrasts = [], []
    for (cid, regime), group in raw.groupby(["config_id", "regime"]):
        for metric in METRICS:
            mean, low, high = interval(group[metric])
            rows.append(dict(config_id=cid, regime=regime, metric=metric, mean=mean, ci_low=low, ci_high=high, n=len(group)))
    summary = pd.DataFrame(rows).merge(design, on="config_id")
    summary.to_csv(output / "followup_summary.csv", index=False)

    for cid, group in raw.groupby("config_id"):
        a = group[group.regime == "marx"].set_index("seed")
        b = group[group.regime == "neutral"].set_index("seed")
        for metric in METRICS:
            mean, low, high = interval(a[metric] - b[metric])
            contrasts.append(dict(config_id=cid, comparison="marx_minus_neutral", metric=metric, mean=mean, ci_low=low, ci_high=high, n=len(a)))
    baseline_cid = int(design.loc[design.scenario == "baseline", "config_id"].iloc[0])
    base = raw[raw.config_id == baseline_cid].set_index(["regime", "seed"])
    for (cid, regime), group in raw.groupby(["config_id", "regime"]):
        a = group.set_index("seed")
        b = base.loc[regime]
        for metric in METRICS:
            mean, low, high = interval(a[metric] - b[metric])
            contrasts.append(dict(config_id=cid, comparison=f"{regime}_minus_baseline", metric=metric, mean=mean, ci_low=low, ci_high=high, n=len(a)))
    contrasts = pd.DataFrame(contrasts).merge(design, on="config_id")
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    return design, raw, summary, contrasts


def make_figures(output: Path, indices: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Generate English publication sensitivity figures."""
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    for regime in ("marx", "neutral"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 8), layout="constrained")
        for ax, metric in zip(axes, ("share_final", "profit_post", "price_final")):
            sub = indices[(indices.regime == regime) & (indices.metric == metric)].nlargest(12, "mu_star").sort_values("mu_star")
            y = np.arange(len(sub))
            ax.barh(y, sub.mu_star, color=COLORS[regime], alpha=0.75)
            ax.hlines(y, sub.ci_low, sub.ci_high, color="#222222", lw=1.4)
            ax.set_yticks(y, [FIG_LABELS[p] for p in sub.parameter])
            ax.set_title(FIG_OUTPUT_LABELS[metric])
            ax.set_xlabel("μ* (with 95% Bootstrap CI)")
            ax.grid(axis="x", alpha=0.2)
        fig.suptitle(
            f"Global Morris Screening — {REGIME_LABELS[regime]} Regime\n"
            "Elementary Effects per Normalized Input Unit (20 Trajectories × 12 Seeds)",
            fontsize=14,
        )
        fig.savefig(figures / f"morris_{regime}.png", dpi=180)
        plt.close(fig)

    top = summary[summary.scenario == "curve"].parameter.unique()
    for metrics, filename in (
        (["share_final", "profit_post"], "response_curves"),
        (["price_final", "kappa_final"], "response_cost_capital"),
    ):
        fig, axes = plt.subplots(len(top), 2, figsize=(13, 3.1 * len(top)), squeeze=False, layout="constrained")
        for i, parameter in enumerate(top):
            base = next(f[3] for f in FACTORS if f[0] == parameter)
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                for regime in ("neutral", "marx"):
                    sub = summary[(summary.parameter == parameter) & (summary.metric == metric) & (summary.regime == regime)].sort_values("value")
                    ax.plot(sub.value, sub["mean"], "o-", color=COLORS[regime], label=REGIME_LABELS[regime])
                    ax.fill_between(sub.value, sub.ci_low, sub.ci_high, color=COLORS[regime], alpha=0.15)
                ax.axvline(base, color="#555555", ls=":", label="Baseline")
                if metric == "share_final":
                    ax.axhline(50, color="#888888", ls="--", lw=1)
                ax.set_xlabel(FIG_LABELS[parameter])
                ax.set_ylabel(FIG_OUTPUT_LABELS[metric])
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        fig.suptitle("Local Confirmation Sweeps — 64 Out-of-Sample Seeds (95% Pointwise CIs)", fontsize=14)
        fig.savefig(figures / f"{filename}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    scenarios = ["baseline", "reversed_types", "horizon_300", "initial_normal_return_10pct"]
    short = ["Baseline", "Reversed Cohorts", "Horizon T = 300", "Initial Price 1.072"]
    for ax, metric in zip(axes, ("share_final", "profit_post", "invalid_prime")):
        for offset, regime in ((-0.15, "neutral"), (0.15, "marx")):
            sub = summary[(summary.parameter == "scenario") & (summary.metric == metric) & (summary.regime == regime)].set_index("scenario").loc[scenarios]
            y = sub["mean"].to_numpy()
            ax.bar(np.arange(4) + offset, y, width=0.28, color=COLORS[regime], label=REGIME_LABELS[regime])
            if metric != "invalid_prime":
                ax.errorbar(
                    np.arange(4) + offset,
                    y,
                    yerr=[y - sub.ci_low.to_numpy(), sub.ci_high.to_numpy() - y],
                    fmt="none",
                    color="#222",
                    capsize=3,
                )
        ax.set_xticks(range(4), short, rotation=25, ha="right")
        ax.set_title(FIG_OUTPUT_LABELS[metric])
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Structural Robustness Scenarios — 64 Out-of-Sample Seeds (Post-Burn-In 41–120; 41–300 for T = 300)")
    fig.savefig(figures / "structural_scenarios.png", dpi=180)
    plt.close(fig)


def report(output: Path) -> int:
    """Assemble full Markdown report, robustness CSV, figures, and finalized manifest."""
    indices = pd.read_csv(output / "morris_indices.csv")
    morris = pd.read_csv(output / "morris_raw.csv", keep_default_na=False)
    design_m = pd.read_csv(output / "morris_design.csv")
    design, follow, summary, contrasts = summarize_followup(output)
    stability = pd.read_csv(output / "rank_stability.csv")
    make_figures(output, indices, summary)

    means = morris.groupby(["config_id", "regime"])[METRICS].mean()
    a = means.xs("marx", level="regime")
    b = means.xs("neutral", level="regime")
    checks = {
        "Part finale Shaikh > 50 % en Marx": (a.share_final > 50),
        "Part finale Shaikh Marx > neutre": (a.share_final > b.share_final),
        "Profit Marx < neutre": (a.profit_post < b.profit_post),
        "Prix final Marx < neutre": (a.price_final < b.price_final),
        "Intensité finale Marx > neutre": (a.kappa_final > b.kappa_final),
    }
    valid = (a.invalid_prime == 0) & (b.invalid_prime == 0)
    robustness = [
        dict(
            Propriété=label,
            Points=int(check.sum()),
            Total=len(check),
            Fraction_pct=100.0 * check.mean(),
            Fraction_sous_ensemble_sans_invalidité_pct=100.0 * check[valid].mean(),
        )
        for label, check in checks.items()
    ]
    pd.DataFrame(robustness).to_csv(output / "robustness_summary.csv", index=False)

    lines = [
        "# Analyse de sensibilité du modèle ABM Shaikh (horizon annuel 120 ans)",
        "",
        f"Criblage : {len(morris):,} simulations ({len(design_m)} configurations × {morris['seed'].nunique()} graines × 2 régimes).",
        f"Confirmation : {len(follow):,} simulations ({follow['seed'].nunique()} graines hors échantillon).",
        "",
        "## Robustesse globale entre régimes",
        "",
        md_table(pd.DataFrame(robustness)),
        "",
        "## Indices de Morris (régime Marx)",
        "",
        md_table(indices[(indices.regime == "marx") & (indices.metric == "share_final")].nlargest(8, "mu_star")[["parameter", "mu", "mu_star", "sigma", "ci_low", "ci_high"]]),
        "",
        "## Stabilité du classement",
        "",
        md_table(stability[stability.metric.isin(["share_final", "profit_post", "price_final"])], digits=3),
        "",
    ]
    (output / "RAPPORT_SENSIBILITE.md").write_text("\n".join(lines), encoding="utf-8")

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            status="complete",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            simulations=len(morris) + len(follow),
            output_sha256={
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
            figure_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output / "figures").glob("*.png")},
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(morris) + len(follow)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run global Morris screening and local confirmation.")
    parser.add_argument("phase", choices=("prepare", "screen", "followup", "analyze", "report", "all"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--trajectories", type=int, default=20)
    args = parser.parse_args()

    output = args.output
    if args.phase in ("prepare", "all"):
        if output is None:
            output = ROOT / "sensitivity_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output.mkdir(parents=True, exist_ok=True)
        archive(output)
        morris_design(trajectories=args.trajectories).to_csv(output / "morris_design.csv", index=False)
        pd.DataFrame([dict(name=n, low=lo, high=hi, baseline=b) for n, lo, hi, b in FACTORS]).to_csv(
            output / "parameter_ranges.csv", index=False
        )
        print(output)
        if args.phase == "prepare":
            return

    if output is None:
        parser.error("--output is required for phases other than 'prepare' or 'all'.")

    if args.phase in ("screen", "all"):
        run_batch(pd.read_csv(output / "morris_design.csv"), list(range(12)), output, "morris", jobs=args.jobs)
        analyze(output)
    if args.phase in ("followup", "all"):
        summary = pd.read_csv(output / "morris_indices.csv")
        design = followup_design(summary)
        design.to_csv(output / "followup_design.csv", index=False)
        run_batch(design, list(range(1000, 1064)), output, "followup", jobs=args.jobs)
    if args.phase == "analyze":
        analyze(output)
    if args.phase in ("report", "all"):
        n_sims = report(output)
        print(f"Report complete: {n_sims} simulations in {output}")


if __name__ == "__main__":
    main()
