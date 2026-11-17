"""
Étape A de validation académique : Analyse de convergence de Monte Carlo (N = 1000 réplications).
Scénario de référence (baseline) du modèle ABM de concurrence réelle d'Anwar Shaikh.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed

from abm_shaikh import RealCompetitionModel, FirmAgent, Vintage


# Configuration des variables macroéconomiques agrégées suivies
VARIABLES_META = {
    "share_shaikh": {
        "label": "Part de marché Shaikh $\\bar{s}_{\\mathrm{Shaikh}}$",
        "short": "Part Shaikh",
        "unit": "%",
        "scale": 100.0,
        "color": "#C8102E",
    },
    "r_branch": {
        "label": "Taux de profit de branche $\\bar{r}^{\\mathrm{branch}}$",
        "short": "Taux de profit $r^{\\mathrm{branch}}$",
        "unit": "%",
        "scale": 100.0,
        "color": "#1F5A9E",
    },
    "u_industry": {
        "label": "Taux d'utilisation $\\bar{u}_{\\mathrm{branche}}$",
        "short": "Utilisation $u_{\\mathrm{branche}}$",
        "unit": "%",
        "scale": 100.0,
        "color": "#2E8B57",
    },
    "kappa_industry": {
        "label": "Intensité capitalistique $\\bar{\\kappa}_{\\mathrm{branche}}$",
        "short": "Intensité $\\bar{\\kappa}_{\\mathrm{branche}}$",
        "unit": "niveau",
        "scale": 1.0,
        "color": "#7B3294",
    },
    "g_p_leader": {
        "label": "Taux de baisse du prix leader $g_{p^{\\mathrm{leader}}}$",
        "short": "Déflation $g_{p^{\\mathrm{leader}}}$",
        "unit": "%/pér.",
        "scale": 100.0,
        "color": "#D95F02",
    },
    "n_firms": {
        "label": "Nombre de firmes actives $\\overline{|\\mathcal{N}_t|}$",
        "short": "Firmes actives $|\\mathcal{N}_t|$",
        "unit": "firmes",
        "scale": 1.0,
        "color": "#555555",
    },
}


def run_single_replication(
    seed: int,
    n_steps: int = 150,
    burn_in: int = 50,
    innovation_regime: str = "marx",
    n_firms_init: int = 10,
    D: float = 85.0,
    r_star: float = 0.12,
) -> dict:
    """
    Exécute une réplication indépendante de l'ABM avec une graine déterministe `seed`
    et calcule les moyennes stationnaires sur la fenêtre post-burn-in [burn_in + 1, n_steps].
    """
    if burn_in >= n_steps:
        raise ValueError(f"burn_in ({burn_in}) doit être strictement inférieur à n_steps ({n_steps}).")

    model = RealCompetitionModel(
        D=D,
        u_n=0.85,
        T=10,
        gamma=0.03,
        delta_cost=0.08,
        eps_0=0.005,
        mu_z=0.01,
        sigma_z=0.005,
        lam=0.8,
        R=0.05,
        innovation_regime=innovation_regime,
        marx_a=0.08,
        marx_theta=-np.pi / 4,
        gamma_thresh=0.75,
        r_star=r_star,
        psi=5.0,
        f_0=0.02,
        K_0=10.0,
        B_0=15.0,
        s_min=0.01,
        tau_grace=20,
        seed=seed,
    )

    uc_0, kappa_0 = 0.50, 1.0
    p_0 = uc_0 + r_star * kappa_0
    k_init_firm = (D / 0.85) / n_firms_init

    for idx in range(n_firms_init):
        rule = "shaikh" if idx % 2 == 0 else "okishio"
        birth_0 = -(idx % 10)
        v0 = Vintage(uc=uc_0, kappa=kappa_0, k_init=k_init_firm, k=k_init_firm, birth_period=birth_0)
        FirmAgent(
            model=model,
            rule=rule,
            initial_vintages=[v0],
            uc_b=uc_0,
            kappa_b=kappa_0,
            price=p_0,
            target_share=1.0 / n_firms_init,
            budget=15.0,
            u_init=0.85,
            entry_period=0,
        )

    share_shaikh_series = []
    r_branch_series = []
    u_industry_series = []
    kappa_industry_series = []
    n_firms_series = []
    p_leader_burn_in = p_0
    p_leader_end = p_0

    for t in range(1, n_steps + 1):
        model.current_period += 1
        firms_t = list(model.agents)
        if not firms_t:
            break

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
    """
    Calcule pour chaque variable et chaque taille d'échantillon cumulée n in [min_n, N] :
    - Moyenne cumulée mu_n et bornes IC 95% (exact Student-t)
    - Écart-type cumulé sigma_n
    - Erreur standard de la moyenne SEM_n = sigma_n / sqrt(n) et RSEM_n = SEM_n / |mu_n|
    - Coefficient de variation CV_n = sigma_n / |mu_n| et écart relatif à CV_N
    """
    N = len(df_raw)
    n_grid = np.arange(min_n, N + 1)
    t_crit = stats.t.ppf(0.975, df=n_grid - 1)

    records = []
    for var_name in VARIABLES_META.keys():
        x = df_raw[var_name].to_numpy(dtype=float)
        idx_all = np.arange(1, N + 1, dtype=float)

        # Sommes cumulées pour calcul O(N) exact
        csum = np.cumsum(x)
        csum_sq = np.cumsum(x ** 2)

        mu_all = csum / idx_all
        # Variance d'échantillon corrigée : (sum(x^2) - n * mu^2) / (n - 1)
        var_all = np.zeros(N, dtype=float)
        var_all[1:] = np.maximum(0.0, (csum_sq[1:] - idx_all[1:] * (mu_all[1:] ** 2)) / (idx_all[1:] - 1.0))
        sigma_all = np.sqrt(var_all)

        mu_n = mu_all[min_n - 1:]
        sigma_n = sigma_all[min_n - 1:]
        sem_n = sigma_n / np.sqrt(n_grid)
        ci_half = t_crit * sem_n
        rsem_n = np.where(np.abs(mu_n) > 1e-12, sem_n / np.abs(mu_n), 0.0)
        cv_n = np.where(np.abs(mu_n) > 1e-12, sigma_n / np.abs(mu_n), 0.0)
        cv_final = cv_n[-1] if cv_n[-1] > 1e-12 else 1.0
        cv_rel_err = np.abs(cv_n - cv_final) / cv_final

        for i, n_val in enumerate(n_grid):
            records.append({
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
            })

    return pd.DataFrame(records)


def determine_optimal_n_star(
    df_conv: pd.DataFrame,
    rsem_threshold_pct: float = 1.75,
    cv_stab_threshold_pct: float = 2.0,
) -> tuple[int, dict]:
    """
    Détermine le seuil optimal N* de réplications Monte Carlo :
    Plus petit n >= 30 tel que pour tout k >= n, l'erreur standard relative RSEM_k <= rsem_threshold_pct
    ET l'écart relatif du CV par rapport à sa valeur finale CV_1000 est <= cv_stab_threshold_pct.
    """
    n_star_by_var = {}
    for var_name, grp in df_conv.groupby("variable"):
        grp = grp.sort_values("n")
        cond = (grp["rsem_pct"] <= rsem_threshold_pct) & (grp["cv_rel_err_pct"] <= cv_stab_threshold_pct) & (grp["n"] >= 30)
        # On cherche le premier n à partir duquel la condition reste vraie jusqu'à N=1000
        vals = cond.to_numpy()
        n_arr = grp["n"].to_numpy()
        # Suffix all-true check
        suffix_ok = np.logical_and.accumulate(vals[::-1])[::-1]
        valid_indices = np.where(suffix_ok)[0]
        if len(valid_indices) > 0:
            n_star_by_var[var_name] = int(n_arr[valid_indices[0]])
        else:
            n_star_by_var[var_name] = int(n_arr[-1])

    # N* global = max des N* des 4 variables principales (hors variables à variance très forte)
    core_vars = ["share_shaikh", "r_branch", "u_industry", "kappa_industry"]
    n_star_global = max(n_star_by_var[v] for v in core_vars)
    return n_star_global, n_star_by_var


def plot_mc_convergence(
    df_conv: pd.DataFrame,
    n_star_global: int,
    n_star_by_var: dict,
    burn_in: int,
    n_steps: int,
    output_path: str = "figures/fig_mc_convergence.png",
):
    """
    Génère la figure académique haute résolution (300 DPI) en deux volets :
    - Volet (a) [2x2 en haut] : Stabilisation des moyennes cumulées et IC à 95% pour les 4 variables clés.
    - Volet (b) [1x2 en bas]  : Décroissance de l'erreur standard relative (RSEM) en 1/sqrt(n)
                                et stabilisation du coefficient de variation (CV).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.8,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
    })

    fig = plt.figure(figsize=(15.0, 13.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.12], hspace=0.34, wspace=0.22)

    core_vars = ["share_shaikh", "r_branch", "u_industry", "kappa_industry"]
    panel_letters = ["(a.1)", "(a.2)", "(a.3)", "(a.4)"]

    # --- VOLET (a) : Moyennes cumulées et bandes de confiance à 95% ---
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

        ax.fill_between(n_vals, low_vals, high_vals, color=c, alpha=0.20, label="IC à 95% ($\\hat{\\mu}_n \\pm t_{0.975}\\,\\mathrm{SEM}_n$)")
        ax.plot(n_vals, mean_vals, color=c, linewidth=2.0, label=f"Moyenne cumulée $\\hat{{\\mu}}_n$ ($\\hat{{\\mu}}_{{1000}} = {final_mean:.2f}$)")
        ax.axhline(final_mean, color="#222222", linestyle=":", linewidth=1.2, label="Valeur asymptotique ($N=1000$)")
        ax.axvline(n_star_v, color="#333333", linestyle="--", linewidth=1.4,
                   label=f"Seuil de stabilisation $N^*_{{{var_name.split('_')[0]}}} = {n_star_v}$")

        ax.set_title(f"{panel_letters[idx]} {meta['label']}", fontweight="bold", pad=7)
        ax.set_xlabel("Nombre de réplications cumulées $n$")
        ax.set_ylabel(f"Moyenne ({meta['unit']})")
        ax.set_xlim(10, int(n_vals[-1]))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.92)

    # --- VOLET (b.1) : Décroissance de l'erreur standard relative RSEM(n) = SEM(n) / |mu(n)| ---
    ax_sem = fig.add_subplot(gs[2, 0])
    for var_name in core_vars + ["g_p_leader", "n_firms"]:
        meta = VARIABLES_META[var_name]
        sub = df_conv[df_conv["variable"] == var_name].sort_values("n")
        ax_sem.plot(
            sub["n"], sub["rsem_pct"],
            color=meta["color"], linewidth=1.8, label=meta["short"]
        )

    ax_sem.axhline(1.75, color="#B22222", linestyle=":", linewidth=1.5, label="Seuil de précision RSEM $= 1.75\\%$")
    ax_sem.axvline(n_star_global, color="#111111", linestyle="--", linewidth=1.8,
                   label=f"Seuil global recommandé $N^* = {n_star_global}$")
    ax_sem.set_yscale("log")
    ax_sem.set_xscale("log")
    ax_sem.set_title("(b.1) Décroissance de l'erreur standard relative $\\mathrm{RSEM}_n = \\mathrm{SEM}_n / |\\hat{\\mu}_n|$ (échelle log-log)",
                     fontweight="bold", pad=7)
    ax_sem.set_xlabel("Nombre de réplications cumulées $n$ (log)")
    ax_sem.set_ylabel("Erreur standard relative RSEM (%) [log]")
    ax_sem.set_xlim(10, 1000)
    ax_sem.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_sem.legend(loc="upper right", ncol=2, framealpha=0.92)

    # --- VOLET (b.2) : Stabilisation du Coefficient de Variation CV(n) = sigma(n) / |mu(n)| ---
    ax_cv = fig.add_subplot(gs[2, 1])
    for var_name in core_vars + ["g_p_leader", "n_firms"]:
        meta = VARIABLES_META[var_name]
        sub = df_conv[df_conv["variable"] == var_name].sort_values("n")
        # Normalisation CV_n / CV_1000 pour visualiser la stabilisation relative de toutes les variables sur la même échelle
        cv_vals = sub["cv"].to_numpy()
        cv_norm = (cv_vals / cv_vals[-1]) if cv_vals[-1] > 0 else np.ones_like(cv_vals)
        ax_cv.plot(
            sub["n"], cv_norm,
            color=meta["color"], linewidth=1.8,
            label=f"{meta['short']} ($\\mathrm{{CV}}_{{1000}}={cv_vals[-1]:.3f}$)"
        )

    ax_cv.axhspan(0.98, 1.02, color="#2E8B57", alpha=0.14, label="Tunnel de stabilisation $\\pm 2\\%$ autour de $\\mathrm{CV}_{1000}$")
    ax_cv.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2)
    ax_cv.axvline(n_star_global, color="#111111", linestyle="--", linewidth=1.8,
                  label=f"Seuil global recommandé $N^* = {n_star_global}$")
    ax_cv.set_title("(b.2) Stabilisation relative du Coefficient de Variation $\\mathrm{CV}_n / \\mathrm{CV}_{1000}$",
                    fontweight="bold", pad=7)
    ax_cv.set_xlabel("Nombre de réplications cumulées $n$")
    ax_cv.set_ylabel("Ratio $\\mathrm{CV}_n / \\mathrm{CV}_{1000}$")
    ax_cv.set_xlim(10, 1000)
    ax_cv.set_ylim(0.82, 1.18)
    ax_cv.grid(True, linestyle="--", alpha=0.3)
    ax_cv.legend(loc="upper right", ncol=2, framealpha=0.92)

    fig.suptitle(
        f"Analyse de convergence de Monte Carlo ($N = 1000$ réplications, horizon $T = {n_steps}$, burn-in $T_0 = {burn_in}$, régime Marx)",
        fontsize=13, fontweight="bold", y=0.94
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_monte_carlo_validation(
    n_replications: int = 1000,
    n_steps: int = 150,
    burn_in: int = 50,
    innovation_regime: str = "marx",
    n_jobs: int = -1,
    results_dir: str = "results",
    figures_dir: str = "figures",
):
    """Lance l'analyse complète de convergence de Monte Carlo (N = 1000) en parallèle."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"=== LANCEMENT DE L'ANALYSE DE CONVERGENCE MONTE CARLO (N = {n_replications} runs) ===")
    print(f"Paramètres : horizon T = {n_steps}, burn-in T_0 = {burn_in}, régime = '{innovation_regime}', n_jobs = {n_jobs}")
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
    print(f"Exécution parallèle terminée en {elapsed:.2f} s ({elapsed / n_replications * 1000:.2f} ms / run).")

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

    print("\n=== TABLEAU RÉCAPITULATIF DE CONVERGENCE (N = 1000, post-burn-in t in [51, 150]) ===")
    summary_rows = []
    for var_name, meta in VARIABLES_META.items():
        sub_final = df_conv[(df_conv["variable"] == var_name) & (df_conv["n"] == n_replications)].iloc[0]
        scale = meta["scale"]
        summary_rows.append({
            "Variable": var_name,
            "Moyenne (N=1000)": f"{sub_final['cum_mean'] * scale:.4f} {meta['unit']}",
            "IC 95%": f"[{sub_final['ci95_low'] * scale:.4f}, {sub_final['ci95_high'] * scale:.4f}]",
            "SEM": f"{sub_final['sem'] * scale:.4f}",
            "RSEM (%)": f"{sub_final['rsem_pct']:.2f}%",
            "CV": f"{sub_final['cv']:.4f}",
            "N* (RSEM<=1.75%, dCV<=2%)": n_star_by_var[var_name],
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    print(f"\n-> Seuil optimal recommandé pour le bloc d'indicateurs clés : N* = {n_star_global} réplications.")
    print(f"-> Fichier brut sauvegardé : {raw_csv_path}")
    print(f"-> Métriques de convergence : {conv_csv_path}")
    print(f"-> Figure haute résolution (300 DPI) : {fig_path}")


if __name__ == "__main__":
    run_monte_carlo_validation(
        n_replications=1000,
        n_steps=150,
        burn_in=50,
        innovation_regime="marx",
        n_jobs=-1,
    )
