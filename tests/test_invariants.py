"""
Comprehensive invariant, unit, statistical neutrality, and experimental-design tests
for the refactored Real Competition ABM (src/ and experiments/).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from experiments.run_convergence import run_single_replication
from experiments.run_morris import (
    BASE,
    METRICS,
    NAMES,
    decode,
    elementary_effects,
    morris_design,
    simulate as morris_simulate,
)
from src.agents import FirmAgent, Vintage
from src.calibration import (
    BASELINE_PARAMS,
    BASELINE_RULES,
    INITIAL_FIRM_PARAMS,
    baseline_parameters,
    build_baseline_model,
    init_paired_homogeneous_firms,
)
from src.innovation import apply_dlf_transformation, evaluate_adoption_criterion
from src.model import RealCompetitionModel, run_detailed_simulation


def test_step_0_aggregates_and_adaptive_demand():
    """Verify firm capacity-weighted aggregates and adaptive price-memory demand (Step 0)."""
    test_model = RealCompetitionModel(u_n=0.82, seed=42)
    v1 = Vintage(uc=0.60, kappa=1.0, k_init=60.0, k=60.0, birth_period=0)
    v2 = Vintage(uc=0.40, kappa=1.5, k_init=40.0, k=40.0, birth_period=1)
    firm = FirmAgent(
        model=test_model,
        rule="shaikh",
        initial_vintages=[v1, v2],
        uc_b=0.40,
        kappa_b=1.5,
        price=0.80,
        target_share=1.0,
        budget=50.0,
    )
    assert np.isclose(firm.total_capacity, 100.0)
    assert np.isclose(firm.mean_uc, 0.52)
    assert np.isclose(firm.mean_kappa, 1.20)
    assert np.isclose(firm.normal_profit_rate(), 0.82 * (0.28 / 1.20))

    m0 = RealCompetitionModel(D=82.0, g_D=0.015, eta=0.5, alpha_p=0.0, p_0=1.0, seed=42)
    m0.current_period = 1
    m0.step_0_demand()
    assert np.isclose(m0.D, 82.0 * 1.015)
    m0.p_min_prev = 0.80
    m0.current_period = 2
    m0.step_0_demand()
    assert np.isclose(m0.D, 82.0 * (1.015 ** 2) * ((1.0 / 0.80) ** 0.5))

    m_adapt = RealCompetitionModel(D=82.0, g_D=0.015, eta=2.0, alpha_p=0.15, p_0=1.0, seed=42)
    f_ref = FirmAgent(
        m_adapt, "shaikh", [Vintage(0.50, 1.0, 50.0, 50.0, 0)], 0.50, 1.0, price=0.80, target_share=1.0, budget=20.0
    )
    f_ref.sales = 40.0
    m_adapt.current_period = 1
    m_adapt.step_0_demand()
    assert np.isclose(m_adapt.D, 82.0 * 1.015)
    m_adapt.step_3_production_and_accounting()
    assert np.isclose(m_adapt.p_min_prev, 0.80)
    assert np.isclose(m_adapt.p_tilde, 0.85 * 1.0 + 0.15 * 0.80)
    m_adapt.current_period = 2
    m_adapt.step_0_demand()
    assert np.isclose(m_adapt.D, 82.0 * (1.015 ** 2) * ((0.97 / 0.80) ** 2.0))


def test_step_1_pricing_and_cost_floor():
    """Verify undercutting, shortfall markup, cost floor p_i >= uc_bar_i, and lone-firm behavior."""
    m_floor = RealCompetitionModel(D=85.0, u_n=0.85, gamma=0.50, delta_cost=0.08, eps_0=0.01, seed=42)
    fA = FirmAgent(m_floor, "shaikh", [Vintage(0.40, 1.0, 60.0, 60.0, 0)], 0.40, 1.0, price=0.65, target_share=0.6, budget=20.0, u_init=0.85)
    fB = FirmAgent(m_floor, "okishio", [Vintage(0.68, 1.0, 40.0, 40.0, 0)], 0.68, 1.0, price=0.72, target_share=0.4, budget=20.0, u_init=0.85)
    m_floor.step_1_pricing()
    assert np.isclose(fA.price, 0.6235)
    assert np.isclose(fB.price, 0.68)
    assert fA.price >= fA.mean_uc and fB.price >= fB.mean_uc

    m_rat = RealCompetitionModel(D=80.0, u_n=0.82, gamma=0.50, delta_cost=0.10, eps_0=0.0, seed=42)
    f_lowcap = FirmAgent(m_rat, "shaikh", [Vintage(0.50, 1.0, 20.0, 20.0, 0)], 0.50, 1.0, price=1.0, target_share=0.5, budget=20.0)
    f_spare = FirmAgent(m_rat, "okishio", [Vintage(0.50, 1.0, 80.0, 80.0, 0)], 0.50, 1.0, price=1.0, target_share=0.5, budget=20.0)
    m_rat.step_1_pricing()
    assert np.isclose(f_lowcap.price, 1.295)
    assert np.isclose(f_spare.price, 0.95)

    # Core baseline disables special monopoly markup, keeping only the standard shortage rule.
    m_single = RealCompetitionModel(**baseline_parameters(D=50, gamma=0, delta_cost=0, eps_0=0, mu_z=0.02, sigma_z=0))
    f_s = FirmAgent(m_single, "shaikh", [Vintage(0.4, 1.0, 100.0, 100.0, 0)], 0.4, 1.0, 1.0, 1.0, 10.0)
    m_single.step_1_pricing()
    assert f_s.price == 1.0
    f_s.vintages[0].k = 10.0
    m_single.step_1_pricing()
    assert np.isclose(f_s.price, 1.02)


def test_step_2_demand_allocation_and_tie_symmetry():
    """Verify quasi-replicator share conservation sum(f_i) = 1, spare spillover, and tie symmetry."""
    m2 = RealCompetitionModel(D=100.0, lam=0.8, seed=42)
    fA = FirmAgent(m2, "shaikh", [Vintage(0.30, 1.0, 55.0, 55.0, 0)], 0.30, 1.0, price=0.50, target_share=0.50, budget=20.0)
    fB = FirmAgent(m2, "okishio", [Vintage(0.40, 1.0, 80.0, 80.0, 0)], 0.40, 1.0, price=0.625, target_share=0.50, budget=20.0)
    m2.step_2_demand_allocation()
    assert np.isclose(fA.target_share + fB.target_share, 1.0)
    assert np.isclose(fA.realized_share + fB.realized_share, 1.0)
    assert np.isclose(fA.target_share, 0.60)
    assert np.isclose(fA.sales, 55.0)
    assert np.isclose(fB.sales, 45.0)
    assert np.isclose(fA.u, 1.0)

    m_tie = RealCompetitionModel(D=100.0, lam=0.8, seed=1)
    fa = FirmAgent(m_tie, "shaikh", [Vintage(0.30, 1.0, 50.0, 50.0, 0)], 0.30, 1.0, price=0.50, target_share=0.30, budget=20.0)
    fb = FirmAgent(m_tie, "okishio", [Vintage(0.30, 1.0, 50.0, 50.0, 0)], 0.30, 1.0, price=0.50, target_share=0.30, budget=20.0)
    _fc = FirmAgent(m_tie, "okishio", [Vintage(0.40, 1.0, 50.0, 50.0, 0)], 0.40, 1.0, price=0.625, target_share=0.40, budget=20.0)
    m_tie.step_2_demand_allocation()
    assert np.isclose(fa.target_share, fb.target_share)
    assert np.isclose(sum(f.target_share for f in m_tie.agents), 1.0)


def test_step_3_and_4_merit_order_and_one_hoss_shay_scrapping():
    """Verify merit-order production, depreciation cash-flow retention, and one-hoss shay scrapping."""
    m3 = RealCompetitionModel(T=10, u_n=0.85, seed=42)
    v_old = Vintage(uc=0.50, kappa=1.0, k_init=50.0, k=50.0, birth_period=0)
    v_new = Vintage(uc=0.25, kappa=1.5, k_init=50.0, k=50.0, birth_period=2)
    v_mid = Vintage(uc=0.40, kappa=1.2, k_init=30.0, k=30.0, birth_period=1)
    firm = FirmAgent(m3, "shaikh", [v_old, v_new, v_mid], uc_b=0.25, kappa_b=1.5, price=0.60, target_share=1.0, budget=100.0)
    firm.sales = 65.0
    m3.step_3_production_and_accounting()
    expected_dep = (50.0 * 1.5 + 15.0 * 1.2) / (10 * 0.85)
    expected_cap_val = 50.0 * 1.0 + 50.0 * 1.5 + 30.0 * 1.2
    assert np.isclose(firm.profit, 20.5)
    assert np.isclose(firm.budget, 120.5 + expected_dep)
    assert np.isclose(firm.r_real, 20.5 / expected_cap_val)

    m4 = RealCompetitionModel(T=10, seed=42)
    m4.current_period = 10
    v_expired = Vintage(uc=0.30, kappa=1.0, k_init=40.0, k=40.0, birth_period=1)
    v_unprof = Vintage(uc=0.55, kappa=1.0, k_init=30.0, k=30.0, birth_period=8)
    v_alive = Vintage(uc=0.35, kappa=1.2, k_init=50.0, k=50.0, birth_period=7)
    firm4 = FirmAgent(m4, "shaikh", [v_expired, v_unprof, v_alive], uc_b=0.35, kappa_b=1.2, price=0.50, target_share=1.0, budget=100.0)
    m4.step_4_scrapping()
    assert firm4.vintages == [v_alive]
    assert np.isclose(firm4.k_surv, 50.0)


def test_step_5_innovation_adoption_consolidation_and_budget_bound():
    """Verify DLF transformation (ucs > 0), Shaikh vs Okishio divergence, consolidation, and budget ceiling."""
    ucs_plus, uc_plus, kappa_plus = apply_dlf_transformation(
        uc_b=0.672, kappa_b=3.280, d_ucs=0.032, d_kappa=-0.043, T=16, u_n=0.82
    )
    assert ucs_plus > 0.0 and uc_plus < 0.672 and kappa_plus > 3.280
    assert evaluate_adoption_criterion(
        "shaikh", 0.672, 3.280, uc_plus, kappa_plus, u_n=0.82, r_star=0.04, current_price=0.80, p_ref_okishio="normal_price"
    )
    assert not evaluate_adoption_criterion(
        "okishio", 0.672, 3.280, uc_plus, kappa_plus, u_n=0.82, r_star=0.04, current_price=0.80, p_ref_okishio="normal_price"
    )

    m5 = RealCompetitionModel(u_n=0.82, T=16, gamma_thresh=0.25, beta_r=2.0, beta_u=0.20, r_star=0.04, p_ref_okishio="normal_price", seed=42)
    m5.current_period = 1
    f_sh = FirmAgent(m5, "shaikh", [Vintage(0.672, 3.280, 10.0, 10.0, 0)], 0.672, 3.280, price=0.80, target_share=0.5, budget=100.0, u_init=0.92)
    f_ok = FirmAgent(m5, "okishio", [Vintage(0.672, 3.280, 10.0, 10.0, 0)], 0.672, 3.280, price=0.80, target_share=0.5, budget=100.0, u_init=0.92)
    f_sh.innovate_and_invest(current_t=1, forced_draw=(0.032, -0.043))
    f_ok.innovate_and_invest(current_t=1, forced_draw=(0.032, -0.043))
    assert f_ok.uc_a is None and np.isclose(f_ok.mean_uc, 0.672)
    assert f_sh.mean_uc < 0.672
    assert f_sh.budget >= 0.0 and f_ok.budget >= 0.0

    # Budget-constrained investment and consolidation threshold gamma_thresh = 0.75
    m_dl = RealCompetitionModel(g_D=0.015, u_n=0.82, T=16, beta_r=2.5, beta_u=0.20, r_star=0.04, gamma_thresh=0.75, seed=42)
    m_dl.current_period = 1
    f_constrained = FirmAgent(m_dl, "shaikh", [Vintage(0.50, 1.5, 100.0, 100.0, 0)], 0.50, 1.5, price=0.70, target_share=1.0, budget=60.0, u_init=0.92)
    f_constrained.uc_a, f_constrained.kappa_a = 0.40, 1.5
    f_constrained.k_pre, f_constrained.k_surv = 100.0, 80.0
    f_constrained.r_real_pre, f_constrained.u_pre = 0.92 * ((0.70 - 0.50) / 1.5), 0.92
    f_constrained.innovate_and_invest(current_t=1)
    assert np.isclose(f_constrained.total_capacity, 140.0)
    assert np.isclose(f_constrained.budget, 0.0)
    assert f_constrained.uc_a == 0.40


def test_step_6_entry_exit_and_baseline_guarantees():
    """Verify fixed 50/50 entry draw, marginalization/exhaustion exits, and passive interest_rate."""
    for flag in BASELINE_RULES:
        with pytest.raises(ValueError, match="Extensions are disabled"):
            baseline_parameters(**{flag: True})

    m = RealCompetitionModel(**baseline_parameters(r_star=0.01, tau_grace=100))
    for rule, share in (("shaikh", 0.9), ("okishio", 0.1)):
        f = FirmAgent(m, rule, [Vintage(0.3, 1.0, 10.0, 10.0, 0)], 0.3, 1.0, 0.6, share, 10.0)
        f.profit, f.realized_share = 2.0, share
    existing = {f.unique_id for f in m.agents}

    class EntryDraws:
        def __init__(self):
            self.draws = iter([0.4, 0.6])
        def poisson(self, intensity):
            assert intensity > 0
            return 2
        def random(self):
            return next(self.draws)

    m.rng = EntryDraws()
    m.step_6_entry_and_exit(capital_value_t=20.0)
    entrants = [f for f in m.agents if f.unique_id not in existing]
    assert [f.rule for f in entrants] == ["shaikh", "okishio"]
    assert np.isclose(sum(f.target_share for f in m.agents), 1.0)

    # Marginalization and physical+financial exhaustion exits
    m_ex = RealCompetitionModel(**baseline_parameters(r_star=100.0, s_min=0.03, tau_grace=30))
    m_ex.current_period = 40
    survivor = FirmAgent(m_ex, "shaikh", [Vintage(0.3, 1.0, 10.0, 10.0, 0)], 0.3, 1.0, 0.6, 0.8, 10.0)
    survivor.realized_share = 0.999
    marg = FirmAgent(m_ex, "okishio", [Vintage(0.4, 1.0, 10.0, 10.0, 0)], 0.4, 1.0, 0.7, 0.1, 10.0)
    marg.realized_share = 0.001
    exh = FirmAgent(m_ex, "okishio", [], 0.4, 1.0, 0.7, 0.1, 0.0, entry_period=40)
    m_ex.step_6_entry_and_exit(capital_value_t=20.0)
    assert list(m_ex.agents) == [survivor]
    assert marg.unique_id in m_ex.last_exits_marginalized
    assert exh.unique_id in m_ex.last_exits_exhausted

    # Passive interest_rate invariance across baseline runs
    models = []
    for rate in (0.0, 100.0):
        mm = build_baseline_model("marx", seed=19, overrides={"interest_rate": rate})
        init_paired_homogeneous_firms(mm, seed=19)
        models.append(mm)
    for _ in range(35):
        for mm in models:
            mm.step()
        assert [f.price for f in models[0].agents] == [f.price for f in models[1].agents]
        assert [f.budget for f in models[0].agents] == [f.budget for f in models[1].agents]


@pytest.mark.parametrize("regime", ["neutral", "marx"])
def test_dynamic_invariants_full_horizon(regime):
    """Enforce share conservation, price floor, non-negative budgets, and ucs_v > 0 over T=120."""
    for seed in (0, 7, 42, 99):
        model = build_baseline_model(innovation_regime=regime, seed=seed)
        init_paired_homogeneous_firms(model, seed=seed)
        for _ in range(120):
            model.current_period += 1
            model.step_0_demand()
            capital_value_t = sum(v.kappa * v.k for f in model.agents for v in f.vintages)
            model.step_1_pricing()
            model.step_2_demand_allocation()
            model.step_3_production_and_accounting()
            inv_mid = model.check_invariants(check_prime_costs=True, check_realized_shares=True)
            assert inv_mid["n_firms"] >= 2
            assert inv_mid["min_ucs"] > 0.0

            model.step_4_scrapping()
            model.step_5_innovation_and_investment()
            model.step_6_entry_and_exit(capital_value_t)
            inv_end = model.check_invariants(check_prime_costs=True, check_realized_shares=False)
            assert inv_end["n_firms"] >= 2
            assert inv_end["min_ucs"] > 0.0
            assert inv_end["min_budget"] >= -1e-9
            assert inv_end["min_price_margin"] >= -1e-12


def test_neutral_50_50_parity_and_marx_selection():
    """Verify 50/50 parity under Neutral change vs Shaikh selection & capital deepening under Marx."""
    seeds = list(range(40))
    neutral_shares, marx_shares = [], []
    neutral_kappas, marx_kappas = [], []

    for seed in seeds:
        res_n = run_single_replication(seed=seed, n_steps=120, burn_in=40, innovation_regime="neutral")
        res_m = run_single_replication(seed=seed, n_steps=120, burn_in=40, innovation_regime="marx")
        neutral_shares.append(res_n["share_shaikh_final"] * 100.0)
        marx_shares.append(res_m["share_shaikh_final"] * 100.0)
        neutral_kappas.append(res_n["kappa_industry"])
        marx_kappas.append(res_m["kappa_industry"])

    n_arr = np.asarray(neutral_shares, dtype=float)
    m_arr = np.asarray(marx_shares, dtype=float)
    n_mean = float(n_arr.mean())
    n_sem = float(n_arr.std(ddof=1) / np.sqrt(len(n_arr)))

    # Under Neutral change, 50% parity lies comfortably inside the 99% confidence band.
    assert abs(n_mean - 50.0) <= max(12.0, 2.7 * n_sem)
    assert 35.0 <= n_mean <= 65.0
    # Under Marx-biased change, Shaikh firms strongly dominate and deepen capital intensity.
    assert float(m_arr.mean()) > n_mean + 20.0
    assert float(m_arr.mean()) > 75.0
    assert float(np.mean(marx_kappas)) > float(np.mean(neutral_kappas))


def test_morris_design_and_cross_module_consistency():
    """Verify Morris OAT geometry, elementary effects, ellipse decoding, and cross-script parity."""
    design = morris_design()
    assert len(design) == 20 * (len(NAMES) + 1)
    assert design.config_id.is_unique
    for _, group in design.groupby("trajectory"):
        z = group[[f"z_{name}" for name in NAMES]].to_numpy()
        assert np.all(z >= -1e-12) and np.all(z <= 1.0 + 1e-12)
        np.testing.assert_allclose(z * 5, np.round(z * 5), atol=1e-12)
        steps = np.diff(z, axis=0)
        assert np.all((np.abs(steps) > 1e-12).sum(axis=1) == 1)
        assert np.all((np.abs(steps) > 1e-12).sum(axis=0) == 1)

    # Exact linear elementary effect recovery
    design3 = morris_design(trajectories=3)
    coeffs = np.arange(1.0, len(NAMES) + 1)
    rows = []
    for row in design3.to_dict("records"):
        y = sum(row[f"z_{name}"] * coeffs[i] for i, name in enumerate(NAMES))
        for seed in (0, 1):
            for regime in ("neutral", "marx"):
                rows.append(dict(config_id=row["config_id"], seed=seed, regime=regime, **{m: y + seed for m in METRICS}))
    ee = elementary_effects(design3, pd.DataFrame(rows))
    for i, name in enumerate(NAMES):
        np.testing.assert_allclose(ee.loc[ee.parameter == name, METRICS], coeffs[i], atol=1e-12)

    # Inactive ellipse parameters in Neutral regime produce bit-identical results
    expected_n = morris_simulate(BASE, "neutral", 42)
    changed_n = BASE | dict(ellipse_ratio=3.0, marx_theta=-0.5, marx_mu_uc=0.012, marx_mu_kappa=-0.0375)
    actual_n = morris_simulate(changed_n, "neutral", 42)
    for metric in METRICS:
        np.testing.assert_allclose(actual_n[metric], expected_n[metric], rtol=0, atol=0)

    # Exact numerical consistency between run_morris.simulate and run_convergence.run_single_replication
    for regime in ("neutral", "marx"):
        conv = run_single_replication(42, n_steps=120, burn_in=40, innovation_regime=regime, model_overrides=decode(BASE))
        morr = morris_simulate(BASE, regime, 42, n_steps=120)
        np.testing.assert_allclose(morr["share_final"], conv["share_shaikh_final"] * 100.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(morr["share_post"], conv["share_shaikh"] * 100.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(morr["profit_post"], conv["r_branch"] * 100.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(morr["util_post"], conv["u_industry"] * 100.0, rtol=1e-12, atol=1e-12)
