"""
Main Real Competition simulation model (Steps 0 to 6) and invariant checks.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd
import mesa

from src.agents import FirmAgent, Vintage
from src.innovation import draw_dlf_innovation, select_best_practice


class RealCompetitionModel(mesa.Model):
    """Single-sector Real Competition ABM (Shaikh vs Okishio, annual Steps 0 to 6)."""

    def __init__(
        self,
        D: float = 82.0,
        g_D: float = 0.015,
        eta: float = 2.0,
        alpha_p: float = 0.15,
        p_0: float = 1.0,
        u_n: float = 0.82,
        T: int = 16,
        gamma: float = 1.30,
        delta_cost: float = 0.030,
        eps_0: float = 0.0,
        mu_z: float = 0.07,
        sigma_z: float = 0.025,
        lam: float = 0.70,
        beta_r: float = 2.6,
        beta_u: float = 0.14,
        R: float = 0.01,
        innovation_regime: str = "neutral",
        marx_a: float = 0.020,
        marx_b: Optional[float] = 0.005,
        marx_theta: float = -0.9310265801,
        marx_mu_uc: float = 0.032,
        marx_mu_kappa: float = -0.043,
        marx_mu_ucs: Optional[float] = None,
        gamma_thresh: float = 0.25,
        p_ref_okishio: Optional[float | str] = "normal_price",
        r_star: float = 0.04,
        interest_rate: float = 0.0,
        tau_low_profit: int = 10,
        enable_low_profit_exit: bool = False,
        evolutionary_entry: bool = False,
        enable_monopoly_pricing: bool = False,
        psi: float = 20.0,
        f_0: float = 0.06,
        K_0: float = 5.0,
        B_0: float = 30.0,
        s_min: float = 0.060,
        tau_grace: int = 5,
        seed: int = 42,
    ):
        super().__init__(rng=seed)
        self.D_0 = float(D)
        self.D = float(D)
        self.g_D = float(g_D)
        self.eta = float(eta)
        if not (0.0 <= float(alpha_p) <= 1.0):
            raise ValueError(f"alpha_p must be in [0, 1]: alpha_p={alpha_p}")
        self.alpha_p = float(alpha_p)
        self.p_0 = float(p_0)
        self.u_n = float(u_n)
        self.T = int(T)
        self.gamma = float(gamma)
        self.delta_cost = float(delta_cost)
        self.eps_0 = float(eps_0)
        self.mu_z = float(mu_z)
        self.sigma_z = float(sigma_z)
        self.lam = float(lam)
        self.beta_r = float(beta_r)
        self.beta_u = float(beta_u)
        self.R = float(R)
        self.innovation_regime = innovation_regime
        self.marx_a = float(marx_a)
        self.marx_b = float(marx_b) if marx_b is not None else (self.R ** 2) / self.marx_a
        self.marx_theta = float(marx_theta)
        mu_ucs_val = float(marx_mu_ucs) if marx_mu_ucs is not None else float(marx_mu_uc)
        self.marx_mu_uc = mu_ucs_val
        self.marx_mu_ucs = mu_ucs_val
        self.marx_mu_kappa = float(marx_mu_kappa)
        self.gamma_thresh = float(gamma_thresh)
        if isinstance(p_ref_okishio, str):
            if p_ref_okishio not in {"normal_price", "dynamic_normal"}:
                raise ValueError(f"Unknown p_ref_okishio mode: {p_ref_okishio}")
            self.p_ref_okishio: Optional[float | str] = "normal_price"
        elif p_ref_okishio is not None:
            self.p_ref_okishio = float(p_ref_okishio)
        else:
            self.p_ref_okishio = None

        self.r_star = float(r_star)
        self.interest_rate = float(interest_rate)
        self.tau_low_profit = int(tau_low_profit)
        self.enable_low_profit_exit = bool(enable_low_profit_exit)
        self.evolutionary_entry = bool(evolutionary_entry)
        self.enable_monopoly_pricing = bool(enable_monopoly_pricing)
        self.psi = float(psi)
        self.f_0 = float(f_0)
        self.K_0 = float(K_0)
        self.B_0 = float(B_0)
        self.s_min = float(s_min)
        self.tau_grace = int(tau_grace)

        self.rng = np.random.default_rng(seed)
        self.current_period = 0
        self.p_min_prev = float(p_0)
        self.p_tilde = float(p_0)
        self.r_branch = 0.0
        self.last_entries = 0
        self.last_exits_marginalized: list[int] = []
        self.last_exits_exhausted: list[int] = []
        self.last_exits_low_profit: list[int] = []

    def draw_truncated_normal(self) -> float:
        """Draw z_i(t) ~ N_+(mu_z, sigma_z^2) truncated to R_+ via rejection sampling."""
        while True:
            z = float(self.rng.normal(self.mu_z, self.sigma_z))
            if z >= 0.0:
                return z

    def draw_innovation(self) -> tuple[float, float]:
        """Draw uniform (d_ucs, d_kappa) in I_neutral(R) or I_Marx(a, b, theta, mu_ucs, mu_kappa)."""
        return draw_dlf_innovation(
            self.rng,
            self.innovation_regime,
            self.R,
            self.marx_a,
            self.marx_b,
            self.marx_theta,
            self.marx_mu_uc,
            self.marx_mu_kappa,
        )

    def step_0_demand(self) -> None:
        """Step 0: Adaptive-memory endogenous aggregate demand D(t)."""
        p_lag = self.p_0 if self.current_period <= 1 else self.p_min_prev
        if p_lag <= 0.0:
            raise ValueError(f"Invalid lagged leader price at Step 0: p_lag={p_lag}")
        if self.p_tilde <= 0.0:
            raise ValueError(f"Invalid adaptive price norm at Step 0: p_tilde={self.p_tilde}")
        self.D = float(
            self.D_0
            * ((1.0 + self.g_D) ** self.current_period)
            * ((self.p_tilde / p_lag) ** self.eta)
        )

    def step_1_pricing(self) -> None:
        """Step 1: Determine market regime (S >= D vs S < D) and post firm prices >= uc_bar_i."""
        firms = list(self.agents)
        if not firms:
            return
        if self.enable_monopoly_pricing and len(firms) == 1:
            monopolist = firms[0]
            z_i = self.draw_truncated_normal()
            monopolist.price = max(monopolist.mean_uc, monopolist.price * (1.0 + z_i))
            assert monopolist.price >= monopolist.mean_uc - 1e-12
            return
        S_t = sum(f.total_capacity for f in firms)
        min_p = min(f.price for f in firms)
        tied = [f for f in firms if abs(f.price - min_p) <= 1e-12]
        leader = tied[0] if len(tied) == 1 else tied[int(self.rng.integers(0, len(tied)))]
        p_leader = leader.price
        is_shortage = S_t < self.D
        for f in firms:
            f.set_price(p_leader=p_leader, is_shortage=is_shortage)
            assert f.price >= f.mean_uc - 1e-12, (
                f"Price floor violation for firm {f.unique_id}"
            )

    def step_2_demand_allocation(self) -> None:
        """Step 2: Quasi-replicator target share adjustment and capacity-constrained demand allocation."""
        firms = list(self.agents)
        if not firms:
            return
        p_l = min(f.price for f in firms)
        leaders = [f for f in firms if abs(f.price - p_l) <= 1e-12]
        sum_shed = 0.0
        for f in firms:
            if f not in leaders:
                old_f = f.target_share
                price_gap = (p_l - f.price) / p_l
                f.target_share = old_f * max(0.0, 1.0 + self.lam * price_gap)
                sum_shed += old_f - f.target_share
        if sum_shed > 0.0:
            sum_leaders_f = sum(f.target_share for f in leaders)
            if sum_leaders_f > 0.0:
                for f in leaders:
                    f.target_share += sum_shed * (f.target_share / sum_leaders_f)
            else:
                add_each = sum_shed / len(leaders)
                for f in leaders:
                    f.target_share += add_each

        sum_f = sum(f.target_share for f in firms)
        assert np.isclose(sum_f, 1.0, atol=1e-9), (
            f"Target market share conservation failed at Step 2: sum_f = {sum_f}"
        )

        q_max = {f: f.profitable_supply() for f in firms}
        d_init = {f: f.target_share * self.D for f in firms}
        q_init = {f: min(q_max[f], d_init[f]) for f in firms}

        delta_D = sum(d_init[f] - q_init[f] for f in firms)
        spare = {f: max(0.0, q_max[f] - q_init[f]) for f in firms}
        total_spare = sum(spare.values())

        for f in firms:
            if total_spare > 0.0 and delta_D > 0.0:
                f.sales = q_init[f] + spare[f] * min(1.0, delta_D / total_spare)
            else:
                f.sales = q_init[f]

        total_sales = sum(f.sales for f in firms)
        for f in firms:
            f.realized_share = (f.sales / total_sales) if total_sales > 0.0 else 0.0
            K_i = f.total_capacity
            f.u = (f.sales / K_i) if K_i > 0.0 else 0.0

    def step_3_production_and_accounting(self) -> None:
        """Step 3: Merit-order production, budget retention, and adaptive price-memory update."""
        firms = list(self.agents)
        if firms:
            self.p_min_prev = min(f.price for f in firms)
            self.p_tilde = (1.0 - self.alpha_p) * self.p_tilde + self.alpha_p * self.p_min_prev
        for f in firms:
            f.k_pre = f.total_capacity
            f.r_bar_pre = f.normal_profit_rate()
            f.u_pre = f.u
            f.produce_and_account()
            f.r_real_pre = f.r_real
            if self.enable_low_profit_exit:
                if f.r_real < self.interest_rate:
                    f.low_profit_streak += 1
                else:
                    f.low_profit_streak = 0

    def step_4_scrapping(self) -> None:
        """Step 4: One-hoss shay retirement (age >= T) and economic scrapping (uc_v > p_i)."""
        for f in self.agents:
            f.scrap_vintages(self.current_period)

    def step_5_innovation_and_investment(self) -> None:
        """Step 5: DLF technical search, adoption, and budget-constrained capacity expansion."""
        for f in self.agents:
            budget_before = f.budget
            k_surv_before = f.k_surv
            f.innovate_and_invest(self.current_period)
            assert f.budget >= -1e-9 and f.budget <= budget_before + 1e-9, (
                f"Budget invariant violated for firm {f.unique_id}"
            )
            assert f.total_capacity >= k_surv_before - 1e-9

    def _best_practice_technique(self, rule: str, p_ref: float, firms: list) -> tuple[float, float]:
        """Identify best-practice technique in `rule` evaluated at `p_ref`."""
        return select_best_practice(rule, p_ref, firms)

    def step_6_entry_and_exit(self, capital_value_t: float) -> None:
        """Step 6: Branch realized profit rate, exits (marginalized/exhausted), and Poisson entry."""
        firms = list(self.agents)
        self.last_entries = 0
        self.last_exits_marginalized = []
        self.last_exits_exhausted = []
        self.last_exits_low_profit = []
        if not firms:
            return

        total_profits = sum(f.profit for f in firms)
        self.r_branch = (total_profits / capital_value_t) if capital_value_t > 0.0 else 0.0
        p_leader_now = min(f.price for f in firms)

        best_shaikh = self._best_practice_technique("shaikh", p_leader_now, firms)
        best_okishio = self._best_practice_technique("okishio", p_leader_now, firms)

        if self.evolutionary_entry:
            s_shaikh_tot = sum(f.realized_share for f in firms if f.rule == "shaikh")
            s_tot = sum(f.realized_share for f in firms)
            p_shaikh_entry = (s_shaikh_tot / s_tot) if s_tot > 0.0 else 0.5
        else:
            p_shaikh_entry = 0.5

        survivors = []
        lp_candidates = []
        for f in firms:
            firm_age = self.current_period - f.entry_period
            marginalized = (firm_age >= self.tau_grace) and (f.realized_share < self.s_min)
            min_kappa_avail = min(f.kappa_b, f.kappa_a) if f.kappa_a is not None else f.kappa_b
            exhausted = (f.total_capacity <= 1e-12) and (f.budget < min_kappa_avail)

            if marginalized or exhausted:
                if marginalized:
                    self.last_exits_marginalized.append(f.unique_id)
                else:
                    self.last_exits_exhausted.append(f.unique_id)
                f.remove()
            else:
                survivors.append(f)
                if (
                    self.enable_low_profit_exit
                    and (firm_age >= self.tau_grace)
                    and (abs(f.price - p_leader_now) > 1e-12)
                    and (f.low_profit_streak >= self.tau_low_profit)
                ):
                    cap_i = sum(v.kappa * v.k for v in f.vintages)
                    r_i = (f.profit / cap_i) if cap_i > 0.0 else 0.0
                    lp_candidates.append((r_i, f))

        if self.enable_low_profit_exit and lp_candidates and len(survivors) > 1:
            lp_candidates.sort(key=lambda x: x[0])
            _, f_out = lp_candidates[0]
            survivors.remove(f_out)
            self.last_exits_low_profit.append(f_out.unique_id)
            f_out.remove()

        sum_f_surv = sum(f.target_share for f in survivors)
        if sum_f_surv > 0.0:
            for f in survivors:
                f.target_share /= sum_f_surv
        elif survivors:
            for f in survivors:
                f.target_share = 1.0 / len(survivors)

        excess_r = max(0.0, self.r_branch - self.r_star)
        n_entry = int(self.rng.poisson(self.psi * excess_r))
        if self.f_0 > 0.0:
            max_entries = int((1.0 - 1e-9) / self.f_0)
            n_entry = min(n_entry, max_entries)

        self.last_entries = n_entry
        if n_entry > 0:
            if survivors:
                scale_factor = 1.0 - n_entry * self.f_0
                for f in survivors:
                    f.target_share *= scale_factor
                entrant_share = self.f_0
            else:
                entrant_share = 1.0 / n_entry

            next_t = self.current_period + 1
            scale = (self.D / self.D_0) if self.D_0 > 0.0 else 1.0
            k_entry = self.K_0 * scale
            b_entry = self.B_0 * scale
            for _ in range(n_entry):
                rule = "shaikh" if self.rng.random() < p_shaikh_entry else "okishio"
                uc_bp, kappa_bp = best_shaikh if rule == "shaikh" else best_okishio
                v_init = Vintage(
                    uc=uc_bp,
                    kappa=kappa_bp,
                    k_init=k_entry,
                    k=k_entry,
                    birth_period=next_t,
                )
                FirmAgent(
                    model=self,
                    rule=rule,
                    initial_vintages=[v_init],
                    uc_b=uc_bp,
                    kappa_b=kappa_bp,
                    price=p_leader_now,
                    target_share=entrant_share,
                    budget=b_entry,
                    u_init=self.u_n,
                    entry_period=next_t,
                )

        active_firms = list(self.agents)
        if active_firms:
            sum_f_post = sum(f.target_share for f in active_firms)
            assert np.isclose(sum_f_post, 1.0, atol=1e-9), (
                f"Target market share conservation failed at Step 6: sum_f = {sum_f_post}"
            )

    def check_invariants(
        self, check_prime_costs: bool = True, check_realized_shares: bool = False
    ) -> dict[str, Any]:
        """Verify structural accounting, pricing, budget, and prime-cost invariants."""
        firms = list(self.agents)
        if not firms:
            return {
                "n_firms": 0,
                "sum_target_share": 0.0,
                "sum_realized_share": 0.0,
                "min_price_margin": np.nan,
                "min_budget": np.nan,
                "min_ucs": np.nan,
            }

        sum_target = float(sum(f.target_share for f in firms))
        sum_realized = float(sum(f.realized_share for f in firms))
        min_margin = float(min(f.price - f.mean_uc for f in firms))
        min_budget = float(min(f.budget for f in firms))
        dep_denom = self.T * self.u_n
        ucs_values = [
            v.uc - v.kappa / dep_denom for f in firms for v in f.vintages
        ]
        min_ucs = float(min(ucs_values)) if ucs_values else np.nan

        assert np.isclose(sum_target, 1.0, atol=1e-8), f"sum(f_i)={sum_target} != 1"
        if check_realized_shares and sum(f.sales for f in firms) > 0.0:
            assert np.isclose(sum_realized, 1.0, atol=1e-8), f"sum(s_i)={sum_realized} != 1"
        assert min_margin >= -1e-12, f"Price floor violated: min(p - uc_bar)={min_margin}"
        assert min_budget >= -1e-9, f"Negative budget detected: min(B_i)={min_budget}"
        if check_prime_costs and ucs_values:
            assert min_ucs > 0.0, f"Non-positive implied prime cost detected: min(ucs_v)={min_ucs}"

        return {
            "n_firms": len(firms),
            "sum_target_share": sum_target,
            "sum_realized_share": sum_realized,
            "min_price_margin": min_margin,
            "min_budget": min_budget,
            "min_ucs": min_ucs,
        }

    def step(self) -> None:
        """Execute one complete annual period t -> t+1 in strict Step 0..6 order."""
        self.current_period += 1
        self.step_0_demand()
        capital_value_t = sum(v.kappa * v.k for f in self.agents for v in f.vintages)
        self.step_1_pricing()
        self.step_2_demand_allocation()
        self.step_3_production_and_accounting()
        self.step_4_scrapping()
        self.step_5_innovation_and_investment()
        self.step_6_entry_and_exit(capital_value_t)


def run_detailed_simulation(
    n_steps: int = 120,
    innovation_regime: str = "marx",
    seed: int = 42,
    overrides: Optional[dict[str, Any]] = None,
    *,
    D: Optional[float] = None,
    r_star: Optional[float] = None,
    interest_rate: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a single detailed annual simulation, returning firm-level and branch-level panels."""
    from src.calibration import build_baseline_model, init_paired_homogeneous_firms

    settings = dict(overrides or {})
    for name, value in [("D", D), ("r_star", r_star), ("interest_rate", interest_rate)]:
        if value is not None:
            settings[name] = value
    model = build_baseline_model(
        innovation_regime=innovation_regime, seed=seed, overrides=settings
    )
    init_paired_homogeneous_firms(model, seed=seed)

    firm_records = []
    agg_records = []

    for t in range(1, n_steps + 1):
        model.current_period += 1
        firms_t = list(model.agents)
        if not firms_t:
            break

        model.step_0_demand()
        S_t = sum(f.total_capacity for f in firms_t)
        p_leader_in = min(f.price for f in firms_t)
        is_shortage = S_t < model.D
        capital_value_t = sum(v.kappa * v.k for f in firms_t for v in f.vintages)

        model.step_1_pricing()
        model.step_2_demand_allocation()
        model.step_3_production_and_accounting()

        step_records = {}
        for f in firms_t:
            step_records[f.unique_id] = {
                "t": t,
                "firm_id": f.unique_id,
                "rule": f.rule,
                "entry_period": f.entry_period,
                "K_i": f.total_capacity,
                "uc_bar_i": f.mean_uc,
                "kappa_bar_i": f.mean_kappa,
                "p_i": f.price,
                "r_bar_i": f.normal_profit_rate(),
                "f_i": f.target_share,
                "q_max_i": f.profitable_supply(),
                "q_i": f.sales,
                "s_i": f.realized_share,
                "u_i": f.u,
                "profit_i": f.profit,
                "B_i": f.budget,
                "exited": False,
                "exit_reason": None,
            }

        p_min_t = min(f.price for f in firms_t)
        total_sales_t = sum(f.sales for f in firms_t)
        total_q_max_t = sum(f.profitable_supply() for f in firms_t)
        p_avg_t = (
            sum(f.price * f.sales for f in firms_t) / total_sales_t
            if total_sales_t > 0
            else p_min_t
        )

        shaikh_t = [f for f in firms_t if f.rule == "shaikh"]
        okishio_t = [f for f in firms_t if f.rule == "okishio"]
        K_shaikh = sum(f.total_capacity for f in shaikh_t)
        K_okishio = sum(f.total_capacity for f in okishio_t)

        uc_shaikh = (
            sum(f.mean_uc * f.total_capacity for f in shaikh_t) / K_shaikh
            if K_shaikh > 0
            else np.nan
        )
        uc_okishio = (
            sum(f.mean_uc * f.total_capacity for f in okishio_t) / K_okishio
            if K_okishio > 0
            else np.nan
        )
        r_bar_shaikh = (
            sum(f.normal_profit_rate() * f.total_capacity for f in shaikh_t) / K_shaikh
            if K_shaikh > 0
            else np.nan
        )
        r_bar_okishio = (
            sum(f.normal_profit_rate() * f.total_capacity for f in okishio_t) / K_okishio
            if K_okishio > 0
            else np.nan
        )
        u_shaikh = (sum(f.sales for f in shaikh_t) / K_shaikh) if K_shaikh > 0 else np.nan
        u_okishio = (sum(f.sales for f in okishio_t) / K_okishio) if K_okishio > 0 else np.nan
        share_shaikh = sum(f.realized_share for f in shaikh_t)
        share_okishio = sum(f.realized_share for f in okishio_t)

        model.step_4_scrapping()
        model.step_5_innovation_and_investment()
        model.step_6_entry_and_exit(capital_value_t)

        for fid in model.last_exits_marginalized:
            if fid in step_records:
                step_records[fid]["exited"] = True
                step_records[fid]["exit_reason"] = "marginalized"
        for fid in model.last_exits_exhausted:
            if fid in step_records:
                step_records[fid]["exited"] = True
                step_records[fid]["exit_reason"] = "exhausted"
        for fid in model.last_exits_low_profit:
            if fid in step_records:
                step_records[fid]["exited"] = True
                step_records[fid]["exit_reason"] = "low_profit"

        firm_records.extend(step_records.values())
        agg_records.append(
            {
                "t": t,
                "n_firms": len(firms_t),
                "n_shaikh": len(shaikh_t),
                "n_okishio": len(okishio_t),
                "S_t": S_t,
                "Q_max_t": total_q_max_t,
                "Q_t": total_sales_t,
                "D_t": model.D,
                "D_trend_t": model.D_0 * ((1.0 + model.g_D) ** t),
                "is_shortage": is_shortage,
                "p_leader_in": p_leader_in,
                "p_min_t": p_min_t,
                "p_avg_t": p_avg_t,
                "uc_shaikh": uc_shaikh,
                "uc_okishio": uc_okishio,
                "r_branch": model.r_branch,
                "r_star": model.r_star,
                "r_bar_shaikh": r_bar_shaikh,
                "r_bar_okishio": r_bar_okishio,
                "share_shaikh": share_shaikh,
                "share_okishio": share_okishio,
                "u_industry": (total_sales_t / S_t) if S_t > 0 else 0.0,
                "u_shaikh": u_shaikh,
                "u_okishio": u_okishio,
                "n_entries": model.last_entries,
                "n_exits_marg": len(model.last_exits_marginalized),
                "n_exits_exh": len(model.last_exits_exhausted),
                "n_exits_lp": len(model.last_exits_low_profit),
                "n_exits_total": (
                    len(model.last_exits_marginalized)
                    + len(model.last_exits_exhausted)
                    + len(model.last_exits_low_profit)
                ),
            }
        )

    return pd.DataFrame(firm_records), pd.DataFrame(agg_records)
