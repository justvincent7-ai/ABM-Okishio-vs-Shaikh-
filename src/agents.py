"""
Microeconomic agents and fixed-capital vintages (one-hoss shay depreciation,
merit-order production, real-competition pricing, and Dumenil-Levy accumulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import mesa

from src.innovation import apply_dlf_transformation, evaluate_adoption_criterion


@dataclass
class Vintage:
    """Single fixed-capital vintage owned by a firm (one-hoss shay service life T)."""

    uc: float          # Complete unit cost: uc = ucs + kappa / (T * u_n)
    kappa: float       # Unit capital cost (capital-output ratio at full capacity)
    k_init: float      # Initial productive capacity Delta k_v
    k: float           # Surviving productive capacity k_v(t)
    birth_period: int  # Acquisition period tau_v

    def prime_cost(self, T: int, u_n: float) -> float:
        """Recover implied prime cost ucs_v = uc_v - kappa_v / (T * u_n)."""
        return float(self.uc - self.kappa / (T * u_n))

    def age_at(self, t: int) -> int:
        """Elapsed service periods at the end of period t."""
        return int(t - self.birth_period + 1)


class FirmAgent(mesa.Agent):
    """Real-competition firm managing a heterogeneous capital-vintage portfolio."""

    def __init__(
        self,
        model: mesa.Model,
        rule: str,
        initial_vintages: list[Vintage],
        uc_b: float,
        kappa_b: float,
        price: float,
        target_share: float,
        budget: float,
        u_init: float = 0.85,
        entry_period: int = 0,
    ):
        super().__init__(model)
        self.rule = rule
        self.vintages = list(initial_vintages)
        self.uc_b = float(uc_b)
        self.kappa_b = float(kappa_b)
        self.uc_a: Optional[float] = None
        self.kappa_a: Optional[float] = None

        self.price = float(price)
        self.target_share = float(target_share)
        self.realized_share = float(target_share)
        self.budget = float(budget)
        self.u = float(u_init)
        self.entry_period = int(entry_period)
        self.sales = 0.0
        self.profit = 0.0
        self.k_surv = self.total_capacity
        self.k_pre = self.total_capacity
        self.r_bar_pre = self.normal_profit_rate()
        self.r_real = (
            self.u * ((self.price - self.mean_uc) / self.mean_kappa)
            if (self.total_capacity > 0.0 and self.mean_kappa > 0.0)
            else 0.0
        )
        self.r_real_pre = self.r_real
        self.u_pre = self.u
        self.low_profit_streak = 0

    @property
    def total_capacity(self) -> float:
        """K_i(t) = sum_{v in C_i(t)} k_v(t)."""
        return float(sum(v.k for v in self.vintages))

    @property
    def mean_uc(self) -> float:
        """Capacity-weighted mean complete unit cost uc_bar_i(t)."""
        K = self.total_capacity
        if K <= 0.0:
            return self.uc_b
        return float(sum(v.uc * v.k for v in self.vintages) / K)

    @property
    def mean_kappa(self) -> float:
        """Capacity-weighted mean unit capital cost kappa_bar_i(t)."""
        K = self.total_capacity
        if K <= 0.0:
            return self.kappa_b
        return float(sum(v.kappa * v.k for v in self.vintages) / K)

    @property
    def capital_value(self) -> float:
        """Installed replacement capital value sum_{v in C_i(t)} kappa_v * k_v(t)."""
        return float(sum(v.kappa * v.k for v in self.vintages))

    def normal_profit_rate(self) -> float:
        """Normal profit rate evaluated at u_n: u_n * (p_i(t) - uc_bar_i(t)) / kappa_bar_i(t)."""
        K = self.total_capacity
        if K <= 0.0 or self.mean_kappa <= 0.0:
            return 0.0
        u_n = float(getattr(self.model, "u_n", 0.82))
        return u_n * (self.price - self.mean_uc) / self.mean_kappa

    def set_price(self, p_leader: float, is_shortage: bool) -> None:
        """Step 1: Price adjustment under normal (S >= D) or shortage (S < D) regime."""
        p_floor = self.mean_uc
        orders = self.target_share * self.model.D
        if not is_shortage:
            norm_cap = self.model.u_n * self.total_capacity
            if norm_cap >= orders or orders <= 1e-12:
                cost_adv = max(0.0, (p_leader - p_floor) / p_leader) if p_leader > 0.0 else 0.0
                eps_i = max(0.0, self.model.delta_cost * cost_adv + self.model.eps_0)
                eps_i = min(eps_i, 1.0 - 1e-12)
                self.price = max(p_floor, p_leader * (1.0 - eps_i))
            else:
                shortfall = 1.0 - (norm_cap / orders)
                self.price = max(p_floor, self.price * (1.0 + self.model.gamma * shortfall))
        else:
            z_i = self.model.draw_truncated_normal()
            self.price = max(p_floor, self.price, p_leader * (1.0 + z_i))

        assert self.price >= p_floor - 1e-12, (
            f"Price floor violation for firm {self.unique_id}: "
            f"p_i(t)={self.price} < uc_bar_i(t)={p_floor}"
        )

    def profitable_supply(self) -> float:
        """q_i^max(t) = sum_{v in C_i(t), uc_v <= p_i(t)} k_v(t)."""
        return float(sum(v.k for v in self.vintages if v.uc <= self.price + 1e-12))

    def produce_and_account(self) -> None:
        """Step 3: Merit-order production allocation, net profit, and retained cash flow B_i(t)."""
        sorted_vintages = sorted(self.vintages, key=lambda v: (v.uc, -v.birth_period))
        remaining_output = self.sales
        net_profit = 0.0
        depreciation_allowance = 0.0
        dep_denom = self.model.T * self.model.u_n
        for v in sorted_vintages:
            if remaining_output <= 0.0:
                break
            if v.uc > self.price + 1e-12:
                continue
            q_v = min(v.k, remaining_output)
            net_profit += (self.price - v.uc) * q_v
            depreciation_allowance += (v.kappa / dep_denom) * q_v
            remaining_output -= q_v
        self.profit = net_profit
        self.budget += net_profit + depreciation_allowance
        cap_val = sum(v.kappa * v.k for v in self.vintages)
        self.r_real = (net_profit / cap_val) if cap_val > 0.0 else 0.0

    def scrap_vintages(self, current_t: int) -> None:
        """Step 4: One-hoss shay retirement (age >= T) and economic obsolescence (uc_v > p_i(t))."""
        self.k_pre = self.total_capacity
        self.r_bar_pre = self.normal_profit_rate()
        self.u_pre = self.u
        cap_val = sum(v.kappa * v.k for v in self.vintages)
        self.r_real_pre = (self.profit / cap_val) if cap_val > 0.0 else 0.0
        surviving = []
        for v in self.vintages:
            age = current_t - v.birth_period + 1
            if age < self.model.T and v.uc <= self.price + 1e-12:
                v.k = v.k_init
                surviving.append(v)
        self.vintages = surviving
        self.k_surv = self.total_capacity

    def _frontier_capacity_share(self) -> float:
        """Share of surviving capacity embodying frontier technique (uc_a, kappa_a)."""
        K = self.total_capacity
        if self.uc_a is None or K <= 0.0:
            return 0.0
        k_frontier = sum(
            v.k
            for v in self.vintages
            if np.isclose(v.uc, self.uc_a, rtol=1e-9, atol=1e-12)
            and np.isclose(v.kappa, self.kappa_a, rtol=1e-9, atol=1e-12)
        )
        return float(k_frontier / K)

    def innovate_and_invest(
        self, current_t: int, forced_draw: Optional[tuple[float, float]] = None
    ) -> None:
        """Step 5: Symmetric frontier consolidation, DLF R&D, adoption, and pure-internal investment."""
        can_search = True
        if self.uc_a is not None:
            if self._frontier_capacity_share() >= self.model.gamma_thresh - 1e-12:
                self.uc_b, self.kappa_b = self.uc_a, self.kappa_a
                self.uc_a, self.kappa_a = None, None
            else:
                can_search = False

        if can_search:
            d_ucs, d_kappa = (
                forced_draw if forced_draw is not None else self.model.draw_innovation()
            )
            _, uc_plus, kappa_plus = apply_dlf_transformation(
                self.uc_b, self.kappa_b, d_ucs, d_kappa, self.model.T, self.model.u_n
            )
            if evaluate_adoption_criterion(
                self.rule,
                self.uc_b,
                self.kappa_b,
                uc_plus,
                kappa_plus,
                u_n=self.model.u_n,
                r_star=self.model.r_star,
                current_price=self.price,
                p_ref_okishio=self.model.p_ref_okishio,
            ):
                self.uc_a, self.kappa_a = uc_plus, kappa_plus

        uc_act = self.uc_a if self.uc_a is not None else self.uc_b
        kappa_act = self.kappa_a if self.kappa_a is not None else self.kappa_b

        assert self.budget >= -1e-12, f"Negative budget before investment: B_i(t)={self.budget}"
        if self.budget <= 0.0:
            self.k_surv = self.total_capacity
            return

        min_kappa_avail = (
            min(self.kappa_b, self.kappa_a) if self.kappa_a is not None else self.kappa_b
        )
        if self.k_surv <= 1e-12 and self.budget < min_kappa_avail:
            self.k_surv = self.total_capacity
            return

        budget_before = self.budget
        replacement = max(0.0, self.k_pre - self.k_surv)
        trend_acc = self.model.g_D * self.k_pre
        dl_net = self.k_pre * (
            self.model.beta_r * (self.r_real_pre - self.model.r_star)
            + self.model.beta_u * (self.u_pre - self.model.u_n)
        )
        I_req = max(0.0, trend_acc + replacement + dl_net)
        dk_i = min(I_req, budget_before / kappa_act)
        cost_total = kappa_act * dk_i

        assert cost_total <= budget_before + 1e-9, (
            f"Budget exceeded for firm {self.unique_id}: cost={cost_total} > B_i(t)={budget_before}"
        )

        if dk_i > 0.0:
            new_birth = current_t + 1  # Initial age a_v = 0 entering t+1
            self.vintages.append(Vintage(uc_act, kappa_act, dk_i, dk_i, new_birth))
            self.budget = max(0.0, budget_before - cost_total)

        self.k_surv = self.total_capacity
        if (
            self.uc_a is not None
            and self._frontier_capacity_share() >= self.model.gamma_thresh - 1e-12
        ):
            self.uc_b, self.kappa_b = self.uc_a, self.kappa_a
            self.uc_a, self.kappa_a = None, None

        assert self.budget >= -1e-9, f"Negative budget after investment: {self.budget}"
