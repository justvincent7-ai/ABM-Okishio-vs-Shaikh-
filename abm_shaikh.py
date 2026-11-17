import os
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mesa


@dataclass
class Vintage:
    """Représente une génération de capital fixe (un équipement) détenue par une firme."""
    uc: float           # Coût unitaire d'exploitation (hors amortissement)
    kappa: float        # Coût unitaire en capital
    k_init: float       # Capacité productive initiale acquise (Delta k_v)
    k: float            # Capacité productive survivante actuelle k_v(t)
    birth_period: int   # Période d'installation tau_v


class FirmAgent(mesa.Agent):
    """Représente une firme qui gère un portefeuille de capitaux (vintages) et fixe ses prix."""
    def __init__(
        self,
        model: mesa.Model,
        rule: str,
        initial_vintages: list,
        uc_b: float,
        kappa_b: float,
        price: float,
        target_share: float,
        budget: float,
        u_init: float = 0.85,
        entry_period: int = 0
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

    @property
    def total_capacity(self) -> float:
        """K_i(t) = sum_{v in C_i(t)} k_v(t)"""
        return float(sum(v.k for v in self.vintages))

    @property
    def mean_uc(self) -> float:
        """uc_bar_i(t) = (1 / K_i(t)) * sum_{v in C_i(t)} uc_v * k_v(t)"""
        K = self.total_capacity
        if K <= 0.0:
            return self.uc_b
        return float(sum(v.uc * v.k for v in self.vintages) / K)

    @property
    def mean_kappa(self) -> float:
        """kappa_bar_i(t) = (1 / K_i(t)) * sum_{v in C_i(t)} kappa_v * k_v(t)"""
        K = self.total_capacity
        if K <= 0.0:
            return self.kappa_b
        return float(sum(v.kappa * v.k for v in self.vintages) / K)

    def normal_profit_rate(self) -> float:
        """r_bar_i(t) = (p_i(t) - uc_bar_i(t)) / kappa_bar_i(t)"""
        K = self.total_capacity
        if K <= 0.0 or self.mean_kappa <= 0.0:
            return 0.0
        return (self.price - self.mean_uc) / self.mean_kappa

    def set_price(self, p_leader: float, is_shortage: bool, kappa_ref: float = 0.0):
        """Étape 1 : Fixation du prix selon le régime normal (S >= D) ou pénurie (S < D)."""
        p_floor = self.mean_uc
        if not is_shortage:
            p_repro = p_floor + (kappa_ref / (self.model.T * self.model.u_n) if kappa_ref > 0.0 else 0.0)
            cost_adv = max(0.0, (p_leader - p_repro) / p_leader) if p_leader > 0.0 else 0.0
            eps_i = max(
                0.0,
                self.model.gamma * (self.model.u_n - self.u)
                + self.model.delta_cost * cost_adv
                + self.model.eps_0
            )
            eps_i = min(eps_i, 1.0 - 1e-12)  # Garantit epsilon_i(t) in [0, 1)
            self.price = max(p_repro, p_leader * (1.0 - eps_i))
        else:
            z_i = self.model.draw_truncated_normal()
            self.price = max(p_floor, self.price, p_leader * (1.0 + z_i))

        assert self.price >= p_floor - 1e-12, (
            f"Violation du plancher de prix pour la firme {self.unique_id}: "
            f"p_i(t)={self.price} < uc_bar_i(t)={p_floor}"
        )

    def profitable_supply(self) -> float:
        """q_i^max(t) = sum_{v in C_i(t), uc_v <= p_i(t)} k_v(t)"""
        return float(sum(v.k for v in self.vintages if v.uc <= self.price + 1e-12))

    def produce_and_account(self):
        """Étape 3 : Activation par ordre de mérite (uc_v croissant, âge croissant / tau_v décroissant)."""
        sorted_vintages = sorted(self.vintages, key=lambda v: (v.uc, -v.birth_period))
        remaining_output = self.sales
        net_profit = 0.0
        for v in sorted_vintages:
            if remaining_output <= 0.0:
                break
            if v.uc > self.price + 1e-12:
                continue
            q_v = min(v.k, remaining_output)
            net_profit += (self.price - v.uc) * q_v
            remaining_output -= q_v
        self.profit = net_profit
        self.budget += self.profit

    def scrap_vintages(self, current_t: int):
        """Étape 4 : Déclassement physique (a_v(t) >= T) et économique (uc_v > p_i(t))."""
        surviving = []
        for v in self.vintages:
            age = current_t - v.birth_period + 1
            if age < self.model.T and v.uc <= self.price + 1e-12:
                v.k = v.k_init
                surviving.append(v)
        self.vintages = surviving
        self.k_surv = self.total_capacity

    def _frontier_capacity_share(self) -> float:
        """Part de la capacité active incarnée dans la technique frontière (uc_a, kappa_a)."""
        K = self.total_capacity
        if self.uc_a is None or K <= 0.0:
            return 0.0
        k_frontier = sum(
            v.k for v in self.vintages
            if np.isclose(v.uc, self.uc_a, rtol=1e-9, atol=1e-12)
            and np.isclose(v.kappa, self.kappa_a, rtol=1e-9, atol=1e-12)
        )
        return float(k_frontier / K)

    def innovate_and_invest(self, current_t: int, forced_draw: Optional[tuple] = None):
        """Étape 5 : Innovation DLF, adoption (Shaikh vs Okishio), consolidation symétrique et investissement."""
        # --- 1. Vérification de la consolidation symétrique (Shaikh et Okishio) ---
        can_search = True
        if self.uc_a is not None:
            if self._frontier_capacity_share() >= self.model.gamma_thresh - 1e-12:
                self.uc_b, self.kappa_b = self.uc_a, self.kappa_a
                self.uc_a, self.kappa_a = None, None
            else:
                can_search = False

        # --- 2. Tirage et évaluation d'une innovation candidate ---
        if can_search:
            d_uc, d_kappa = forced_draw if forced_draw is not None else self.model.draw_innovation()
            uc_plus = self.uc_b / (1.0 + d_uc)
            kappa_plus = self.kappa_b / (1.0 + d_kappa)

            if self.rule == "okishio":
                r_b = (self.price - self.uc_b) / self.kappa_b
                r_plus = (self.price - uc_plus) / kappa_plus
                if r_plus > r_b:
                    self.uc_a, self.kappa_a = uc_plus, kappa_plus
            elif self.rule == "shaikh":
                if uc_plus < self.uc_b:  # Critère de réduction des coûts (Delta uc > 0)
                    self.uc_a, self.kappa_a = uc_plus, kappa_plus

        # --- 3. Technique active (uc_act, kappa_act) et investissement Delta k_i(t) ---
        uc_act = self.uc_a if self.uc_a is not None else self.uc_b
        kappa_act = self.kappa_a if self.kappa_a is not None else self.kappa_b

        assert self.budget >= -1e-12, f"Budget négatif avant investissement : B_i(t)={self.budget}"
        if self.budget <= 0.0:
            return

        # Si la firme n'a plus aucune capacité survivante (K_i^surv(t) = 0) et que son budget B_i(t)
        # est inférieur au coût minimal d'une unité de capacité B_{min,i}, elle est en épuisement
        # physique et financier (Condition 2 de sortie à l'Étape 6).
        min_kappa_avail = min(self.kappa_b, self.kappa_a) if self.kappa_a is not None else self.kappa_b
        if self.k_surv <= 1e-12 and self.budget < min_kappa_avail:
            return

        budget_before = self.budget
        effective_demand = max(self.target_share * self.model.D, self.sales)
        k_desired = effective_demand / self.model.u_n
        I_req = max(0.0, k_desired - self.k_surv)
        dk_i = min(I_req, budget_before / kappa_act)
        cost_total = kappa_act * dk_i

        assert cost_total <= budget_before + 1e-9, (
            f"Dépassement de budget pour la firme {self.unique_id} : coût={cost_total} > B_i(t)={budget_before}"
        )

        if dk_i > 0.0:
            new_birth = current_t + 1  # Âge initial a_v = 0 en entrant en t+1
            self.vintages.append(Vintage(uc_act, kappa_act, dk_i, dk_i, new_birth))
            self.budget = max(0.0, budget_before - cost_total)

        # --- 4. Vérification symétrique de la promotion de (uc_a, kappa_a) si gamma_thresh est atteint ---
        self.k_surv = self.total_capacity
        if self.uc_a is not None and self._frontier_capacity_share() >= self.model.gamma_thresh - 1e-12:
            self.uc_b, self.kappa_b = self.uc_a, self.kappa_a
            self.uc_a, self.kappa_a = None, None

        assert self.budget >= -1e-9, f"Budget négatif après investissement : {self.budget}"


class RealCompetitionModel(mesa.Model):
    """Modèle complet de concurrence réelle d'Anwar Shaikh (Étapes 1 à 6)."""
    def __init__(
        self,
        D: float = 85.0,
        u_n: float = 0.85,
        T: int = 10,
        gamma: float = 0.03,
        delta_cost: float = 0.08,
        eps_0: float = 0.005,
        mu_z: float = 0.01,
        sigma_z: float = 0.005,
        lam: float = 0.8,
        R: float = 0.05,
        innovation_regime: str = "neutral",
        marx_a: float = 0.08,
        marx_theta: float = -np.pi / 4,
        gamma_thresh: float = 0.75,
        r_star: float = 0.12,
        psi: float = 5.0,
        f_0: float = 0.02,
        K_0: float = 10.0,
        B_0: float = 15.0,
        s_min: float = 0.01,
        tau_grace: int = 20,
        seed: int = 42
    ):
        super().__init__(rng=seed)
        self.D = float(D)
        self.u_n = float(u_n)
        self.T = int(T)
        self.gamma = float(gamma)
        self.delta_cost = float(delta_cost)
        self.eps_0 = float(eps_0)
        self.mu_z = float(mu_z)
        self.sigma_z = float(sigma_z)
        self.lam = float(lam)
        self.R = float(R)
        self.innovation_regime = innovation_regime
        self.marx_a = float(marx_a)
        self.marx_b = (self.R ** 2) / self.marx_a
        self.marx_theta = float(marx_theta)
        self.gamma_thresh = float(gamma_thresh)

        # Paramètres d'entrée / sortie (Étape 6)
        self.r_star = float(r_star)
        self.psi = float(psi)
        self.f_0 = float(f_0)
        self.K_0 = float(K_0)
        self.B_0 = float(B_0)
        self.s_min = float(s_min)
        self.tau_grace = int(tau_grace)

        self.rng = np.random.default_rng(seed)
        self.current_period = 0
        self.r_branch = 0.0
        self.last_entries = 0
        self.last_exits_marginalized = []
        self.last_exits_exhausted = []

    def draw_truncated_normal(self) -> float:
        """Tire z_i(t) ~ N_+(mu_z, sigma_z^2) tronquée à R_+ (par rejet)."""
        while True:
            z = float(self.rng.normal(self.mu_z, self.sigma_z))
            if z >= 0.0:
                return z

    def draw_innovation(self) -> tuple:
        """Tire uniformément (Delta uc, Delta kappa) dans I_neutral(R) ou I_Marx(a, b, theta)."""
        if self.innovation_regime == "neutral":
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            radius = self.R * np.sqrt(self.rng.uniform(0.0, 1.0))
            return float(radius * np.cos(angle)), float(radius * np.sin(angle))
        else:
            c, s = np.cos(self.marx_theta), np.sin(self.marx_theta)
            bound = max(self.marx_a, self.marx_b)
            while True:
                x = self.rng.uniform(-bound, bound)
                y = self.rng.uniform(-bound, bound)
                u_rot = x * c + y * s
                v_rot = -x * s + y * c
                if (u_rot / self.marx_a) ** 2 + (v_rot / self.marx_b) ** 2 <= 1.0:
                    return float(x), float(y)

    def step_1_pricing(self):
        """Étape 1 : Détermination du régime de marché et fixation des prix p_i(t) >= uc_bar_i(t)."""
        firms = list(self.agents)
        if not firms:
            return
        S_t = sum(f.total_capacity for f in firms)
        p_leader = min(f.price for f in firms)
        is_shortage = S_t < self.D
        kappa_branch = (
            sum(f.mean_kappa * f.total_capacity for f in firms) / S_t
            if (self.current_period > 0 and S_t > 0.0)
            else 0.0
        )
        for f in firms:
            f.set_price(p_leader=p_leader, is_shortage=is_shortage, kappa_ref=kappa_branch)
            assert f.price >= f.mean_uc - 1e-12, (
                f"Assertion p_i(t) >= uc_bar_i(t) échouée pour la firme {f.unique_id}"
            )

    def step_2_demand_allocation(self):
        """Étape 2 : Réallocation quasi-réplicatrice des parts cibles f_i(t) et ventes réalisées q_i(t)."""
        firms = list(self.agents)
        if not firms:
            return
        leader = min(firms, key=lambda f: f.price)
        p_l = leader.price
        sum_rivals_f = 0.0
        for f in firms:
            if f is not leader:
                price_gap = (p_l - f.price) / p_l
                f.target_share = f.target_share * max(0.0, 1.0 + self.lam * price_gap)
                sum_rivals_f += f.target_share
        leader.target_share = max(0.0, 1.0 - sum_rivals_f)

        # Assertion : conservation des parts de marché cibles sum_i f_i(t) = 1
        sum_f = sum(f.target_share for f in firms)
        assert np.isclose(sum_f, 1.0, atol=1e-9), (
            f"Assertion sum(f_i(t)) = 1 échouée à l'Étape 2 : sum_f = {sum_f}"
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

    def step_3_production_and_accounting(self):
        """Étape 3 : Production par ordre de mérite et mise à jour de B_i(t) = B_i(t-1) + Pi_i(t)."""
        for f in self.agents:
            f.produce_and_account()

    def step_4_scrapping(self):
        """Étape 4 : Déclassement one-hoss shay (âge >= T) et obsolescence économique (uc_v > p_i(t))."""
        for f in self.agents:
            f.scrap_vintages(self.current_period)

    def step_5_innovation_and_investment(self):
        """Étape 5 : Innovation DLF, adoption et investissement sous contrainte de budget B_i(t)."""
        for f in self.agents:
            budget_before = f.budget
            k_surv_before = f.k_surv
            f.innovate_and_invest(self.current_period)
            assert f.budget >= -1e-9 and f.budget <= budget_before + 1e-9, (
                f"Assertion de respect du budget B_i(t) échouée pour la firme {f.unique_id}"
            )
            assert f.total_capacity >= k_surv_before - 1e-9

    def _best_practice_technique(self, rule: str, p_ref: float, firms: list) -> tuple:
        """Identifie la technique best-practice de la catégorie 'rule' au prix p_ref."""
        candidates = []
        for f in firms:
            if f.rule == rule:
                candidates.append((f.uc_b, f.kappa_b))
                if f.uc_a is not None:
                    candidates.append((f.uc_a, f.kappa_a))
        if not candidates:
            for f in firms:
                candidates.append((f.uc_b, f.kappa_b))
        return max(candidates, key=lambda tech: (p_ref - tech[0]) / tech[1])

    def step_6_entry_and_exit(self, capital_value_t: float):
        """Étape 6 : Calcul de r_branch, sorties (exit) et entrées (entry) avec conservation sum(f_i)=1."""
        firms = list(self.agents)
        self.last_entries = 0
        self.last_exits_marginalized = []
        self.last_exits_exhausted = []
        if not firms:
            return

        # 1. Taux de profit agrégé de la branche en période t
        total_profits = sum(f.profit for f in firms)
        self.r_branch = (total_profits / capital_value_t) if capital_value_t > 0.0 else 0.0
        p_leader_now = min(f.price for f in firms)

        # Techniques best-practice des firmes présentes en période t
        best_shaikh = self._best_practice_technique("shaikh", p_leader_now, firms)
        best_okishio = self._best_practice_technique("okishio", p_leader_now, firms)

        # 2. Sorties (Capital Exit) : marginalisation post-grâce OU épuisement physique & financier
        survivors = []
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

        # Redistribution proportionnelle des parts cibles f_i aux survivants
        sum_f_surv = sum(f.target_share for f in survivors)
        if sum_f_surv > 0.0:
            for f in survivors:
                f.target_share /= sum_f_surv
        elif survivors:
            for f in survivors:
                f.target_share = 1.0 / len(survivors)

        # 3. Entrées (Capital Entry) via loi de Poisson
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
            for _ in range(n_entry):
                rule = "shaikh" if self.rng.random() < 0.5 else "okishio"
                uc_bp, kappa_bp = best_shaikh if rule == "shaikh" else best_okishio
                v_init = Vintage(
                    uc=uc_bp, kappa=kappa_bp, k_init=self.K_0, k=self.K_0, birth_period=next_t
                )
                FirmAgent(
                    model=self,
                    rule=rule,
                    initial_vintages=[v_init],
                    uc_b=uc_bp,
                    kappa_b=kappa_bp,
                    price=p_leader_now,
                    target_share=entrant_share,
                    budget=self.B_0,
                    u_init=self.u_n,
                    entry_period=next_t
                )

        # Assertion : conservation des parts de marché cibles sum_i f_i(t+1) = 1
        active_firms = list(self.agents)
        if active_firms:
            sum_f_post = sum(f.target_share for f in active_firms)
            assert np.isclose(sum_f_post, 1.0, atol=1e-9), (
                f"Assertion sum(f_i(t+1)) = 1 échouée à l'Étape 6 : sum_f = {sum_f_post}"
            )

    def step(self):
        """Exécute une période complète t -> t+1 dans l'ordre strict des 6 étapes."""
        self.current_period += 1

        # Valeur du capital installé en période t pour le calcul de r_t^branch à l'Étape 6
        capital_value_t = sum(v.kappa * v.k for f in self.agents for v in f.vintages)

        self.step_1_pricing()
        self.step_2_demand_allocation()
        self.step_3_production_and_accounting()
        self.step_4_scrapping()
        self.step_5_innovation_and_investment()
        self.step_6_entry_and_exit(capital_value_t)


def run_unit_tests():
    """Exécute tous les tests unitaires des Étapes 0 à 6 avec vérifications par assertions."""
    print("=== TEST ÉTAPE 0 : Agrégats de la firme ===")
    test_model = mesa.Model(rng=42)
    v1 = Vintage(uc=0.60, kappa=1.0, k_init=60.0, k=60.0, birth_period=0)
    v2 = Vintage(uc=0.40, kappa=1.5, k_init=40.0, k=40.0, birth_period=1)
    firme_test = FirmAgent(
        model=test_model, rule="shaikh", initial_vintages=[v1, v2],
        uc_b=0.40, kappa_b=1.5, price=0.80, target_share=1.0, budget=50.0
    )
    assert np.isclose(firme_test.total_capacity, 100.0)
    assert np.isclose(firme_test.mean_uc, 0.52)
    assert np.isclose(firme_test.mean_kappa, 1.20)
    assert np.isclose(firme_test.normal_profit_rate(), 0.28 / 1.20)
    print("OK - Étape 0 validée.")

    print("\n=== TEST ÉTAPE 1 : Fixation des prix & plancher p_i(t) >= uc_bar_i(t) ===")
    m_floor = RealCompetitionModel(D=85.0, u_n=0.85, gamma=0.05, delta_cost=0.08, eps_0=0.01, seed=42)
    fA = FirmAgent(m_floor, "shaikh", [Vintage(0.40, 1.0, 60, 60, 0)], 0.40, 1.0, price=0.65, target_share=0.6, budget=20, u_init=0.85)
    fB = FirmAgent(m_floor, "okishio", [Vintage(0.68, 1.0, 40, 40, 0)], 0.68, 1.0, price=0.72, target_share=0.4, budget=20, u_init=0.85)
    m_floor.step_1_pricing()
    # eps_A = 0 + 0.08 * (0.65 - 0.40) / 0.65 + 0.01 -> p_A = 0.65 * (1 - eps_A) = 0.6235
    assert np.isclose(fA.price, 0.6235)
    assert np.isclose(fB.price, 0.68)
    assert fA.price >= fA.mean_uc and fB.price >= fB.mean_uc
    print("OK - Étape 1 validée.")

    print("\n=== TEST ÉTAPE 2 : Allocation de la demande & sum(f_i(t)) = 1 ===")
    m2 = RealCompetitionModel(D=100.0, lam=0.8, seed=42)
    fA = FirmAgent(m2, "shaikh", [Vintage(0.30, 1.0, 55, 55, 0)], 0.30, 1.0, price=0.50, target_share=0.50, budget=20)
    fB = FirmAgent(m2, "okishio", [Vintage(0.40, 1.0, 80, 80, 0)], 0.40, 1.0, price=0.625, target_share=0.50, budget=20)
    m2.step_2_demand_allocation()
    assert np.isclose(fA.target_share + fB.target_share, 1.0)
    assert np.isclose(fA.target_share, 0.60)
    assert np.isclose(fB.target_share, 0.40)
    assert np.isclose(fA.sales, 55.0)
    assert np.isclose(fB.sales, 45.0)
    assert np.isclose(fA.u, 1.0)
    print("OK - Étape 2 validée.")

    print("\n=== TEST ÉTAPE 3 : Production par ordre de mérite & comptabilité ===")
    m3 = RealCompetitionModel(seed=42)
    v_vieux = Vintage(uc=0.50, kappa=1.0, k_init=50.0, k=50.0, birth_period=0)
    v_recent = Vintage(uc=0.25, kappa=1.5, k_init=50.0, k=50.0, birth_period=2)
    v_moyen = Vintage(uc=0.40, kappa=1.2, k_init=30.0, k=30.0, birth_period=1)
    firme = FirmAgent(
        model=m3, rule="shaikh", initial_vintages=[v_vieux, v_recent, v_moyen],
        uc_b=0.25, kappa_b=1.5, price=0.60, target_share=1.0, budget=100.0
    )
    firme.sales = 65.0
    m3.step_3_production_and_accounting()
    assert np.isclose(firme.profit, 20.5)
    assert np.isclose(firme.budget, 120.5)
    print("OK - Étape 3 validée.")

    print("\n=== TEST ÉTAPE 4 : Déclassement physique (a_v(t) >= T) et économique (uc_v > p_i(t)) ===")
    m4 = RealCompetitionModel(T=10, seed=42)
    m4.current_period = 10
    v_trop_vieux = Vintage(uc=0.30, kappa=1.0, k_init=40.0, k=40.0, birth_period=1)
    v_non_rentable = Vintage(uc=0.55, kappa=1.0, k_init=30.0, k=30.0, birth_period=8)
    v_survivant = Vintage(uc=0.35, kappa=1.2, k_init=50.0, k=50.0, birth_period=7)
    firme4 = FirmAgent(
        model=m4, rule="shaikh", initial_vintages=[v_trop_vieux, v_non_rentable, v_survivant],
        uc_b=0.35, kappa_b=1.2, price=0.50, target_share=1.0, budget=100.0
    )
    m4.step_4_scrapping()
    assert len(firme4.vintages) == 1
    assert firme4.vintages == [v_survivant]
    assert np.isclose(firme4.k_surv, 50.0)
    print("OK - Étape 4 validée.")

    print("\n=== TEST ÉTAPE 5 : Innovation, adoption Shaikh vs Okishio, Delta k_i(t) & budget B_i(t) ===")
    m5 = RealCompetitionModel(u_n=0.85, gamma_thresh=0.75, seed=42)
    m5.current_period = 1
    f_shaikh = FirmAgent(m5, "shaikh", [Vintage(0.50, 1.0, 50.0, 50.0, 0)], 0.50, 1.0, price=0.70, target_share=0.5, budget=40.0)
    f_okishio = FirmAgent(m5, "okishio", [Vintage(0.50, 1.0, 50.0, 50.0, 0)], 0.50, 1.0, price=0.70, target_share=0.5, budget=40.0)
    f_shaikh.sales = 59.5   # K_desired = 59.5 / 0.85 = 70.0 -> I_req = 20.0
    f_okishio.sales = 59.5
    choc_marx = (0.05, -0.15)
    f_shaikh.innovate_and_invest(current_t=1, forced_draw=choc_marx)
    f_okishio.innovate_and_invest(current_t=1, forced_draw=choc_marx)
    assert np.isclose(f_okishio.mean_uc, 0.50)
    assert f_shaikh.mean_uc < 0.50
    assert f_shaikh.uc_a is not None

    m_inv = RealCompetitionModel(u_n=0.85, gamma_thresh=0.75, seed=42)
    # Cas 1 : Budget contraint (B = 60.0, I_req = 80.0, kappa_act = 1.5 -> dk_i = min(80, 60/1.5) = 40.0)
    f_contraint = FirmAgent(m_inv, "shaikh", [Vintage(0.50, 1.0, 20.0, 20.0, 0)], 0.50, 1.0, price=0.70, target_share=1.0, budget=60.0)
    f_contraint.uc_a, f_contraint.kappa_a = 0.40, 1.5
    f_contraint.sales = 85.0
    f_contraint.k_surv = 20.0
    f_contraint.innovate_and_invest(current_t=1)
    k_a_contraint = sum(v.k for v in f_contraint.vintages if np.isclose(v.uc, 0.40))
    assert np.isclose(k_a_contraint, 40.0)
    assert np.isclose(f_contraint.total_capacity, 60.0)
    assert np.isclose(f_contraint.budget, 0.0)
    assert f_contraint.uc_a == 0.40  # Part = 40 / 60 = 66.7% < 75% -> reste en consolidation

    # Cas 2 : Budget confortable (B = 135.0, I_req = 80.0, kappa_act = 1.5 -> dk_i = min(80, 90) = 80.0)
    f_riche = FirmAgent(m_inv, "shaikh", [Vintage(0.50, 1.0, 20.0, 20.0, 0)], 0.50, 1.0, price=0.70, target_share=1.0, budget=135.0)
    f_riche.uc_a, f_riche.kappa_a = 0.40, 1.5
    f_riche.sales = 85.0
    f_riche.k_surv = 20.0
    f_riche.innovate_and_invest(current_t=1)
    assert np.isclose(f_riche.total_capacity, 100.0)
    assert np.isclose(f_riche.budget, 15.0)  # Reliquat conservé : 135 - 80 * 1.5 = 15.0
    assert np.isclose(f_riche.uc_b, 0.40)    # Part = 80 / 100 = 80% >= 75% -> promue benchmark !
    assert f_riche.uc_a is None
    print("OK - Étape 5 validée.")

    print("\n=== TEST ÉTAPE 6 : Entrées, sorties et conservation sum(f_i(t)) = 1 ===")
    m6 = RealCompetitionModel(r_star=0.10, psi=20.0, f_0=0.05, K_0=10.0, B_0=15.0, s_min=0.01, tau_grace=20, seed=42)
    m6.current_period = 25
    f1 = FirmAgent(m6, "shaikh", [Vintage(0.30, 1.0, 50.0, 50.0, 0)], 0.30, 1.0, price=0.60, target_share=0.60, budget=50.0, entry_period=0)
    f1.profit = 15.0
    f1.realized_share = 0.65
    f2 = FirmAgent(m6, "okishio", [Vintage(0.35, 1.0, 40.0, 40.0, 0)], 0.35, 1.0, price=0.62, target_share=0.35, budget=40.0, entry_period=0)
    f2.profit = 8.0
    f2.realized_share = 0.345
    f3_sortante = FirmAgent(m6, "okishio", [Vintage(0.55, 1.0, 10.0, 10.0, 0)], 0.55, 1.0, price=0.65, target_share=0.05, budget=5.0, entry_period=0)
    f3_sortante.profit = 0.0
    f3_sortante.realized_share = 0.005
    m6.step_6_entry_and_exit(capital_value_t=100.0)
    firms_after = list(m6.agents)
    assert f3_sortante not in firms_after
    assert len(firms_after) > 2
    assert np.isclose(sum(f.target_share for f in firms_after), 1.0)
    print("OK - Étape 6 validée.")


def run_detailed_simulation(
    n_steps: int = 150,
    innovation_regime: str = "marx",
    n_firms_init: int = 10,
    D: float = 85.0,
    r_star: float = 0.12,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exécute une simulation de n_steps périodes et collecte à chaque période t :
    - df_firms : panel complet des variables endogènes (Table 2) pour chaque firme active i in N_t
      (mesurées après l'Étape 3 sur le portefeuille actif C_i(t), avec statut de sortie en Étape 6).
    - df_agg   : séries temporelles agrégées de la branche (S(t), D(t), p^leader(t), r_t^branch, démographie).
    """
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
        seed=seed
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
            entry_period=0
        )

    firm_records = []
    agg_records = []

    for t in range(1, n_steps + 1):
        model.current_period += 1
        firms_t = list(model.agents)
        if not firms_t:
            break

        # Grandeurs d'entrée de période t (avant Étape 1)
        S_t = sum(f.total_capacity for f in firms_t)
        p_leader_in = min(f.price for f in firms_t)
        is_shortage = S_t < model.D
        capital_value_t = sum(v.kappa * v.k for f in firms_t for v in f.vintages)

        # Étapes 1, 2 et 3 sur la population active N_t et le portefeuille C_i(t)
        model.step_1_pricing()
        model.step_2_demand_allocation()
        model.step_3_production_and_accounting()

        # Enregistrement des variables endogènes de période t (Table 2) pour chaque firme active i in N_t
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
        p_avg_t = sum(f.price * f.sales for f in firms_t) / total_sales_t if total_sales_t > 0 else p_min_t

        shaikh_t = [f for f in firms_t if f.rule == "shaikh"]
        okishio_t = [f for f in firms_t if f.rule == "okishio"]

        K_shaikh = sum(f.total_capacity for f in shaikh_t)
        K_okishio = sum(f.total_capacity for f in okishio_t)

        uc_shaikh = (sum(f.mean_uc * f.total_capacity for f in shaikh_t) / K_shaikh) if K_shaikh > 0 else np.nan
        uc_okishio = (sum(f.mean_uc * f.total_capacity for f in okishio_t) / K_okishio) if K_okishio > 0 else np.nan

        r_bar_shaikh = (sum(f.normal_profit_rate() * f.total_capacity for f in shaikh_t) / K_shaikh) if K_shaikh > 0 else np.nan
        r_bar_okishio = (sum(f.normal_profit_rate() * f.total_capacity for f in okishio_t) / K_okishio) if K_okishio > 0 else np.nan

        u_shaikh = (sum(f.sales for f in shaikh_t) / K_shaikh) if K_shaikh > 0 else np.nan
        u_okishio = (sum(f.sales for f in okishio_t) / K_okishio) if K_okishio > 0 else np.nan

        share_shaikh = sum(f.realized_share for f in shaikh_t)
        share_okishio = sum(f.realized_share for f in okishio_t)

        # Étapes 4, 5 et 6 (déclassement, investissement, calcul de r_t^branch, sorties et entrées)
        model.step_4_scrapping()
        model.step_5_innovation_and_investment()
        model.step_6_entry_and_exit(capital_value_t)

        # Marquage des firmes sorties à la fin de la période t
        for fid in model.last_exits_marginalized:
            if fid in step_records:
                step_records[fid]["exited"] = True
                step_records[fid]["exit_reason"] = "marginalized"
        for fid in model.last_exits_exhausted:
            if fid in step_records:
                step_records[fid]["exited"] = True
                step_records[fid]["exit_reason"] = "exhausted"

        firm_records.extend(step_records.values())

        agg_records.append({
            "t": t,
            "n_firms": len(firms_t),
            "n_shaikh": len(shaikh_t),
            "n_okishio": len(okishio_t),
            "S_t": S_t,
            "Q_max_t": total_q_max_t,
            "Q_t": total_sales_t,
            "D_t": model.D,
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
            "n_exits_total": len(model.last_exits_marginalized) + len(model.last_exits_exhausted),
        })

    return pd.DataFrame(firm_records), pd.DataFrame(agg_records)


def _shade_shortage_regimes(ax, df_agg: pd.DataFrame, label_first: bool = True):
    """Ajoute un ombrage vertical sur les périodes de pénurie (S(t) < D(t))."""
    shortage_mask = df_agg["is_shortage"].values
    t_vals = df_agg["t"].values
    in_span = False
    start_t = None
    first = True
    for idx, is_short in enumerate(shortage_mask):
        if is_short and not in_span:
            in_span = True
            start_t = t_vals[idx] - 0.5
        elif not is_short and in_span:
            in_span = False
            end_t = t_vals[idx - 1] + 0.5
            lbl = "Régime de pénurie ($S(t) < D(t)$)" if (first and label_first) else None
            ax.axvspan(start_t, end_t, color="#FFE5B4", alpha=0.42, zorder=0, label=lbl)
            first = False
    if in_span:
        end_t = t_vals[-1] + 0.5
        lbl = "Régime de pénurie ($S(t) < D(t)$)" if (first and label_first) else None
        ax.axvspan(start_t, end_t, color="#FFE5B4", alpha=0.42, zorder=0, label=lbl)


def generate_diagnostic_figures(output_dir: str = "figures", n_steps: int = 150, seed: int = 42):
    """Génère les 4 graphiques diagnostiques haute résolution (300 dpi) dans output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    df_firms, df_agg = run_detailed_simulation(
        n_steps=n_steps, innovation_regime="marx", D=85.0, r_star=0.12, seed=seed
    )

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 8.8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })

    color_shaikh = "#C8102E"   # Rouge carmin (Shaikh)
    color_okishio = "#1F5A9E"  # Bleu acier (Okishio)

    # =========================================================================
    # FIGURE (a) : Formation des prix individuels p_i(t), planchers uc_bar_i(t)
    #              et prix leader p^leader(t)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.1], "hspace": 0.14}
    )
    _shade_shortage_regimes(ax1, df_agg, label_first=True)
    _shade_shortage_regimes(ax2, df_agg, label_first=False)

    for fid, grp in df_firms.groupby("firm_id"):
        grp = grp.sort_values("t")
        rule = grp["rule"].iloc[0]
        c = color_shaikh if rule == "shaikh" else color_okishio
        ls_price = "-" if rule == "shaikh" else "--"

        # Courbe fine semi-transparente du prix p_i(t), interrompue à la sortie
        ax1.plot(grp["t"], grp["p_i"], color=c, linestyle=ls_price, linewidth=1.0, alpha=0.45, zorder=2)
        # Courbe fine pointillée du plancher de coût unitaire uc_bar_i(t)
        ax1.plot(grp["t"], grp["uc_bar_i"], color=c, linestyle=":", linewidth=0.8, alpha=0.28, zorder=1)

        # Marge unitaire p_i(t) - uc_bar_i(t) >= 0 sur le panneau inférieur
        margin_i = grp["p_i"] - grp["uc_bar_i"]
        ax2.plot(grp["t"], margin_i, color=c, linestyle=ls_price, linewidth=1.0, alpha=0.45, zorder=2)

        # Marqueur de sortie si la firme quitte la branche
        if grp["exited"].iloc[-1]:
            ax1.scatter(grp["t"].iloc[-1], grp["p_i"].iloc[-1], color=c, marker="x", s=32, alpha=0.8, zorder=4)
            ax2.scatter(grp["t"].iloc[-1], margin_i.iloc[-1], color=c, marker="x", s=28, alpha=0.8, zorder=4)

    # Indicateurs agrégés et de référence en gras
    ax1.plot(df_agg["t"], df_agg["p_leader_in"], color="#111111", linewidth=2.4,
             label="Prix leader entrant $p^{\\mathrm{leader}}(t) = \\min_j p_j(t-1)$", zorder=5)
    ax1.plot(df_agg["t"], df_agg["p_avg_t"], color="#444444", linestyle="-.", linewidth=2.0,
             label="Prix moyen pondéré de la branche $\\bar{p}(t)$", zorder=5)
    ax1.plot(df_agg["t"], df_agg["uc_shaikh"], color=color_shaikh, linewidth=2.2, linestyle=":",
             label="Plancher moyen Shaikh $\\overline{uc}_{\\mathrm{Shaikh}}(t)$", zorder=5)
    ax1.plot(df_agg["t"], df_agg["uc_okishio"], color=color_okishio, linewidth=2.2, linestyle=":",
             label="Plancher moyen Okishio $\\overline{uc}_{\\mathrm{Okishio}}(t)$", zorder=5)

    handles_a1, labels_a1 = ax1.get_legend_handles_labels()
    custom_a1 = [
        Line2D([0], [0], color=color_shaikh, lw=1.2, ls="-", alpha=0.65, label="Trajectoires $p_i(t)$ — Firmes Shaikh"),
        Line2D([0], [0], color=color_okishio, lw=1.2, ls="--", alpha=0.65, label="Trajectoires $p_i(t)$ — Firmes Okishio"),
        Line2D([0], [0], color="gray", marker="x", ls="None", ms=6, label="Sortie de firme (interruption de trajectoire)"),
    ]
    ax1.set_title("(a) Formation des prix individuels $p_i(t)$, planchers de coûts $\\overline{uc}_i(t)$ et prix leader $p^{\\mathrm{leader}}(t)$",
                  fontweight="bold", pad=10)
    ax1.set_ylabel("Prix $p_i(t)$ et coût unitaire $\\overline{uc}_i(t)$")
    ax1.legend(handles=custom_a1 + handles_a1, loc="upper right", ncol=2, framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panneau inférieur : Marge unitaire p_i(t) - uc_bar_i(t) (vérification visuelle de p_i >= uc_bar_i)
    ax2.axhline(0.0, color="#B22222", linestyle="--", linewidth=1.8,
                label="Borne inférieure stricte : $p_i(t) - \\overline{uc}_i(t) = 0$ (plancher atteint)")
    margin_shaikh = df_agg["p_avg_t"] - df_agg["uc_shaikh"]
    margin_okishio = df_agg["p_avg_t"] - df_agg["uc_okishio"]
    ax2.plot(df_agg["t"], margin_shaikh, color=color_shaikh, linewidth=2.0,
             label="Écart moyen $\\bar{p}(t) - \\overline{uc}_{\\mathrm{Shaikh}}(t)$", zorder=4)
    ax2.plot(df_agg["t"], margin_okishio, color=color_okishio, linewidth=2.0, linestyle="--",
             label="Écart moyen $\\bar{p}(t) - \\overline{uc}_{\\mathrm{Okishio}}(t)$", zorder=4)
    ax2.set_xlabel("Période $t$")
    ax2.set_ylabel("Marge $p_i(t) - \\overline{uc}_i(t)$")
    ax2.set_xlim(1, n_steps)
    ax2.legend(loc="upper right", ncol=2, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.3)

    path_a = os.path.join(output_dir, "fig_a_prices_and_cost_floors.png")
    fig.savefig(path_a, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE (b) : Dispersion et gravitation des taux de profit normaux r_bar_i(t)
    #              autour de r_branch(t) et du taux général r*
    # =========================================================================
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    _shade_shortage_regimes(ax, df_agg, label_first=True)

    for fid, grp in df_firms.groupby("firm_id"):
        grp = grp.sort_values("t")
        rule = grp["rule"].iloc[0]
        c = color_shaikh if rule == "shaikh" else color_okishio
        ls = "-" if rule == "shaikh" else "--"

        ax.plot(grp["t"], grp["r_bar_i"] * 100.0, color=c, linestyle=ls, linewidth=1.05, alpha=0.42, zorder=2)
        if grp["entry_period"].iloc[0] > 0:
            ax.scatter(grp["t"].iloc[0], grp["r_bar_i"].iloc[0] * 100.0, color=c, marker="o", s=24, alpha=0.75, zorder=3)
        if grp["exited"].iloc[-1]:
            ax.scatter(grp["t"].iloc[-1], grp["r_bar_i"].iloc[-1] * 100.0, color=c, marker="x", s=34, alpha=0.85, zorder=4)

    ax.plot(df_agg["t"], df_agg["r_branch"] * 100.0, color="#111111", linewidth=2.5,
            label="Taux de profit réalisé de la branche $r_t^{\\mathrm{branch}}$", zorder=6)
    ax.plot(df_agg["t"], df_agg["r_bar_shaikh"] * 100.0, color=color_shaikh, linewidth=2.1,
            label="Taux de profit normal moyen $\\bar{r}_{\\mathrm{Shaikh}}(t)$", zorder=5)
    ax.plot(df_agg["t"], df_agg["r_bar_okishio"] * 100.0, color=color_okishio, linewidth=2.1, linestyle="--",
            label="Taux de profit normal moyen $\\bar{r}_{\\mathrm{Okishio}}(t)$", zorder=5)
    ax.axhline(df_agg["r_star"].iloc[0] * 100.0, color="#2E8B57", linestyle="-.", linewidth=2.2,
               label="Taux de profit général $r^* = 12\\%$ (seuil d'entrée)", zorder=5)
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.2, label="Marge nulle ($\\bar{r}_i = 0\\%$)", zorder=1)

    handles_b, _ = ax.get_legend_handles_labels()
    custom_b = [
        Line2D([0], [0], color=color_shaikh, lw=1.2, ls="-", alpha=0.65, label="Trajectoires $\\bar{r}_i(t)$ — Firmes Shaikh"),
        Line2D([0], [0], color=color_okishio, lw=1.2, ls="--", alpha=0.65, label="Trajectoires $\\bar{r}_i(t)$ — Firmes Okishio"),
        Line2D([0], [0], color="gray", marker="o", ls="None", ms=5, label="Entrée d'une firme ($t_{i,0}$)"),
        Line2D([0], [0], color="gray", marker="x", ls="None", ms=6, label="Sortie d'une firme"),
    ]
    ax.set_title("(b) Dispersion et gravitation des taux de profit normaux individuels $\\bar{r}_i(t)$ autour de $r_t^{\\mathrm{branch}}$ et $r^*$",
                 fontweight="bold", pad=10)
    ax.set_xlabel("Période $t$")
    ax.set_ylabel("Taux de profit (%)")
    ax.set_xlim(1, n_steps)
    ax.legend(handles=custom_b + handles_b, loc="upper right", ncol=2, framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.3)

    path_b = os.path.join(output_dir, "fig_b_profit_rates_gravitation.png")
    fig.savefig(path_b, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE (c) : Parts de marché réalisées individuelles s_i(t) et démographie
    #              des firmes (entrées Poissoniennes et sorties)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8.8), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.1], "hspace": 0.14}
    )

    for fid, grp in df_firms.groupby("firm_id"):
        grp = grp.sort_values("t")
        rule = grp["rule"].iloc[0]
        c = color_shaikh if rule == "shaikh" else color_okishio
        ls = "-" if rule == "shaikh" else "--"

        ax1.plot(grp["t"], grp["s_i"] * 100.0, color=c, linestyle=ls, linewidth=1.05, alpha=0.45, zorder=2)
        if grp["entry_period"].iloc[0] > 0:
            ax1.scatter(grp["t"].iloc[0], grp["s_i"].iloc[0] * 100.0, color=c, marker="^", s=30, alpha=0.85, zorder=4)
        if grp["exited"].iloc[-1]:
            marker_exit = "x" if grp["exit_reason"].iloc[-1] == "marginalized" else "D"
            ax1.scatter(grp["t"].iloc[-1], grp["s_i"].iloc[-1] * 100.0, color="#111111",
                        marker=marker_exit, s=38, alpha=0.9, zorder=5)

    ax1.plot(df_agg["t"], df_agg["share_shaikh"] * 100.0, color=color_shaikh, linewidth=2.5,
             label="Part de marché agrégée Shaikh $\\sum_{i \\in \\mathrm{Shaikh}} s_i(t)$", zorder=6)
    ax1.plot(df_agg["t"], df_agg["share_okishio"] * 100.0, color=color_okishio, linewidth=2.5, linestyle="--",
             label="Part de marché agrégée Okishio $\\sum_{i \\in \\mathrm{Okishio}} s_i(t)$", zorder=6)
    ax1.axhline(1.0, color="#B22222", linestyle=":", linewidth=1.6,
                label="Seuil de viabilité $s_{\\min} = 1\\%$", zorder=3)
    ax1.axvline(20, color="#555555", linestyle="-.", linewidth=1.4,
                label="Fin de période de grâce initiale ($\\tau_{\\mathrm{grace}} = 20$)", zorder=3)

    handles_c1, _ = ax1.get_legend_handles_labels()
    custom_c1 = [
        Line2D([0], [0], color=color_shaikh, lw=1.2, ls="-", alpha=0.65, label="Trajectoires $s_i(t)$ — Firmes Shaikh"),
        Line2D([0], [0], color=color_okishio, lw=1.2, ls="--", alpha=0.65, label="Trajectoires $s_i(t)$ — Firmes Okishio"),
        Line2D([0], [0], color="gray", marker="^", ls="None", ms=6, label="Entrée Poissonienne ($f_0 = 2\\%$)"),
        Line2D([0], [0], color="#111111", marker="x", ls="None", ms=6, label="Sortie par marginalisation ($s_i < s_{\\min}$)"),
        Line2D([0], [0], color="#111111", marker="D", ls="None", ms=5, label="Sortie par épuisement ($K_i=0, B_i < B_{\\min,i}$)"),
    ]
    ax1.set_title("(c) Évolution des parts de marché réalisées $s_i(t)$ et démographie des firmes (entrées et sorties)",
                  fontweight="bold", pad=10)
    ax1.set_ylabel("Part de marché réalisée (%)")
    ax1.set_ylim(-1.5, 103.0)
    ax1.legend(handles=custom_c1 + handles_c1, loc="center right", ncol=2, framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panneau inférieur : Démographie de la branche (effectifs actifs et flux d'entrées/sorties)
    ax2.plot(df_agg["t"], df_agg["n_firms"], color="#111111", linewidth=2.2, label="Population active totale $|\\mathcal{N}_t|$")
    ax2.plot(df_agg["t"], df_agg["n_shaikh"], color=color_shaikh, linewidth=1.8, label="Firmes Shaikh actives")
    ax2.plot(df_agg["t"], df_agg["n_okishio"], color=color_okishio, linewidth=1.8, linestyle="--", label="Firmes Okishio actives")
    ax2.bar(df_agg["t"], df_agg["n_entries"], width=0.8, color="#2E8B57", alpha=0.65, label="Entrées $N_t^{\\mathrm{entry}}$ (flux)")
    ax2.bar(df_agg["t"], -df_agg["n_exits_total"], width=0.8, color="#D9534F", alpha=0.65, label="Sorties $-N_t^{\\mathrm{exit}}$ (flux)")
    ax2.axhline(0, color="#666666", linewidth=0.8)
    ax2.set_xlabel("Période $t$")
    ax2.set_ylabel("Nombre de firmes / Flux")
    ax2.set_xlim(1, n_steps)
    ax2.legend(loc="upper right", ncol=3, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.3)

    path_c = os.path.join(output_dir, "fig_c_market_shares_and_demography.png")
    fig.savefig(path_c, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE (d) : Taux d'utilisation individuels u_i(t) face à la cible u_n
    #              et régimes d'offre/demande (S(t) >= D(t) vs S(t) < D(t))
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8.8), sharex=True,
        gridspec_kw={"height_ratios": [1.85, 1.25], "hspace": 0.14}
    )
    _shade_shortage_regimes(ax1, df_agg, label_first=True)
    _shade_shortage_regimes(ax2, df_agg, label_first=False)

    for fid, grp in df_firms.groupby("firm_id"):
        grp = grp.sort_values("t")
        rule = grp["rule"].iloc[0]
        c = color_shaikh if rule == "shaikh" else color_okishio
        ls = "-" if rule == "shaikh" else "--"

        ax1.plot(grp["t"], grp["u_i"] * 100.0, color=c, linestyle=ls, linewidth=1.0, alpha=0.42, zorder=2)
        if grp["exited"].iloc[-1]:
            ax1.scatter(grp["t"].iloc[-1], grp["u_i"].iloc[-1] * 100.0, color=c, marker="x", s=32, alpha=0.85, zorder=4)

    ax1.plot(df_agg["t"], df_agg["u_industry"] * 100.0, color="#111111", linewidth=2.4,
             label="Utilisation agrégée de la branche $u_{\\mathrm{branche}}(t) = Q(t)/S(t)$", zorder=6)
    ax1.plot(df_agg["t"], df_agg["u_shaikh"] * 100.0, color=color_shaikh, linewidth=2.0,
             label="Utilisation moyenne Shaikh $\\bar{u}_{\\mathrm{Shaikh}}(t)$", zorder=5)
    ax1.plot(df_agg["t"], df_agg["u_okishio"] * 100.0, color=color_okishio, linewidth=2.0, linestyle="--",
             label="Utilisation moyenne Okishio $\\bar{u}_{\\mathrm{Okishio}}(t)$", zorder=5)
    ax1.axhline(85.0, color="#2E8B57", linestyle="-.", linewidth=2.1,
                label="Cible d'utilisation normale $u_n = 85\\%$", zorder=5)
    ax1.axhline(100.0, color="#B22222", linestyle=":", linewidth=1.5,
                label="Plafond de pleine capacité ($u_{\\max} = 100\\%$)", zorder=4)

    handles_d1, _ = ax1.get_legend_handles_labels()
    custom_d1 = [
        Line2D([0], [0], color=color_shaikh, lw=1.2, ls="-", alpha=0.65, label="Trajectoires $u_i(t)$ — Firmes Shaikh"),
        Line2D([0], [0], color=color_okishio, lw=1.2, ls="--", alpha=0.65, label="Trajectoires $u_i(t)$ — Firmes Okishio"),
        Line2D([0], [0], color="gray", marker="x", ls="None", ms=6, label="Sortie de firme"),
    ]
    ax1.set_title("(d) Taux d'utilisation individuels $u_i(t)$ face à la cible normale $u_n$ et régimes d'offre/demande ($S(t) \\geq D(t)$ vs $S(t) < D(t)$)",
                  fontweight="bold", pad=10)
    ax1.set_ylabel("Taux d'utilisation $u_i(t)$ (%)")
    ax1.set_ylim(-3.0, 108.0)
    ax1.legend(handles=custom_d1 + handles_d1, loc="lower left", ncol=2, framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panneau inférieur : Capacité agrégée S(t), offre rentable Q^max(t), ventes Q(t) et demande D(t)
    ax2.plot(df_agg["t"], df_agg["S_t"], color="#111111", linewidth=2.3,
             label="Capacité installée totale $S(t) = \\sum_i K_i(t)$", zorder=5)
    ax2.plot(df_agg["t"], df_agg["Q_max_t"], color="#1F5A9E", linestyle="--", linewidth=1.9,
             label="Offre rentable disponible $\\sum_i q_i^{\\max}(t)$", zorder=4)
    ax2.plot(df_agg["t"], df_agg["Q_t"], color="#C8102E", linestyle=":", linewidth=2.0,
             label="Ventes totales réalisées $Q(t) = \\sum_i q_i(t)$", zorder=5)
    ax2.axhline(df_agg["D_t"].iloc[0], color="#D9534F", linestyle="-.", linewidth=2.0,
                label="Demande agrégée $D(t) = 85$ (seuil $S(t) = D(t)$)", zorder=3)
    ax2.axhline(df_agg["D_t"].iloc[0] / 0.85, color="#2E8B57", linestyle=":", linewidth=1.6,
                label="Capacité normale cible $D(t)/u_n = 100$", zorder=3)
    ax2.set_xlabel("Période $t$")
    ax2.set_ylabel("Volume de capacité / Demande")
    ax2.set_xlim(1, n_steps)
    ax2.legend(loc="upper right", ncol=2, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.3)

    path_d = os.path.join(output_dir, "fig_d_capacity_utilization_and_regimes.png")
    fig.savefig(path_d, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== DIAGNOSTIC DE LA SIMULATION (150 périodes, seed={seed}, régime Marx) ===")
    print(f"- Nombre total de firmes distinctes ayant opéré : {df_firms['firm_id'].nunique()}")
    print(f"- Entrées totales (Poisson) : {df_agg['n_entries'].sum()}")
    print(f"- Sorties totales : {df_agg['n_exits_total'].sum()} "
          f"(marginalisation s_i < s_min : {df_agg['n_exits_marg'].sum()}, "
          f"épuisement physique/financier : {df_agg['n_exits_exh'].sum()})")
    print(f"- Périodes en régime normal (S >= D) : {(~df_agg['is_shortage']).sum()} / {n_steps}")
    print(f"- Périodes en régime de pénurie (S < D) : {df_agg['is_shortage'].sum()} / {n_steps}")
    print(f"- Marge minimale observée min(p_i(t) - uc_bar_i(t)) : {(df_firms['p_i'] - df_firms['uc_bar_i']).min():.6f} >= 0")
    print(f"- Part de marché finale (t={n_steps}) : Shaikh = {df_agg['share_shaikh'].iloc[-1]:.2%}, "
          f"Okishio = {df_agg['share_okishio'].iloc[-1]:.2%}")
    print(f"- Figures sauvegardées (300 dpi) :\n  1. {path_a}\n  2. {path_b}\n  3. {path_c}\n  4. {path_d}")


if __name__ == "__main__":
    run_unit_tests()
    generate_diagnostic_figures(output_dir="figures", n_steps=150, seed=42)
    print("\n✅ Toutes les étapes (0 à 6), assertions et générations de graphiques ont réussi !")
