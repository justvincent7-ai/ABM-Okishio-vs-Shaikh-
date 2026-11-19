"""
Dumenil-Levy-Foley (DLF) technical change draws, prime-cost technique mapping,
adoption criteria (Shaikh cost-cutting vs Okishio profit-rate maximizing),
and entrant best-practice selection.
"""

from __future__ import annotations

from typing import Iterable, Optional
import numpy as np


def draw_dlf_innovation(
    rng: np.random.Generator,
    regime: str,
    R: float,
    marx_a: float,
    marx_b: float,
    marx_theta: float,
    marx_mu_ucs: float,
    marx_mu_kappa: float,
) -> tuple[float, float]:
    """Sample a factor-saving innovation pair (d_ucs, d_kappa) uniformly from I_neutral or I_Marx."""
    if regime == "neutral":
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = R * np.sqrt(rng.uniform(0.0, 1.0))
        return float(radius * np.cos(angle)), float(radius * np.sin(angle))

    c, s = np.cos(marx_theta), np.sin(marx_theta)
    bound = max(marx_a, marx_b)
    while True:
        dx = rng.uniform(-bound, bound)
        dy = rng.uniform(-bound, bound)
        d_uc = marx_mu_ucs + dx
        d_kappa = marx_mu_kappa + dy
        u_rot = (d_uc - marx_mu_ucs) * c + (d_kappa - marx_mu_kappa) * s
        v_rot = -(d_uc - marx_mu_ucs) * s + (d_kappa - marx_mu_kappa) * c
        if (u_rot / marx_a) ** 2 + (v_rot / marx_b) ** 2 <= 1.0:
            return float(d_uc), float(d_kappa)


def apply_dlf_transformation(
    uc_b: float,
    kappa_b: float,
    d_ucs: float,
    d_kappa: float,
    T: int,
    u_n: float,
) -> tuple[float, float, float]:
    """Compute candidate technique (ucs_plus, uc_plus, kappa_plus) from benchmark (uc_b, kappa_b)."""
    dep_denom = T * u_n
    ucs_b = uc_b - kappa_b / dep_denom
    if ucs_b <= 0.0:
        raise AssertionError(f"Non-positive baseline prime cost: ucs_b={ucs_b}")
    if (1.0 + d_ucs) <= 0.0 or (1.0 + d_kappa) <= 0.0:
        raise ValueError(f"Invalid innovation factor: d_ucs={d_ucs}, d_kappa={d_kappa}")
    ucs_plus = ucs_b / (1.0 + d_ucs)
    kappa_plus = kappa_b / (1.0 + d_kappa)
    uc_plus = ucs_plus + kappa_plus / dep_denom
    return ucs_plus, uc_plus, kappa_plus


def evaluate_adoption_criterion(
    rule: str,
    uc_b: float,
    kappa_b: float,
    uc_plus: float,
    kappa_plus: float,
    *,
    u_n: float,
    r_star: float,
    current_price: float,
    p_ref_okishio: Optional[float | str] = "normal_price",
) -> bool:
    """Return True if technique (uc_plus, kappa_plus) satisfies the firm's viability criterion."""
    if rule == "okishio":
        if p_ref_okishio == "normal_price":
            p_eval = uc_b + (r_star / u_n) * kappa_b
        elif p_ref_okishio is not None:
            p_eval = float(p_ref_okishio)
        else:
            p_eval = current_price
        r_b = u_n * (p_eval - uc_b) / kappa_b
        r_plus = u_n * (p_eval - uc_plus) / kappa_plus
        return bool(r_plus > r_b)

    if rule == "shaikh":
        return bool(uc_plus < uc_b)

    return False


def select_best_practice(
    rule: str, p_ref: float, firms: Iterable
) -> tuple[float, float]:
    """Identify the highest normal-return technique (uc, kappa) within `rule` at `p_ref`."""
    firms_list = list(firms)
    candidates: list[tuple[float, float]] = []
    for f in firms_list:
        if f.rule == rule:
            candidates.append((f.uc_b, f.kappa_b))
            if f.uc_a is not None:
                candidates.append((f.uc_a, f.kappa_a))
    if not candidates:
        for f in firms_list:
            candidates.append((f.uc_b, f.kappa_b))
    return max(candidates, key=lambda tech: (p_ref - tech[0]) / tech[1])
