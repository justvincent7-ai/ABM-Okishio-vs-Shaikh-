"""
Centralized baseline calibration (Table 3 / Appendix A), Morris sensitivity bounds,
paired firm initialization, and provenance utilities for the Real Competition ABM.
"""

from __future__ import annotations

import hashlib
import math
import platform
from importlib.metadata import version as pkg_version
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.model import RealCompetitionModel


BASELINE_RULES = MappingProxyType(
    {
        "enable_monopoly_pricing": False,
        "evolutionary_entry": False,
        "enable_low_profit_exit": False,
    }
)

# Baseline parameterization (annual periodicity: 1 step = 1 year, horizon T = 120)
BASELINE_PARAMS: dict[str, Any] = {
    # Demand (Step 0), normal capacity utilization, and capital service life
    "D": 82.0,
    "g_D": 0.015,
    "eta": 2.0,
    "alpha_p": 0.15,
    "p_0": 1.0,
    "u_n": 0.82,
    "T": 16,
    # Pricing (Step 1) and quasi-replicator demand reallocation (Step 2)
    "gamma": 1.30,
    "delta_cost": 0.030,
    "eps_0": 0.0,
    "mu_z": 0.07,
    "sigma_z": 0.025,
    "lam": 0.70,
    # Dumenil-Levy internal accumulation (Step 5)
    "beta_r": 2.6,
    "beta_u": 0.14,
    # DLF Innovation Sets: Neutral disk vs Marx-biased rotated ellipse (Step 5)
    "R": 0.01,
    "marx_a": 0.020,
    "marx_b": 0.005,
    "marx_theta": -0.9310265801,  # atan2(-0.043, 0.032)
    "marx_mu_uc": 0.032,           # Prime-cost saving drift mu_ucs (alias: marx_mu_ucs)
    "marx_mu_kappa": -0.043,       # Capital-deepening drift mu_kappa
    "gamma_thresh": 0.25,
    "p_ref_okishio": "normal_price",
    # Entry and exit (Step 6, entrant endowments scaled proportionally by D(t)/D_0)
    "r_star": 0.04,
    "psi": 20.0,
    "f_0": 0.06,
    "K_0": 5.0,
    "B_0": 30.0,
    "s_min": 0.060,
    "tau_grace": 5,
    # Disabled structural extensions & passive compatibility parameters
    "enable_monopoly_pricing": False,
    "evolutionary_entry": False,
    "enable_low_profit_exit": False,
    "interest_rate": 0.0,
    "tau_low_profit": 10,
}

# Symmetric paired firm initialization (5 age cohorts x 2 rules = 10 initial firms)
INITIAL_FIRM_PARAMS: dict[str, Any] = {
    "n_pairs": 5,
    "n_firms_init": 10,
    "uc_0": 0.672,
    "kappa_0": 3.280,
    "r_0_technical": 0.10,  # Implied p_0 = uc_0 + 0.10 * kappa_0 = 1.000
    "p_0": 1.0,
    "K_i0": 10.0,
    "B_i0": 30.0,
    "target_share_0": 0.10,
    "realized_share_0": 0.10,
    "u_init": 0.82,
}

# Standard experiment protocol
EXPERIMENT_PARAMS: dict[str, Any] = {
    "n_steps": 120,
    "burn_in": 40,
    "n_replications": 1000,
    "seeds": list(range(1000)),
    "illustrative_seed": 42,
}

# 27 exogenous parameters and exploratory Morris ranges (Appendix A): (name, low, high, baseline)
MORRIS_FACTORS: list[tuple[str, float, float, float]] = [
    ("D", 65.6, 98.4, 82.0),
    ("g_D", 0.0075, 0.0225, 0.015),
    ("eta", 1.0, 3.0, 2.0),
    ("alpha_p", 0.05, 0.25, 0.15),
    ("u_n", 0.74, 0.90, 0.82),
    ("T", 12.0, 20.0, 16.0),
    ("gamma", 0.80, 1.80, 1.30),
    ("delta_cost", 0.015, 0.045, 0.030),
    ("eps_0", 0.0, 0.005, 0.0),
    ("mu_z", 0.04, 0.10, 0.07),
    ("sigma_z", 0.015, 0.035, 0.025),
    ("lam", 0.40, 1.00, 0.70),
    ("beta_r", 1.8, 3.4, 2.6),
    ("beta_u", 0.08, 0.20, 0.14),
    ("R", 0.005, 0.015, 0.01),
    ("ellipse_ratio", 2.5, 5.5, 4.0),
    ("marx_theta", -1.08, -0.78, -0.9310265801),
    ("marx_mu_uc", 0.020, 0.044, 0.032),
    ("marx_mu_kappa", -0.055, -0.031, -0.043),
    ("gamma_thresh", 0.15, 0.35, 0.25),
    ("r_star", 0.025, 0.055, 0.04),
    ("psi", 10.0, 30.0, 20.0),
    ("f_0", 0.03, 0.09, 0.06),
    ("K_0", 2.5, 7.5, 5.0),
    ("B_0", 15.0, 45.0, 30.0),
    ("s_min", 0.03, 0.09, 0.060),
    ("tau_grace", 3.0, 7.0, 5.0),
]


def baseline_parameters(
    *, overrides: Optional[dict[str, Any]] = None, **parameters: Any
) -> dict[str, Any]:
    """Return validated baseline configuration with all extension switches explicitly disabled."""
    result = {**BASELINE_PARAMS, **BASELINE_RULES, **parameters}
    if overrides:
        result.update(overrides)
    enabled = [name for name in BASELINE_RULES if bool(result[name])]
    if enabled:
        raise ValueError(
            "Extensions are disabled in the baseline: "
            + ", ".join(enabled)
            + ". Instantiate an explicit, separately named extension scenario instead."
        )
    result.update(BASELINE_RULES)
    return result


def build_baseline_model(
    innovation_regime: str,
    seed: int,
    overrides: Optional[dict[str, Any]] = None,
) -> "RealCompetitionModel":
    """Instantiate RealCompetitionModel with centralized baseline parameters."""
    from src.model import RealCompetitionModel

    cfg = baseline_parameters(
        overrides=overrides, innovation_regime=innovation_regime, seed=seed
    )
    return RealCompetitionModel(**cfg)


def init_paired_homogeneous_firms(model: "RealCompetitionModel", seed: int = 0) -> None:
    """Initialize 10 firms in 5 age-matched Shaikh/Okishio pairs with identical endowments."""
    from src.agents import FirmAgent, Vintage

    ip = INITIAL_FIRM_PARAMS
    n_pairs = ip["n_pairs"]
    for g in range(n_pairs):
        birth_g = -math.floor(g * model.T / n_pairs)
        pair_rules = (
            ("shaikh", "okishio") if ((g + seed) % 2 == 0) else ("okishio", "shaikh")
        )
        for rule in pair_rules:
            v0 = Vintage(
                uc=ip["uc_0"],
                kappa=ip["kappa_0"],
                k_init=ip["K_i0"],
                k=ip["K_i0"],
                birth_period=birth_g,
            )
            firm = FirmAgent(
                model=model,
                rule=rule,
                initial_vintages=[v0],
                uc_b=ip["uc_0"],
                kappa_b=ip["kappa_0"],
                price=ip["p_0"],
                target_share=ip["target_share_0"],
                budget=ip["B_i0"],
                u_init=ip["u_init"],
                entry_period=0,
            )
            firm.realized_share = ip["realized_share_0"]


def init_paired_staircase_firms(
    model: "RealCompetitionModel",
    seed: int = 0,
    ucs_steps: tuple[float, ...] = (0.222, 0.322, 0.422, 0.522, 0.622),
) -> None:
    """Robustness-only paired 5-step initial prime-cost staircase initialization."""
    from src.agents import FirmAgent, Vintage

    ip = INITIAL_FIRM_PARAMS
    n_pairs = ip["n_pairs"]
    kappa_0 = ip["kappa_0"]
    r_0 = ip["r_0_technical"]
    for g in range(n_pairs):
        birth_g = -math.floor(g * model.T / n_pairs)
        ucs_g = ucs_steps[g]
        uc_g = ucs_g + kappa_0 / (model.T * model.u_n)
        p_g = uc_g + r_0 * kappa_0
        pair_rules = (
            ("shaikh", "okishio") if ((g + seed) % 2 == 0) else ("okishio", "shaikh")
        )
        for rule in pair_rules:
            v0 = Vintage(
                uc=uc_g,
                kappa=kappa_0,
                k_init=ip["K_i0"],
                k=ip["K_i0"],
                birth_period=birth_g,
            )
            firm = FirmAgent(
                model=model,
                rule=rule,
                initial_vintages=[v0],
                uc_b=uc_g,
                kappa_b=kappa_0,
                price=p_g,
                target_share=ip["target_share_0"],
                budget=ip["B_i0"],
                u_init=ip["u_init"],
                entry_period=0,
            )
            firm.realized_share = ip["realized_share_0"]


def collect_provenance(source_files: list[str]) -> dict[str, Any]:
    """Record runtime environment, dependency versions, config, and SHA-256 checksums."""
    hashes = {}
    for path in source_files:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        hashes[path] = h.hexdigest()

    deps = {}
    for pkg in ("numpy", "pandas", "scipy", "matplotlib", "mesa", "joblib", "pytest"):
        try:
            deps[pkg] = pkg_version(pkg)
        except Exception:
            deps[pkg] = "not-installed"

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": deps,
        "baseline_params": BASELINE_PARAMS,
        "initial_firm_params": INITIAL_FIRM_PARAMS,
        "experiment_params": EXPERIMENT_PARAMS,
        "source_sha256": hashes,
    }
