# Real Competition ABM: Shaikh vs. Okishio Technical Change

Agent-based model of within-sector **real competition** (Shaikh, 1978, 2016) with heterogeneous *one-hoss shay* fixed-capital vintages, comparing **Shaikh** (complete unit-cost cutting) and **Okishio** (normal profit-rate maximizing) investment rules under Neutral and Marx-biased technical change.

---

## Repository Structure

```text
├── README.md                  # Model presentation, key equations, reproduction commands
├── requirements.txt           # Pinned Python scientific stack
├── src/
│   ├── model.py               # Main annual loop (Steps 0 to 6) and invariant checks
│   ├── agents.py              # Firms (Shaikh / Okishio) and one-hoss shay vintages
│   ├── innovation.py          # DLF stochastic draws (Neutral vs Marx-biased) and adoption rules
│   └── calibration.py         # Baseline parameters (Table 3), Morris bounds, paired init
├── experiments/
│   ├── run_baseline.py        # N = 1000 paired runs/regime (Neutral vs Marx) & figures
│   ├── run_convergence.py     # Monte Carlo RSEM & CV diagnostics (N* = 840)
│   └── run_morris.py          # Global Morris screening (13,440 runs) + local sweeps (3,072 runs)
└── tests/
    └── test_invariants.py     # Pytest suite: share conservation, ucs > 0, Neutral 50/50 parity
```

---

## Key Model Equations (Annual Step t, Horizon T = 120 Years)

1. **Fixed-Capital Vintages & Firms** (*one-hoss shay* physical life `T = 16` years):

```math
uc_v = ucs_v + \frac{\kappa_v}{T \cdot u_n}, \qquad
\overline{uc}_i(t) = \frac{1}{K_i(t)} \sum_{v \in \mathcal{C}_i(t)} uc_v k_v(t), \qquad
\overline{\kappa}_i(t) = \frac{1}{K_i(t)} \sum_{v \in \mathcal{C}_i(t)} \kappa_v k_v(t)
```

   with strict reproduction cost floor `p_i(t) >= uc_bar_i(t)`.

2. **Step 0 — Aggregate Demand with Adaptive Price Memory** (`eta = 2.0`, `alpha_p = 0.15`):

```math
\tilde{p}(t-1) = (1 - \alpha_p) \tilde{p}(t-2) + \alpha_p p^{\text{leader}}(t), \qquad
D(t) = D_0 (1 + g_D)^t \left(\frac{\tilde{p}(t-1)}{p^{\text{leader}}(t)}\right)^{\eta}
```

3. **Step 1 — Real-Competition Pricing** (normal regime `S(t) >= D(t)` vs. shortage regime `S(t) < D(t)`):

   - Normal capacity sufficient (`u_n * K_i(t) >= f_i(t-1) * D(t)`): undercut incoming leader

```math
p_i(t) = \max\left\lbrace \overline{uc}_i(t),\; p^{\text{leader}}(t)\big(1 - \epsilon_i(t)\big) \right\rbrace, \qquad
\epsilon_i(t) = \max\left(0,\; \delta_{\text{cost}}\frac{\max(0, p^{\text{leader}}(t) - \overline{uc}_i(t))}{p^{\text{leader}}(t)} + \epsilon_0\right)
```

   - Normal capacity shortfall (`u_n * K_i(t) < f_i(t-1) * D(t)`): upward price adjustment scaled by `gamma * (1 - u_n * K_i / (f_i * D))`:

```math
p_i(t) = \max\left\lbrace \overline{uc}_i(t),\; p_i(t-1)\left(1 + \gamma\left(1 - \frac{u_n K_i(t)}{f_i(t-1) D(t)}\right)\right) \right\rbrace
```

   - Shortage regime (`S(t) < D(t)`), with truncated normal draw `z_i(t) ~ N_+(mu_z, sigma_z^2)`:

```math
p_i(t) = \max\left\lbrace \overline{uc}_i(t),\; p_i(t-1),\; p^{\text{leader}}(t)\big(1 + z_i(t)\big) \right\rbrace
```

4. **Step 2 — Quasi-Replicator Demand Allocation** (`sum_i f_i(t) = 1`, capacity-constrained redistribution):

```math
f_i(t) = f_i(t-1) \cdot \max\left\lbrace 0,\; 1 + \lambda \frac{p_{l(t)}(t) - p_i(t)}{p_{l(t)}(t)} \right\rbrace \quad (i \notin \mathcal{L}(t)), \qquad
q_i^{\max}(t) = \sum_{v:\, uc_v \le p_i(t)} k_v(t)
```

   Shed shares are reallocated proportionally among tied lowest-price leaders in $\mathcal{L}(t)$, and unfulfilled demand is redistributed across rivals' spare profitable supply.

5. **Steps 3–4 — Merit-Order Production & Scrapping**:

   Output $q_i(t)$ is allocated by ascending unit cost $uc_v$ (tie-broken by newest vintage); budgets retain net profits plus depreciation allowances $\sum_v \frac{\kappa_v}{T \cdot u_n} q_{i,v}(t)$. Vintages retire permanently if age reaches $T$ or `uc_v > p_i(t)`.

6. **Step 5 — DLF Technical Change, Viability Criteria & Internal Finance**:

```math
ucs_+ = \frac{ucs_b}{1 + \Delta ucs}, \qquad
\kappa_+ = \frac{\kappa_b}{1 + \Delta \kappa}, \qquad
uc_+ = ucs_+ + \frac{\kappa_+}{T \cdot u_n}
```

   - **Okishio rule** at benchmark normal price $p_i^*(t) = uc_{b,i} + \frac{r^*}{u_n}\kappa_{b,i}$: adopt iff

```math
u_n \frac{p_i^*(t) - uc_+}{\kappa_+} > u_n \frac{p_i^*(t) - uc_{b,i}}{\kappa_{b,i}}
```

   - **Shaikh rule**: adopt iff

```math
uc_+ < uc_{b,i}
```

   - Frontier consolidation occurs symmetrically once capacity share reaches `gamma_thresh = 0.25`. Gross investment is strictly bounded by pure internal finance $B_i(t)$:

```math
I_i(t) = \max\left\lbrace 0,\; \min\left(g_D K_i(t) + \delta_K K_i(t) + K_i(t)\big[\beta_r (r_i(t) - r^*) + \beta_u (u_i(t) - u_n)\big],\; \frac{B_i(t)}{\kappa_{\text{act}, i}(t)}\right) \right\rbrace
```

7. **Step 6 — Poisson Entry & Exit**:

```math
N_t^{\text{entry}} \sim \mathcal{P}\big(\psi \max\lbrace 0,\; r_t^{\text{branch}} - r^* \rbrace\big)
```

   Entrants adopt the Shaikh or Okishio rule symmetrically (50% / 50%) with demand-scaled endowments `D(t) / D_0`. Firms exit if realized share falls below `s_min` after `tau_grace`, or if physically and financially exhausted (`K_i = 0` and `B_i < min(kappa_b, kappa_a)`).

---

## Baseline Calibration Summary (`src/calibration.py`)

| Parameter | Baseline | Description | Parameter | Baseline | Description |
| --- | --- | --- | --- | --- | --- |
| `D_0 / g_D` | `82.0 / 0.015` | Initial demand & 1.5%/yr trend growth | `R` | `0.010` | Neutral DLF innovation disk radius |
| `eta / alpha_p` | `2.0 / 0.15` | Price elasticity & adaptive memory speed | `a / b / theta` | `0.020 / 0.005 / -0.9310` | Marx-biased rotated ellipse axes & angle |
| `u_n / T` | `0.82 / 16` | Normal utilization & *one-hoss shay* life (yrs) | `mu_ucs / mu_kappa` | `0.032 / -0.043` | Prime-cost saving & capital-using drifts |
| `gamma / delta_cost` | `1.30 / 0.030` | Shortfall markup & undercutting slope | `gamma_thresh` | `0.25` | Symmetric frontier consolidation share |
| `lam / beta_r / beta_u` | `0.70 / 2.6 / 0.14` | Share replicator & D&L investment gains | `r_star / psi / s_min` | `0.04 / 20.0 / 0.060` | General profit rate, Poisson rate, exit floor |

---

## Environment Setup

```bash
pip install -r requirements.txt
```

Validated runtime used for all paper results: **Python 3.13.9** with pinned dependencies in `requirements.txt`.

---

## Exact Reproduction Commands

1. **Run the unit, invariant, and Neutral 50/50 parity suite**:
   ```bash
   pytest tests/test_invariants.py -v
   ```

2. **Reproduce the N = 1000 paired baseline (Neutral vs. Marx-biased, T = 120, feasibility audit & figures)**:
   ```bash
   python experiments/run_baseline.py --n-reps 1000 --n-steps 120
   ```
   *(Optional flags: `--skip-staircase`, `--skip-figures`, `--n-jobs 4`)*

3. **Reproduce Monte Carlo convergence diagnostics (`RSEM <= 1.75%`, `|dCV| <= 2.0%`, `N* = 840`)**:
   ```bash
   python experiments/run_convergence.py --n-reps 1000 --n-steps 120 --burn-in 40
   ```

4. **Reproduce full 2-stage sensitivity analysis (13,440 Morris + 3,072 local/structural runs)**:
   ```bash
   python experiments/run_morris.py all --jobs 4
   ```
   *(Or run stage by stage: `prepare` -> `screen` -> `followup` -> `report`.)*

---

## Reference Baseline Results (N = 1000, T = 120)

| Indicator | Neutral Regime | Marx-Biased Regime |
| --- | --- | --- |
| Shaikh share at `t = 120` (post-burn-in `41-120`) | `51.56%` (`51.13%`) | `86.04%` (`77.58%`) |
| Shaikh win rate at `t = 120` (`s_Shaikh > 50%`) | `50.4%` | `98.0%` |
| Branch utilization (`t = 41-120 -> t = 120`) | `82.11% -> 81.99%` | `84.05% -> 84.18%` |
| Realized profit rate (`t = 41-120 -> t = 120`) | `4.10% -> 3.97%` | `3.99% -> 3.87%` |
| Sales-weighted price `p_120` / leader `p_leader_120` | `0.7717 / 0.7440` | `0.8487 / 0.8045` |
| Capital-output ratio `kappa_120` | `3.078` | `3.953` |
| Negative prime-cost vintages (`ucs_v < 0`) | `0` | `0` |
| Global Morris robustness (`560` configs) | — | `kappa_Marx > kappa_Neutral`: `100.0%` (`560/560`); `r_Marx < r_Neutral`: `86.25%` (`483/560`) |
