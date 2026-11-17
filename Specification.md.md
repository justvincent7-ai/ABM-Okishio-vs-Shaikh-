# Variables 
**Table 1** : Exogenous variables of the model 

| Symbol                   | Name / Description                    | Domain / Typical Value                 | Role in the Model                                                                                                            |
| :----------------------- | :------------------------------------ | :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| $D(t)$                   | Aggregate industry demand             | $\mathbb{R}_+^*$ (e.g., $D_0 = 85$)    | Determines the market regime ($S(t) \ge D(t)$ vs. $S(t) < D(t)$) and total sales volume                                      |
| $u_n$                    | Normal capacity utilization rate      | $(0, 1)$ (e.g., $0.85$)                | Engineering benchmark for desired target capacity $K_i^{\text{desired}}(t)$ and price undercutting[cite: 94, 98]              |
| $T$                      | Physical service life of capital      | $\mathbb{N}^*$ (e.g., $10$ periods)    | Fixed lifetime for abrupt *one-hoss shay* retirement of vintages[cite: 97]                                                   |
| $\gamma$                 | Utilization reaction parameter        | $\mathbb{R}_+^*$ (e.g., $0.02 - 0.05$) | Sensitivity of price undercutting to unused capacity $(u_n - u_i(t-1))$                                                      |
| $\delta_{\text{cost}}$   | Cost-advantage reaction parameter     | $\mathbb{R}_+$ (e.g., $0.08$)          | Sensitivity of price undercutting to relative cost advantage $\frac{p^{\text{leader}}(t) - \overline{uc}_i(t)}{p^{\text{leader}}(t)}$ |
| $\epsilon_0$             | Baseline undercutting step            | $\mathbb{R}_+$ (e.g., $\ge 0$)         | Minimum competitive price-cutting aggressiveness in the normal regime                                                        |
| $\mu_z, \sigma_z$        | Shortage markup parameters            | $\mathbb{R}_+^*$ (e.g., $0.01, 0.005$) | Mean and standard deviation of the truncated normal shock $\mathcal{N}_+$ when $S(t) < D(t)$                                 |
| $\lambda$                | Demand reallocation speed             | $(0, 1]$ (e.g., $0.8$)                 | Captures market frictions in the quasi-replicator selection dynamics                                                         |
| $R$                      | Neutral innovation set radius         | $(0, 1)$ (e.g., $0.05$)                | Maximum per-period factor-saving rate in the isotropic disk $\mathcal{I}_{\text{neutral}}$                                   |
| $a, b, \theta$           | Marx-biased ellipse parameters        | $a > b > 0, \, \theta < 0$             | Semi-axes ($\pi ab = \pi R^2$) and rotation angle toward $(\Delta uc > 0, \Delta \kappa < 0)$ in $\mathcal{I}_{\text{Marx}}$ |
| $\gamma_{\text{thresh}}$ | Technological consolidation threshold | $(0, 1)$ (e.g., $0.75$)                | Capacity share required to promote frontier technique $(uc_a, \kappa_a)$ to benchmark                                        |
| $r^*$                    | General economy-wide profit rate      | $\mathbb{R}_+^*$ (e.g., $0.10 - 0.15$) | Exogenous profitability benchmark triggering new firm entry                                                                  |
| $\psi$                   | Capital entry speed                   | $\mathbb{R}_+^*$                       | Arrival intensity parameter of the Poisson entry process $N_t^{\text{entry}}$                                                |
| $f_0, K_0, B_0$          | Entrant initial endowments            | $\mathbb{R}_+^*$                       | Baseline target market share, initial capital stock, and initial liquidity of entering firms                                 |
| $s_{\min}$               | Minimum viability threshold           | $(0, 1)$ (e.g., $0.01$)                | Critical market share below which marginalized firms exit the industry                                                       |

**Table 2:** Endogenous variables of the model 

| Symbol                   | Name / Description                   | Equation / Definition                                                                                     | Level & Step                                   |
| :----------------------- | :----------------------------------- | :-------------------------------------------------------------------------------------------------------- | :--------------------------------------------- |
| $K_i(t)$                 | Total productive capacity            | $\sum_{v \in \mathcal{C}_i(t)} k_v(t)$                                                                    | Firm (Aggregate)                               |
| $\overline{uc}_i(t)$     | Weighted average unit operating cost | $\frac{1}{K_i(t)} \sum_{v \in \mathcal{C}_i(t)} uc_v k_v(t)$                                              | Firm (Direct price floor $p_i^{\text{floor}}$) |
| $\overline{\kappa}_i(t)$ | Weighted average unit capital cost   | $\frac{1}{K_i(t)} \sum_{v \in \mathcal{C}_i(t)} \kappa_v k_v(t)$                                          | Firm (Aggregate)                               |
| $\overline{r}_i(t)$      | Normal profit rate                   | $\frac{p_i(t) - \overline{uc}_i(t)}{\overline{\kappa}_i(t)}$                                              | Firm (Profitability)                           |
| $S(t)$                   | Aggregate industry capacity          | $\sum_{i \in \mathcal{N}_t} K_i(t)$                                                                       | Industry (Step 1)                              |
| $p^{\text{leader}}(t)$   | Incoming price leader                | $\min_j \{p_j(t-1)\}$                                                                                     | Industry (Step 1)                              |
| $p_i(t)$                 | Posted market price                  | $\max\{p_i^{\text{floor}}(t), p^{\text{leader}}(t)(1-\epsilon_i(t))\}$ (if $S \ge D$)[cite: 94]           | Firm (Step 1)                                  |
| $\epsilon_i(t)$          | Undercutting aggressiveness          | $\max\left(0, \gamma(u_n - u_i(t-1)) + \delta_{\text{cost}} \cdot \frac{p^{\text{leader}}(t) - \overline{uc}_i(t)}{p^{\text{leader}}(t)} + \epsilon_0\right)$ | Firm (Step 1)                                  |
| $f_i(t)$                 | Target market share                  | Quasi-replicator anchored on $p_{l(t)}(t)$ ($\sum f_i = 1$)                                               | Firm (Step 2)                                  |
| $q_i^{\max}(t)$          | Profitable available supply          | $\sum_{v: uc_v \le p_i(t)} k_v(t)$                                                                        | Firm (Step 2)                                  |
| $q_i(t)$                 | Realized sales / output              | $q_i^{\text{init}}(t) + \text{spare}_i(t) \min\left(1, \frac{\Delta D(t)}{\sum \text{spare}_j(t)}\right)$ | Firm (Step 2)                                  |
| $s_i(t)$                 | Realized market share                | $\frac{q_i(t)}{\sum_j q_j(t)}$                                                                            | Firm (Step 2)                                  |
| $u_i(t)$                 | Actual capacity utilization rate     | $\frac{q_i(t)}{K_i(t)}$                                                                                   | Firm (Step 2)                                  |
| $\Pi_i(t)$               | Net operating profit                 | $\sum_{v \in \mathcal{C}_i(t)} (p_i(t) - uc_v) q_{i,v}(t)$                                                | Firm (Step 3)                                  |
| $B_i(t)$                 | Internal investment budget           | $B_i(t-1) + \Pi_i(t)$                                                                                     | Firm (Step 3)                                  |
| $K_i^{\text{surv}}(t)$   | Surviving capacity post-scrapping    | $\sum_{v \in \mathcal{C}_i(t+1)} \Delta k_v \cdot \mathbb{I}(a_v(t) < T)$                                 | Firm (Step 4)                                  |
| $K_i^{\text{desired}}(t)$ | Desired target capacity             | $\frac{\max(f_i(t)D(t), \, q_i(t))}{u_n}$                                                                 | Firm (Step 5)                                  |
| $I_i(t)$                 | Target capacity expansion             | $\max(0, K_i^{\text{desired}}(t) - K_i^{\text{surv}}(t))$                                                 | Firm (Step 5)                                  |
| $\Delta k_i(t)$          | Acquired new capacity                 | $\min\left(I_i(t), \, \frac{B_i(t)}{\kappa_{\text{act}}}\right)$                                          | Firm (Step 5)                                  |
| $r_t^{\text{branch}}$    | Aggregate branch profit rate         | $\frac{\sum_i \Pi_i(t)}{\sum_i \sum_v \kappa_v k_v(t)}$                                                   | Industry (Step 6)                              |
| $N_t^{\text{entry}}$     | Number of entering firms             | $\mathcal{P}(\psi \cdot \max\{0, r_t^{\text{branch}} - r^*\})$                                            | Industry (Step 6)                              |
| $\mathcal{N}_t$          | Active firm population               | Updated after entry ($N_t^{\text{entry}}$) and exit ($s_i < s_{\min}$)                                    | Industry (Step 6)                              |

# The model 
Contrary to the simultaneous approach, in a dynamic multi-agent system, the profit rate $r$ is not the outcome of an equilibrium. Instead, it is an endogenous variable that depends on the market price and the productive technique in use.

A technique is defined as a combination of a unit operating cost ($uc$) and a unit capital cost ($\kappa$). Hence, the profit rate of a given technique—or rather, of a specific vintage of capital $(uc, \kappa)$—at price $p$ is given by:
$$r = \frac{p - uc}{\kappa}$$

In principle, unit cost $uc$ encompasses fixed capital depreciation ($\delta \kappa$), labour costs, and intermediate input costs; however, we abstract from this internal decomposition in the present model. Particularly, we consider that the unit cost does not inegrate the fixed capital depreciation. 

As noted by Shaikh, it is crucial to distinguish between capitals and firms: *"The distinction between firms and capitals is important because firms set prices but the operating conditions of capitals determine costs."* (Shaikh, 2016, p. 262). Accordingly, the aggregate profit rate of firm $i$ at period $t$ depends on its capacity-weighted mean unit cost $\overline{uc}_i(t)$ and mean unit capital cost $\overline{\kappa}_i(t)$. Let $\mathcal{C}_i(t)$ denote the set of active individual capitals (vintages) owned by firm $i$ at period $t$, where each element $v \in \mathcal{C}_i(t)$ corresponds to an amount of surviving productive capacity $k_v(t)$ acquired under a specific technique $(uc_v, \kappa_v)$. Firm $i$'s total productive capacity, mean unit cost, and mean unit capital cost are then defined as:
$$K_i(t) = \sum_{v \in \mathcal{C}_i(t)} k_v(t)$$
$$\overline{uc}_i(t) = \frac{1}{K_i(t)} \sum_{v \in \mathcal{C}_i(t)} uc_v \, k_v(t), \qquad \overline{\kappa}_i(t) = \frac{1}{K_i(t)} \sum_{v \in \mathcal{C}_i(t)} \kappa_v \, k_v(t)$$

Firm $i$'s normal profit rate at price $p_i(t)$ (evaluated at normal capacity utilization $u_n$) is therefore:
$$\overline{r}_i(t) = \frac{p_i(t) - \overline{uc}_i(t)}{\overline{\kappa}_i(t)}$$

Common to both Duménil and Lévy (1993, pp. 143–144) and Shaikh (2016), price determination is an active managerial decision: firms are price-setters rather than passive price-takers. Hence, the price equation takes the accounting identity form:
$$p_i(t) = \overline{uc}_i(t) + \overline{r}_i(t) \cdot \overline{\kappa}_i(t)$$

Since our model does not include unsold inventories, a firm's pricing behavior is bounded below by its direct price floor $p^{\text{floor}}_i(t) = \overline{uc}_i(t)$, ensuring non-negative operating margins:
$$p_i(t) \ge \overline{uc}_i(t)$$

# Step 1: Price-setting behavior

According to Shaikh, firms' price-setting behavior is driven by the competitive imperative to gain market share and realize profits. Consequently, selling prices within an industry tend to equalize; however, because technical change is ongoing, plants operate with heterogeneous cost structures, generating persistent differentials in profit margins and profit rates (Shaikh, 2016, pp. 261–264).

The price leader entering period $t$ is the active firm that posted the lowest price in the industry at the end of period $t-1$:
$$p^{\text{leader}}(t) = \min_{j} \{ p_j(t-1) \}$$

Adapting Duménil and Lévy's (1993, p. 122) *"establishment of prices procedure"*, we define two pricing regimes depending on aggregate industry capacity. We assume that firms have complete information regarding aggregate industry productive capacity $S(t) = \sum_{i} K_i(t)$ and aggregate market demand $D(t)$.

**The normal regime ($S(t) \ge D(t)$)**

Firms compete for market share by attempting to undercut the current price leader by a fractional step $\epsilon_i(t) \in [0, 1)$, bounded below by their respective direct price floors:
$$p_i(t) = \max \left\{ p^{\text{floor}}_i(t), \; p^{\text{leader}}(t)\big(1 - \epsilon_i(t)\big) \right\}$$

Adapting Duménil and Lévy (1993, pp. 122, 143) to Shaikh's (2016, Ch. 4 & 7) framework of normal capacity utilization and cost-driven real competition, each firm sets its undercutting aggressiveness according to its unused productive capacity relative to the normal utilization target $u_n \in (0,1)$ and its relative unit cost advantage over the incoming price leader $p^{\text{leader}}(t)$:
$$\epsilon_i(t) = \max\left(0, \; \gamma\big(u_n - u_i(t-1)\big) + \delta_{\text{cost}} \cdot \frac{p^{\text{leader}}(t) - \overline{uc}_i(t)}{p^{\text{leader}}(t)} + \epsilon_0\right)$$
where $\gamma > 0$ is a utilization reaction parameter, $\delta_{\text{cost}} \in \mathbb{R}_+$ (typically $0.08$) is the sensitivity of price undercutting to the firm's relative cost advantage, and $\epsilon_0 \ge 0$ is a baseline competitive undercutting step. Firms operating below normal capacity ($u_i(t-1) < u_n$) or enjoying lower unit operating costs ($\overline{uc}_i(t) < p^{\text{leader}}(t)$) actively cut prices to capture rival market share, whereas firms operating significantly above normal capacity without a sufficient cost margin ($\epsilon_i(t) = 0$) halt their price-cutting behavior.

**The shortage regime ($S(t) < D(t)$)**

When $S(t) < D(t)$, demand tension subsides price-cutting pressure; nonetheless, firms mark up their prices relative to the leader, subject to their direct cost floors and previous prices:
$$p_i(t) = \max\Big\{ p^{\text{floor}}_i(t), \; p_i(t-1), \; p^{\text{leader}}(t)\big(1 + z_i(t)\big) \Big\}$$
where $z_i(t) \sim \mathcal{N}_+(\mu_z, \sigma_z^2)$ (truncated to $\mathbb{R}_+$) captures heterogeneous upward price adjustments.

While analytically simple, recent empirical research confirms that this rule captures real-world dynamic pricing algorithms, where a firm's *"best response is often to undercut rivals' prices"* (Brown and MacKay, 2023, p. 122). Indeed, commercial pricing rules closely mirror our cost-floored specification: for instance, Mattes (2022, p. 1502) cites a firm's pricing algorithm in which, whenever a rival's price exceeds the firm's own production cost, the algorithm automatically undercuts the competitor by a fixed proportion.

# Step 2 : Demand allocation 

Adapting the quasi-replicator dynamics of Dosi and Roventini (2025, p. 40) to Shaikh's theory of real competition, we anchor competitive selection to the lowest-price producer in the industry rather than the industry average.

Specifically, let $l(t) = \arg\min_j \{ p_j(t) \}$ denote the current price leader posting the lowest active price $p_{l(t)}(t) = \min_j \{ p_j(t) \}$ in period $t$. Simplifying Dosi and Roventini's (2025, Eq. 13, p. 40) competitiveness function to posted prices ($E_i(t) = p_i(t)$) with the current leader's price acting as the social benchmark $\overline{E}(t) = p_{l(t)}(t)$, each higher-priced firm's target market share $f_i(t) \in [0, 1]$ evolves according to:
$$f_i(t) = f_i(t-1) \cdot \max \left\{ 0, \; 1 + \lambda \frac{p_{l(t)}(t) - p_i(t)}{p_{l(t)}(t)} \right\}, \qquad \forall i \neq l(t)$$
where $\lambda \in (0, 1]$ governs the speed of demand reallocation and captures market frictions. Since $p_i(t) \ge p_{l(t)}(t)$ for all rival producers, the fractional price gap $\frac{p_{l(t)}(t) - p_i(t)}{p_{l(t)}(t)}$ is non-positive, eroding the market share of higher-priced followers (while leaving ties unaffected). The current price leader directly absorbs the market shares shed by rivals, ensuring $\sum_i f_i(t) = 1$ by construction:
$$f_{l(t)}(t) = 1 - \sum_{i \neq l(t)} f_i(t)$$
## Redistribution of demand and realized sales

Each firm's realized sales are strictly bounded by its profitable productive capacity. Consistent with Shaikh (2016, p. 159), firm $i$ offers all active vintages $v \in \mathcal{C}_i(t)$ whose unit operating costs are covered by its posted price $p_i(t)$, defining its maximum available supply:
$$q_i^{\max}(t) = \sum_{\substack{v \in \mathcal{C}_i(t) \\ uc_v \le p_i(t)}} k_v(t)$$
Given target market shares $f_i(t)$, initial allocated demand is $d_i^{\text{init}}(t) = f_i(t) D(t)$, and initial sales are capped at:
$$q_i^{\text{init}}(t) = \min \left\{ q_i^{\max}(t), \; d_i^{\text{init}}(t) \right\}$$
Whenever one or more firms hit their capacity ceiling ($d_j^{\text{init}}(t) > q_j^{\max}(t)$), the resulting aggregate unfulfilled demand $\Delta D(t) = \sum_j \big(d_j^{\text{init}}(t) - q_j^{\text{init}}(t)\big)$ is reallocated to rivals in proportion to their spare capacity $\text{spare}_i(t) = \max\big(0, \; q_i^{\max}(t) - q_i^{\text{init}}(t)\big)$. Whenever $\sum_j \text{spare}_j(t) > 0$, realized sales are given by:
$$q_i(t) = q_i^{\text{init}}(t) + \text{spare}_i(t) \cdot \min \left\{ 1, \; \frac{\Delta D(t)}{\sum_j \text{spare}_j(t)} \right\}$$
(and $q_i(t) = q_i^{\text{init}}(t)$ if $\sum_j \text{spare}_j(t) = 0$). When aggregate unfulfilled demand exceeds total industry spare capacity ($\Delta D(t) > \sum_j \text{spare}_j(t)$), the residual remains unmet demand.

Realized market shares $s_i(t)$ are then defined as:
$$s_i(t) = \frac{q_i(t)}{\sum_j q_j(t)}$$
The capacity utilization entering the next period is $u_i(t) = q_i(t) / K_i(t)$).

# Step 3: Production process and accounting

To fulfill its realized output $q_i(t)$, each firm activates its portfolio of capital vintages $\mathcal{C}_i(t)$ following an internal merit order ranked by ascending unit operating costs (and ascending vintage age in case of ties):
$$uc_{i,(1)} \le uc_{i,(2)} \le \dots \le uc_{i,(|\mathcal{C}_i(t)|)}$$
allocating production sequentially across vintages up to $q_i(t)$ so that $\sum_{v \in \mathcal{C}_i(t)} q_{i,v}(t) = q_i(t)$ with $0 \le q_{i,v}(t) \le k_v(t)$.

Net operating profits $\Pi_i(t)$ are computed by summing margins across active vintages:
$$\Pi_i(t) = \sum_{v \in \mathcal{C}_i(t)} \big(p_i(t) - uc_v\big) q_{i,v}(t)$$

Assuming all net profits are retained for internal accumulation (zero dividend payout), firm $i$'s available investment budget entering the capital accumulation stage evolves according to:
$$B_i(t) = B_i(t-1) + \Pi_i(t), \qquad \text{with } B_i(0) = B_{i,0}$$

# Step 4 : Scrapping 
Following Duménil and Lévy (1994, p. 234), fixed capital exhibits a finite physical service life and a high degree of technological stickiness: *"The service life of fixed capital is longer than 10 years, and there is a large degree of stickiness of technology."* We adopt a «one-hoss shay» (OCDE, 2001, p. 102) depreciation structure: each vintage $v \in \mathcal{C}_i(t)$ retains its full initial productive capacity $\Delta k_v$ throughout a fixed physical service life of $T = 10$ periods regardless of utilization, after which it is retired abruptly. Let $a_v(t) = t - \tau_v + 1$ denote the age of vintage $v$ (acquired in period $\tau_v$) at the end of period $t$. Its surviving capacity entering period $t+1$ is given by:
$$
k_v(t+1) = \Delta k_v \cdot \mathbb{I}\big(a_v(t) < T\big) \cdot \mathbb{I}\big(uc_v \le p_i(t)\big)
$$
where $\mathbb{I}(\cdot)$ is the indicator function and $a_v(t) = t - \tau_v + 1$ denotes the vintage's age at the end of period $t$. Any vintage rendered unprofitable by competitive price reductions ($uc_v > p_i(t)$) is permanently scrapped rather than retained as idle capacity, preventing obsolete equipment from distorting surviving capacity $K_i^{\text{surv}}(t)$ and depressing replacement investment $I_i(t)$ in Step 5:

$$
K_i^{\text{surv}}(t) = \sum_{v \in \mathcal{C}_i(t+1)} k_v(t+1)
$$
# Step 5 : Investment and innovation 
## Pure Internal Finance and Target Capacity
Our model abstracts from external financing by banks and financial institutions; hence, capital accumulation is governed by «pure internal finance» (Shaikh, 2016, p. 618), whereby each firm's investment expenditure is strictly bounded by its accumulated internal retained earnings $B_i(t)$. This specification directly embeds the accumulation process within the Marxian circuit of capital ($M - C - M'$) as well as Joan Robinson's (1956) post-Keynesian emphasis on internal cash flows as the primary determinant and constraint of firm expansion.
Following Shaikh (2009, p. 459), we assume an exogenous industry-wide desired (or normal) rate of capacity utilization $u_n \in (0, 1)$, anchored at the minimum-unit-cost engineering point of production (Shaikh, 2016, pp. 129–136). Since «the amount of investment that is justified depends on the economically desired rate of utilization» (Shaikh, 2009, p. 459), demand and normal capacity mutually adjust over time (Shaikh, 2009, p. 461). The importance over the difference between the normal utilisation rate and the current utilisation rate is central to the marxist investment function (Lavoie et al., 2004, p. 134).
In our model, we suppose that the desired capacity of the firm adjusts to the effective demand addressed to the firm $\max\big(f_i(t)D(t), \, q_i(t)\big)$ (accounting for both target demand $d_i^{\text{init}}(t) = f_i(t)D(t)$ when capacity-rationed and realized sales $q_i(t)$ when absorbing rival spillover) and the normal utilisation rate set by firms ($u_n$). Hence:
$$K^{\text{desired}}_i(t) = \frac{\max\big(f_i(t)D(t), \, q_i(t)\big)}{u_n}$$
Therefore, the target capacity expansion (including physical replacement of retiring vintages) becomes:
$$I_i(t) = \max\left(0, \, K_{i}^{\text{desired}}(t) - K_{i}^{\text{surv}}(t)\right)$$

## Innovation set 
Each firm can innovate and discover a new technique. We suppose that the discovery process—R&D—is internal to the firm. Following the Duménil-Lévy-Foley (DLF) framework and its subsequent extensions developed by Park (2001) and Shaikh (2016), we define a candidate technique relative to the firm's current benchmark technique $(uc_b, \kappa_b)$ by a pair of factor-saving growth rates $(\Delta uc, \Delta \kappa) \in \mathbb{R}^2$ such that:

$$
uc_+ = \frac{uc_b}{1 + \Delta uc} \quad \text{and} \quad \kappa_+ = \frac{\kappa_b}{1 + \Delta \kappa}
$$

where $\Delta uc > 0$ denotes a reduction in unit operating costs (cost-saving technical change) and $\Delta \kappa < 0$ denotes an increase in unit capital requirements (capital-using or capital-deepening technical change).

To represent the localized and path-dependent nature of technological discovery, we define the **innovation set** $\mathcal{I}(R) \subset \mathbb{R}^2$ as a bounded domain centered around the origin $(0,0)$ in the $(\Delta uc, \Delta \kappa)$-plane. We consider two distinct stochastic regimes for technical change:
1. **Neutral Innovation Set (Isotropic Disk):**
   Technological opportunities are symmetrically distributed across all directions around the existing technique, bounded by a maximum per-period innovation radius $R \in (0, 1)$: 
   $$
   \mathcal{I}_{\text{neutral}}(R) = \left\{ (\Delta uc, \Delta \kappa) \in \mathbb{R}^2 \;\middle|\; (\Delta uc)^2 + (\Delta \kappa)^2 \le R^2 \right\}
   $$
2. **Marx-Biased Innovation Set (Rotated Ellipse):**
   Following Marx's and Shaikh's characterization of capitalist mechanization—where firms systematically search for labor- and input-saving methods at the expense of higher fixed capital intensity—the innovation set is elongated and tilted toward the South-East quadrant ($\Delta uc > 0, \Delta \kappa < 0$) by a rotation angle $\theta < 0$:
   $$
   \mathcal{I}_{\text{Marx}}(a, b, \theta) = \left\{ (\Delta uc, \Delta \kappa) \in \mathbb{R}^2 \;\middle|\; \frac{(\Delta uc \cos\theta + \Delta \kappa \sin\theta)^2}{a^2} + \frac{(-\Delta uc \sin\theta + \Delta \kappa \cos\theta)^2}{b^2} \le 1 \right\}
   $$
   where $a > b > 0$ denote the semi-major and semi-minor axes, calibrated to preserve the same total area ($\pi a b = \pi R^2$) as the neutral set.
Whenever a firm's R&D search channel is open, it draws a single candidate innovation $(\Delta uc, \Delta \kappa)$ uniformly from the active innovation set $\mathcal{I}$ and evaluates its adoption according to its behavioral decision rule.

## The Okishio Technical Change

A firm following an Okishio technical change behavior obeys a strict *"optimality criterion"* (Shaikh, 1978, p. 245). Evaluated at the firm's current posted price $p_i(t)$, the firm adopts a candidate innovation $(uc_+, \kappa_+)$ only if it provides a higher profit rate than its current benchmark technique $(uc_b, \kappa_b)$. Letting $r_+ = \frac{p_i(t) - uc_+}{\kappa_+}$ and $r_b = \frac{p_i(t) - uc_b}{\kappa_b}$ denote the respective profit rates, the viable adoption set for an Okishio firm is:

$$
\mathcal{V}_{\text{Okishio}}(t) = \left\{ (\Delta uc, \Delta \kappa) \in \mathcal{I} \;\middle|\; \frac{p_i(t) - uc_+}{\kappa_+} > \frac{p_i(t) - uc_b}{\kappa_b} \right\}
$$

In the $(\Delta uc, \Delta \kappa)$-plane, this defines the half-space strictly above the iso-profit-rate frontier $r_+ = r_b$, whose first-order linear approximation around $(0,0)$ is given by $\Delta \kappa \cdot (r_b \kappa_b) + uc_b \Delta uc > 0$.

## The Shaikh Technical Change

In contrast, a firm governed by Shaikh's theory of real competition obeys a *"cost-cutting criterion"* (Shaikh, 1978, 2016). Under the competitive imperative to undercut rivals and defend market share, the firm adopts any candidate technique that strictly reduces unit operating costs relative to its current benchmark technique, regardless of whether its immediate profit rate rises or falls (i.e., even when $\Delta \kappa < 0$ lowers the instantaneous rate of return). The viable adoption set for a Shaikh firm is therefore:
$$
\mathcal{V}_{\text{Shaikh}} = \left\{ (\Delta uc, \Delta \kappa) \in \mathcal{I} \;\middle|\; uc_+ < uc_b \right\} = \left\{ (\Delta uc, \Delta \kappa) \in \mathcal{I} \;\middle|\; \Delta uc > 0 \right\}
$$

## Technological Consolidation and Active Technique

To avoid financial bottlenecks and premature capital dilution across fragmented vintages, the technological diffusion discipline governed by the consolidation threshold $\gamma_{\text{thresh}} \in (0, 1)$ applies **symmetrically to both Okishio and Shaikh firms**. We define **consolidation** as the internal diffusion process through which an adopted frontier technique $(uc_a, \kappa_a)$ replaces the former benchmark technique $(uc_b, \kappa_b)$ once its capacity share within the firm's active capital stock reaches the critical threshold $\gamma_{\text{thresh}} \in (0, 1)$:

$$
\text{share}_i^{(a)}(t) = \frac{1}{K_i(t)} \sum_{v \in \mathcal{C}_i^{(a)}(t)} k_v(t) \ge \gamma_{\text{thresh}}
$$

where $\mathcal{C}_i^{(a)}(t) \subseteq \mathcal{C}_i(t)$ denotes the subset of active vintages embodied in the frontier technique $(uc_a, \kappa_a)$. Operationally, this determines the firm's active technique $(uc_{\text{act}}, \kappa_{\text{act}})$ at each period $t$:

1. **Consolidation Phase ($\text{share}_i^{(a)}(t) < \gamma_{\text{thresh}}$):** If the firm already holds an adopted frontier technique $(uc_a, \kappa_a)$ whose capacity share is below $\gamma_{\text{thresh}}$, it suspends new R&D draws ($\mathcal{I} = \emptyset$) and sets its active technique to $(uc_{\text{act}}, \kappa_{\text{act}}) = (uc_a, \kappa_a)$ until this share reaches $\gamma_{\text{thresh}}$ (at which point $(uc_a, \kappa_a)$ is promoted to become its new benchmark $(uc_b, \kappa_b)$, $(uc_a, \kappa_a) \leftarrow \emptyset$, and the R&D search channel reopens).
2. **Innovation Draw and Selection:** Otherwise, the firm draws a candidate innovation $(uc_+, \kappa_+)$ from $\mathcal{I}$. If the candidate satisfies the firm's adoption criterion ($\mathcal{V}_{\text{Okishio}}(t)$ for Okishio vs. $\mathcal{V}_{\text{Shaikh}}$ for Shaikh), it is adopted as the new frontier technique $(uc_a, \kappa_a) = (uc_+, \kappa_+)$ and becomes the active technique $(uc_{\text{act}}, \kappa_{\text{act}}) = (uc_a, \kappa_a)$; otherwise, the firm retains its benchmark as the active technique $(uc_{\text{act}}, \kappa_{\text{act}}) = (uc_b, \kappa_b)$.

#### Capital Acquisition and Portfolio Allocation
DLF is a dynamic stochastic framework; however, DLF relies on a representative firm and abstracts from internal budget constraints. Once the firm has updated its active technique $(uc_{\text{act}}, \kappa_{\text{act}})$, it invests exclusively in this active best technique to bridge its capacity gap $I_i(t)$, subject to pure internal finance $B_i(t)$:
$$\Delta k_i(t) = \min\left(I_i(t), \, \frac{B_i(t)}{\kappa_{\text{act}}} \right)$$
Newly acquired vintages enter the firm's capital portfolio $\mathcal{C}_i(t+1)$ with initial age $a_v = 0$, and any unspent retained earnings remain in $B_i(t)$ as an internal liquidity buffer for subsequent periods ($B_i(t) \leftarrow B_i(t) - \kappa_{\text{act}}\Delta k_i(t)$). Once the new vintage is installed (if $\Delta k_i(t) > 0$), if the capacity share of $(uc_a, \kappa_a)$ reaches $\gamma_{\text{thresh}}$, $(uc_a, \kappa_a)$ is immediately promoted to the new benchmark $(uc_b, \kappa_b)$.



# Step 6 : Entry and Exit 

To model the inter-branch mobility of capital and the competitive selection of firms, we introduce an exogenous benchmark rate of profit $r^* > 0$, representing the economy-wide general rate of profit.
## Capital Entry 

Following Shaikh's (2016, Ch. 7) theory of real competition, new capital flows into the industry when the realized profit rate of the branch exceeds the economy-wide normal benchmark $r^*$. Let $r_t^{\text{branch}}$ denote the aggregate branch-wide rate of profit at the end of period $t$:

$$
r_t^{\text{branch}} = \frac{\sum_{i \in \mathcal{N}_t} \Pi_i(t)}{\sum_{i \in \mathcal{N}_t} \sum_{v \in \mathcal{C}_i(t)} \kappa_v k_v(t)}
$$

Whenever $r_t^{\text{branch}} > r^*$, the number of entering firms $N_t^{\text{entry}}$ at the start of period $t+1$ is drawn from a Poisson distribution whose arrival intensity is proportional to the excess profitability spread:

$$
N_t^{\text{entry}} \sim \mathcal{P}\Big( \psi \cdot \max\big\{0, \, r_t^{\text{branch}} - r^*\big\} \Big)
$$

where $\psi > 0$ governs the speed of capital entry. Each newly entering firm $j$ has an equal probability ($0.5$) of adopting either a Shaikh or an Okishio behavioral rule. Firm $j$ enters with initial capital stock $K_0$, initial liquidity $B_0$, and embodies the prevailing best-practice technique of its respective behavioral category at the current industry price $p^{\text{leader}}(t)$. To preserve market-share conservation ($\sum_i f_i = 1$), entrants are initialized with a small baseline target share $f_0 \in (0, 1)$ (subject to $N_t^{\text{entry}} f_0 < 1$), and incumbent shares are rescaled proportionally by $(1 - N_t^{\text{entry}} f_0)$.

## Capital Exit

A firm $i \in \mathcal{N}_t$ permanently exits the industry at the end of period $t$ if it is eliminated by competitive selection or exhausts its productive and financial capacity. Specifically, exit occurs under either of two conditions:

1. **Market Marginalization (Post-Grace Period):** Let $a_i^{\text{firm}}(t) = t - t_{i,0}$ denote the age of firm $i$ since its entry period $t_{i,0}$ (with $a_i^{\text{firm}}(t) \ge \tau_{\text{grace}}$ for initial incumbents). Once a firm completes its initial grace period of $\tau_{\text{grace}} = 20$ periods, it exits if its realized market share falls below the minimum viability threshold $s_{\min} \in (0, 1)$:
   $$
   a_i^{\text{firm}}(t) \ge \tau_{\text{grace}} \quad \text{and} \quad s_i(t) < s_{\min}
   $$
2. **Physical and Financial Exhaustion:** All of the firm's capital vintages have been physically retired or rendered economically obsolete ($K_i(t) = 0$) and its remaining internal funds fall below the minimum capital outlay required to acquire a unit of new capacity ($B_i(t) < B_{\min, i}$, where $B_{\min, i} = \min\{\kappa_{b,i}, \kappa_{a,i}\}$ denotes the lowest unit capital cost available to firm $i$).

Upon exit, firm $i$ is removed from the active population $\mathcal{N}_{t+1}$, and its residual target market share $f_i(t)$ is redistributed proportionally among surviving incumbents so that $\sum_{j \in \mathcal{N}_{t+1}} f_j(t+1) = 1$. 