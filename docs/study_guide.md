# Study guide

> **Repository layout note.** This document was written in the development workspace and carried over unchanged
> in substance. File paths have been updated to this repository's layout (`reports/`, `notebooks/`, `scripts/`,
> `artifacts/`, `course/presentations/`); dated log entries describe work done in that workspace. Notebook section
> numbers refer to `notebooks/walkthrough.ipynb`.

This guide accompanies `walkthrough.ipynb` and `reports/research_report.pdf`. It assumes fluency in Bayesian inference and
numerical methods and explains finance terms from scratch. Sections:

1. Finance vocabulary you need
2. Essential derivations
3. Research decisions and why they were made
4. Answers to the notebook exercises
5. Likely interview questions with model answers
6. Numbers worth remembering

---

## 1. Finance vocabulary

| Term | Meaning in this project |
|---|---|
| **ETF** | Exchange-traded fund: a share that tracks a basket of assets (SPY: S&P 500 equities; QQQ: Nasdaq-100; TLT: 20+-year US Treasury bonds; GLD: gold). Trades like a stock. |
| **Close / closing price** | Last traded price of the day. Signals and trades both use closes. |
| **Adjusted price** | Price history rescaled for dividends and splits so that $P^{adj}_t/P^{adj}_{s}$ equals the total growth of \$1 invested at $s$ with dividends reinvested. Needed so that a dividend is not mistaken for a loss. |
| **Log vs simple return** | $r_t=\log(P_t/P_{t-1})$ (additive over time, used in the model); $R_t=e^{r_t}-1$ (what wealth earns, used in the portfolio). |
| **Volatility** | Standard deviation of returns. Annualised as $\sqrt{252}\times$ daily, because there are about 252 trading days per year. 10% annual ≈ 0.63% daily. |
| **Volatility clustering** | Large moves tend to follow large moves (in either direction). The reason $r_t^2$ is autocorrelated although $r_t$ is not. |
| **Risk-free rate / cash** | What uninvested money earns. Proxied here by DFF, the effective federal funds rate: the overnight rate at which US banks lend reserves to each other. Quoted in annual percent. |
| **ACT/360** | Money-market interest convention: interest = rate × (calendar days)/360. A Friday→Monday holding period earns 3 days. |
| **Basis point (bp)** | 0.01%. Transaction cost here: 5 bp per unit of turnover. |
| **Turnover** | Fraction of wealth traded at a rebalance: $\lvert w_t-\tilde w_t\rvert$, where $\tilde w_t$ is the weight after drifting with returns. |
| **Exposure** | The fraction $w_t$ of wealth in the risky asset. Average exposure ≈ how much risk the strategy takes. |
| **Leverage / shorting** | Holding $w>1$ (borrowing) or $w<0$ (betting on a fall). Both excluded here. |
| **Volatility targeting** | Choosing $w_t$ so that forecast portfolio volatility equals a target: $w_t=\min(1,\sigma^\ast/\hat\sigma_t)$. |
| **Excess return** | Portfolio return minus the cash return over the same period. |
| **Sharpe ratio** | $\sqrt{252}\,\overline{\text{excess}}/\operatorname{sd}(\text{excess})$: return per unit of risk. Invariant to scaling the position (ignoring the cap and costs). |
| **Drawdown / max drawdown** | Loss from the running peak of wealth, $W_t/\max_{s\le t}W_s-1$. The maximum drawdown is its minimum over the sample. Roughly proportional to exposure. |
| **Sortino ratio / downside deviation** | Like Sharpe, but only negative excess returns count as risk. |
| **VaR / ES** | Value-at-risk at level α: the loss exceeded with probability α. Expected shortfall: the mean loss given that the VaR is exceeded. |
| **Back-test** | Simulating a trading rule on historical data using only information available at each decision time. |
| **Look-ahead bias** | Accidentally using information not yet available at the decision time. It inflates back-tests. |
| **Out-of-sample / holdout** | Data not used to design or tune anything. |

---

## 2. Essential derivations

### 2.1 GARCH(1,1) moments
Model: $r_t=\sqrt{h_t}z_t$, $z_t\sim N(0,1)$ iid, $h_t=\omega+\alpha r_{t-1}^2+\beta h_{t-1}$.

- $E[r_t^2\mid\mathcal F_{t-1}]=h_t$. Since $E[r_{t-1}^2]=E[h_{t-1}]$, we get $E[h_t]=\omega+(\alpha+\beta)E[h_{t-1}]$, so under stationarity $V=E[h_t]=\omega/(1-p)$ with $p=\alpha+\beta<1$.
- Multi-step forecast: $E[h_{t+1}\mid\mathcal F_{t-1}]=\omega+\alpha E[r_t^2\mid\mathcal F_{t-1}]+\beta h_t=\omega+p\,h_t$. By induction, $E[h_{t+k}\mid\mathcal F_t]-V=p^{k-1}(h_{t+1}-V)$.
- The half-life of a variance shock is $\log(1/2)/\log p$. For $p=0.99$ that is 69 days; for $p=0.95$, 14 days.
- ARMA representation: with $\nu_t=r_t^2-h_t$ (a martingale difference), $r_t^2=\omega+p\,r_{t-1}^2+\nu_t-\beta\nu_{t-1}$, an ARMA(1,1) in $r_t^2$. This explains the slowly decaying autocorrelation of $r_t^2$.
- The fourth moment exists iff $3\alpha^2+2\alpha\beta+\beta^2<1$. Even Gaussian GARCH has fat-tailed *unconditional* returns; but the *standardised* returns $r_t/\sqrt{h_t}$ should be $N(0,1)$, and empirically they are not (excess kurtosis 0.4–2.9).

### 2.2 Reparametrisation
$u=(\log V,\operatorname{logit}p,\operatorname{logit}q)$ with $q=\alpha/p$ is a bijection from $\mathbb R^3$ onto
$\{\omega>0,\alpha>0,\beta>0,\alpha+\beta<1\}$, with $\omega=V(1-p)$, $\alpha=pq$, $\beta=p(1-q)$.
- The constraints become unconstrained optimisation and sampling problems.
- $V$ is better identified than $\omega$, and logit $p$ stretches the region near $p=1$, where the data often put mass.
- The prior is specified as a density *in $u$*, so the posterior in $u$ needs no Jacobian. Transforming the same prior to $(\omega,\alpha,\beta)$ would require the Jacobian $|\partial u/\partial(\omega,\alpha,\beta)|$.
- MAP estimates are not invariant to reparametrisation: the mode in $u$ differs from the mode in $(\omega,\alpha,\beta)$.

### 2.3 Gaussian log-likelihood
By the prediction-error decomposition, $p(r_{1:n}\mid u)=\prod_t p(r_t\mid\mathcal F_{t-1},u)$. Hence
$$\ell(u)=-\frac12\sum_{t=1}^n\Big[\log2\pi+\log h_t(u)+\frac{r_t^2}{h_t(u)}\Big],\qquad h_1=V.$$
The initialisation $h_1=V$ has an effect that decays like $p^t$; it is negligible after a few hundred observations unless $p\approx1$.

### 2.4 Posterior-mean plug-in vs integrated forecast vs predictive density
- Plug-in: $h_t(\bar\theta)$. Integrated: $\bar h_t=E_\pi[h_t(\theta)]$. By Jensen they differ whenever $h_t(\cdot)$ is non-linear.
- Predictive density: $p(r_t\mid\mathcal F_{t-1})=E_\pi[N(r_t;0,h_t(\theta))]$, a scale mixture of normals. Its variance is $E_\pi[h_t(\theta)]=\bar h_t$, and its kurtosis is $3E[H^2]/E[H]^2=3(1+\mathrm{CV}^2)$, where $H=h_t(\theta)$ under the posterior.
- QLIKE depends on the forecast only through its variance, so plug-in vs integrated is compared on QLIKE. The log score sees the full shape, so it rewards the mixture's slightly heavier tails.

### 2.5 Why QLIKE (and MSE) are "robust" losses
Patton (2011) shows that a loss ranks forecasts consistently even when the true variance is replaced by a
conditionally unbiased proxy such as $r^2$ iff it belongs to a Bregman-type family generated by a convex function.
QLIKE and MSE belong to it, and QLIKE is the member whose loss differences are invariant to the scale of returns.
$E[\log h+r^2/h\mid\mathcal F]=\log h+\sigma^2/h$ is minimised at $h=\sigma^2$. Losses such as MSE on $\log r^2$ are *not* robust.

### 2.6 Diebold–Mariano with HAC
$d_t=L_t^A-L_t^B$; under $H_0:E[d_t]=0$, $\sqrt n\,\bar d/\hat\sigma_{LR}\to N(0,1)$ with
$\hat\sigma^2_{LR}=\hat\gamma_0+2\sum_{k=1}^{L}(1-\tfrac{k}{L+1})\hat\gamma_k$ (Bartlett / Newey–West), $L=\lfloor4(n/100)^{2/9}\rfloor$.
The HAC correction matters because loss differentials inherit volatility clustering.

### 2.7 Laplace (classical) parameter uncertainty
$\hat u\stackrel{a}{\sim}N(u_0,\mathcal I(u_0)^{-1})$, with $\hat{\mathcal I}$ the observed information (the finite-difference Hessian of $-\ell$ at $\hat u$).
Drawing $u^{(m)}\sim N(\hat u,\hat{\mathcal I}^{-1})$ and averaging $N(r;0,h(u^{(m)}))$ gives a "frequentist predictive" mixture.
It approximates the Bayesian posterior predictive with a flat prior in $u$ when the likelihood is near-Gaussian. It breaks down at boundaries.

### 2.8 Volatility targeting and accounting
- Weight: $w_t=\min(1,\sigma^\ast/\sqrt{252h_t})$ with $\sigma^\ast=0.10$.
- Drifted weight after period $t$: $\tilde w_{t+1}=w_t(1+R^a_t)/(1+R^p_t)$.
- Net return: $(1-c\,|w_t-\tilde w_t|)(1+R^p_t)-1$, where $R^p_t=w_tR^a_t+(1-w_t)R^f_t$ and $c=5\times10^{-4}$.
- If returns were exactly $N(0,h_t)$ and $h_t$ known, portfolio volatility would equal $\sigma^\ast$ whenever $w_t<1$.

### 2.9 Why constant de-risking reduces drawdown but not Sharpe
If excess returns scale as $e'_t=c\,e_t$ (with $c\le1$, no cap binding, costs ignored), then
$\mathrm{Sharpe}(e')=\mathrm{Sharpe}(e)$, $\mathrm{vol}(e')=c\,\mathrm{vol}(e)$ and drawdowns shrink roughly in proportion.
The posterior 90th-percentile rule is $w^{(90)}_t=w^{\text{point}}_t\sqrt{h^{\text{point}}_t/q_{0.9,t}}$, and the ratio
$\sqrt{h/q_{0.9}}$ is nearly constant (the uncertainty premium has correlation >0.99 with the variance level). So upper-90 ≈ a constant
haircut, and its drawdown advantage is mechanical.

### 2.10 Block bootstrap for Sharpe differences
Paired circular block bootstrap: draw block starts uniformly, take 20 consecutive days (wrapping around), concatenate to length $n$,
and apply the **same** indices to both strategies. This preserves serial dependence and cross-correlation, so the variance of
$\widehat{\Delta S}$ is estimated without a parametric model. (Analytical alternative: Ledoit & Wolf, 2008.)

---

## 3. Research decisions and why

| Decision | Reason |
|---|---|
| GARCH(1,1), zero mean, Gaussian | A transparent, hard-to-beat benchmark with three parameters, so the effect of parameter uncertainty is isolated cleanly. Means are negligible at daily frequency and hard to estimate. |
| Prior on $u$, centred at $p=0.8$, sd 1.5 | Weakly informative and proper; regularises the near-integrated MLE. Chosen before the market evaluation, never tuned on test results. Sensitivity was checked with a broad prior. |
| Adaptive RWM, adaptation frozen after warm-up | Valid MCMC (the retained chain has a fixed kernel). Three parameters make RWM adequate. Strict R-hat/ESS rule. |
| 300 retained draws per refit | Keeps the forecast Monte Carlo error small relative to posterior spread. The mean and q90 of $h$ are stable across seeds to a few %. |
| Quarterly (63-day) refits, 500/1000-day windows | Standard practice. Two windows test sensitivity to the bias–variance trade-off in estimation. |
| Keep the MLE bound at logit $p\le7$ | Changing it changes the classical estimator. Its effect was *measured* instead (≈5% variance changes in 9% of refits). |
| QLIKE + log score | QLIKE is robust for variance forecasts with a noisy proxy; the log score is proper for densities and is the right score for "uncertainty" claims. |
| EWMA benchmark | Zero-parameter industry baseline (RiskMetrics). Tells you whether estimation is worth it at all. |
| Student-$t$ GARCH | Checks whether the *model* is the bigger problem than *parameter uncertainty*. It is. |
| Laplace mixture | Separates "acknowledging estimation uncertainty" from "doing Bayesian inference". |
| One-close execution delay, lagged DFF | Removes any possibility of trading on information not yet available. |
| Implementable exposure control `mle_const_rt` | The ex-post matched controls of earlier milestones used test-period information. A trailing, past-only haircut is tradable and still removes the conservatism effect. |
| Pooled four-ETF tests and an equal-weight sleeve | More power, fewer tests. Windows of one asset are dependent, so per-configuration tests are Holm-adjusted. |
| 2017–22 as the primary evidence | 2023–26 had been inspected repeatedly by earlier milestones; 2017–22 had not, and specs were frozen before evaluation. |
| No new strategy search in the course | Avoids multiple-testing and garden-of-forking-paths problems. The earlier search is preserved, with its pre-registered rules, in `agent_research`. |

---

## 4. Answers to notebook exercises

The exercise labels date from an earlier notebook layout and were kept. In the current 16-section notebook (reorganised around H1 → H2 → H3 on 2026-11-06): Q A1 is in §2, Q B1 in §3, Q C1 in §4, Q D1 in §5, Q E1 in §6, Q E2 in §10, Q F1–F2 in §13, and Q H1–H3 are in §11.

**Q A1.** If $h_t$ uses $r_t$, the forecast "knows" today's shock: on large-move days it is large exactly when
$r_t^2$ is large, so QLIKE falls artificially. In a back-test the weight would be cut *before* the crash day it
is supposed to forecast, avoiding losses no real investor could avoid. Both forecast skill and returns are inflated.

**Q B1.** With $p=1$: $E[h_{t+k}\mid\mathcal F_t]=h_{t+1}+(k-1)\omega$, which never reverts (with $\omega=0$ it is flat
at $h_{t+1}$). $V=\omega/(1-p)\to\infty$ for fixed $\omega>0$, so in the $u$-parametrisation $\log V\to\infty$
unless $\omega\to0$ jointly. Near $p=1$, $V$ is weakly identified, which is one reason the likelihood ridge runs toward the bound.

**Q C1.** On a bound the score is not zero and the estimator's sampling distribution is a truncated (half-normal-like)
mixture, not Gaussian. The curvature-based covariance describes a quadratic approximation around a point that is not
an interior maximum, so Wald intervals and Laplace draws misrepresent uncertainty in the direction of the bound.

**Q D1.** Not wrong, but informative in the opposite direction to the data. With $n=500$ the posterior mean of $p$
was 0.983 vs MLE 0.992 and prior centre 0.80. The data dominate, but the prior still pulls the tail near $p=1$
in, which is visible as the left-skewed posterior. Under the broad prior ($\operatorname{logit}p\sim N(0,3^2)$) some refits
became bimodal, which shows that prior choice matters when the likelihood is flat.

**Q E1.** $E[r^4]=E[E[r^4\mid H]]=E[3H^2]$ and $E[r^2]=E[H]$, so $\operatorname{kurt}=3E[H^2]/E[H]^2=3(1+\operatorname{Var}H/E[H]^2)$.

**Q E2.** QLIKE depends only on the forecast variance, and integration hardly changes it, since $E[H]$ is close to
$h(\bar\theta)$. The log score also rewards the *shape*: the mixture's slightly heavier tails assign more
density to large standardised moves, which Gaussian plug-ins under-predict. A small but systematic gain on a
few tail days is enough to be statistically detectable over 1,500 days.

**Q F1.** With the cap, $\min(1,c\,w)$ is not proportional to $w$ when $w$ is near 1, so periods of low volatility
(where $w$ is capped) are scaled differently from high-volatility periods. Turnover costs also scale with $c$.
Both change the excess-return distribution beyond a pure scaling.

**Q F2.** Its multiplier was chosen to match the *test-period* average exposure, which is not known in advance:
that is look-ahead. `mle_const_rt` uses only the trailing 252 days of the ratio $\sqrt{h^{MLE}/q_{0.9}}$, which is known
at each decision, so it is a feasible strategy with the same average conservatism.

---

## 5. Likely interview questions (with model answers)

1. **Why GARCH(1,1) rather than a richer model?** It captures most of the forecastable variance dynamics in daily
   data, has three parameters, and is a standard benchmark (Hansen & Lunde, 2005). For this question it isolates
   parameter uncertainty cleanly. Richer models (GJR for leverage, Student-$t$, realised-volatility inputs) are
   natural extensions.
2. **How do you evaluate a volatility forecast when the true volatility is unobservable?** Use a conditionally unbiased
   proxy ($r_t^2$, or realised variance) with a loss function robust to proxy noise (QLIKE or MSE, per Patton 2011),
   and test differences with Diebold–Mariano using HAC standard errors.
3. **What does the Bayesian approach buy you here, concretely?** Shrinkage of an often near-boundary point estimate,
   and a predictive density that accounts for estimation uncertainty. Empirically: about +0.003 nats/day in log score,
   no reliable QLIKE gain, no portfolio gain after exposure control.
4. **Your Bayesian strategy had a smaller drawdown. Why don't you claim it is better?** It held about 4 points less
   equity. Any constant reduction in exposure reduces drawdown about proportionally while leaving Sharpe
   unchanged. Against an implementable exposure-matched control the difference vanishes.
5. **How did you avoid look-ahead bias?** Forecasts use $r_{<t}$; parameters are fitted only on past windows; there is
   one extra close of execution delay; the cash rate is lagged; benchmarks are calibrated on pre-period data; and
   canary tests perturb a future return and check that no earlier forecast or weight changes.
6. **How do you know the MCMC is right?** Rank-normalised R-hat and bulk/tail ESS thresholds, over-dispersed starts,
   adaptation frozen before sampling, and agreement with an independent importance sampler to 0.3%. Monte Carlo
   variability is reported explicitly: across three seeds, Sharpe varies by up to 0.01.
7. **What was the most important robustness check?** Student-$t$ innovations. They improved the log score about 5–6× as much as
   posterior integration in 2017–22 (3.6–7.5× across all four period–window cells), which shows that model misspecification, not parameter uncertainty, dominates.
8. **How much data would you need to detect a 0.02 Sharpe difference?** Use the empirical uncertainty. For the
   four-ETF sleeve over six years, the 95% bootstrap intervals for upper-90 minus the control have half-widths of
   about 0.02–0.03, i.e. a standard error of about 0.013. Detecting 0.02 with a 5% two-sided test and 80% power
   needs SE ≈ 0.02/2.8 ≈ 0.007. Since SE scales like $1/\sqrt{T}$, that requires about $(0.013/0.007)^2\approx3.5$
   times more data: roughly 20 years under the same conditions. The textbook iid approximation
   $\operatorname{se}(\Delta S)\approx\sqrt{2(1-\rho)/T_{\text{years}}}$ would be optimistic here. The two strategies'
   daily returns are 99.7–99.99% correlated, but fat tails and volatility clustering inflate the true uncertainty.
9. **What is the difference between the posterior-mean plug-in and the posterior predictive?** The first is a point
   estimate inserted into the model; the second integrates the conditional density over the posterior. They differ by
   Jensen's inequality and in shape (scale mixture).
10. **What would you do next?** A Bayesian Student-$t$/GJR model (more weakly identified parameters, so possibly
    more value from integration), more frequent posterior updates, realised-volatility proxies for sharper
    evaluation, and a longer, multi-asset sample for power.
11. **Explain the multiple-testing safeguards.** Pre-registered hypotheses and splits, pooled tests across assets,
    Holm adjustment across configurations, and no tuning on evaluation periods. The exploratory strategy search had
    a validation gate and a stop rule, which were triggered.
12. **What is the role of transaction costs here?** Small for GARCH rules (daily turnover of 2–5% at 5 bp), but they
    penalise noisy rules. In the exploratory search, an apparent gain from one candidate disappeared at 20 bp.

---

## 6. Numbers worth remembering

| Quantity | Value |
|---|---|
| Representative fit (SPY, 500 days to 2022-12-30): MLE $p$ / posterior mean $p$ | 0.9918 / 0.9829 |
| Posterior CV of next-day variance (typical) | 7–12% |
| Upper-90 "uncertainty premium" $q_{0.9}/\bar h-1$ | mean 0.09–0.15 |
| Log-score gain, predictive mixture vs MLE (pooled) | +0.0023 to +0.0052 nats/day ($p\le0.024$) |
| Same for Student-$t$ vs MLE | +0.017 to +0.029 nats/day |
| QLIKE, integrated − MLE (pooled) | −0.0009 to −0.0056, $p=0.08$–$0.48$ |
| Upper-90 vs implementable control, ΔSharpe (sleeve) | −0.008 to +0.009, all CIs contain 0 |
| Upper-90 exposure vs MLE | −3.6 / −4.1 percentage points (2017–22 / 2023–26) |
| 1% VaR violation rate, Gaussian vs Student-$t$ (2017–22) | ≈2.1% vs 1.5% |
| Refits failing strict diagnostics (course grid) | 4 of 320 |
| MLE on the persistence bound (course grid) | 30 of 320 |
| Seed-to-seed range of upper-90 Sharpe (per config) | up to 0.010 |

---

## 7. Model adequacy (notebook §11, report Appendix A)

### 7.1 Diagnostics and what each one tests
| Diagnostic | Null hypothesis it checks | Result here (2017–22, 500-day window) |
|---|---|---|
| $\overline{z^2}$ with $z_d=r_d/\sqrt{h_d}$ | variance level calibrated, $E[z^2]=1$ | 1.04–1.12: variance slightly under-predicted |
| Ljung–Box on $z$: $Q(m)=n(n+2)\sum_{k\le m}\hat\rho_k^2/(n-k)\approx\chi^2_m$ | no linear predictability left | not rejected ($p>0.1$ for GARCH models) |
| Ljung–Box on $z^2$ (McLeod–Li) | no volatility clustering left | equities borderline/OK; **TLT, GLD rejected** |
| Excess kurtosis / skewness of $z$ | conditional distribution is $N(0,1)$ | kurtosis 1.9–2.9 (equities, gold); skew −0.67 to −0.85 (equities) |
| PIT $u_d=F_d(r_d)$, histogram and KS | full predictive CDF calibrated, $u\sim U(0,1)$ | rejected for every equity model, including Student-$t$ |
| Normal scores $\Phi^{-1}(u_d)$ (Berkowitz) | $N(0,1)$ iid | Gaussian: fat tails; mixture: almost unchanged; Student-$t$: kurtosis ≈0, but asymmetric |
| Quantile exceedances $\Pr(u<\alpha)$, $\Pr(u>1-\alpha)$ | each tail calibrated | lower tail too frequent, upper tail too rare for equities |

**Caveats.** These are out-of-sample tests on rolling forecasts. The textbook $\chi^2$ and KS null distributions
ignore parameter-estimation error and the dependence created by overlapping fit windows, so treat $p$-values as
approximate. In-sample Ljung–Box tests on GARCH residuals also need degrees-of-freedom corrections (Li & Mak, 1994).

### 7.2 Standardising by variance vs calibrating with the CDF
For a Gaussian forecast, $\Phi^{-1}(F_d(r_d))=r_d/\sqrt{h_d}$ exactly, so the two notions coincide. For the Bayesian
mixture they do not:
- $z_d=r_d/\sqrt{\bar h_d}$ only checks the second moment;
- $\Phi^{-1}\big(M^{-1}\sum_m\Phi(r_d/\sqrt{h^{(m)}_d})\big)$ also credits the mixture's heavier tails.

The difference is small here (SPY excess kurtosis 2.91 vs 2.70) because posterior variance dispersion is small.

### 7.3 Why clustering is captured but extremes are not
- $\operatorname{kurt}(r)=\operatorname{kurt}(z)\,E[h^2]/E[h]^2$. GARCH explains the second factor (time-varying scale)
  and removes the autocorrelation of $z^2$. It is silent about $\operatorname{kurt}(z)$.
- The remaining tail mass comes from three sources: (i) unanticipated jumps, since $h_d$ uses only $r_{<d}$ and the
  first shock of a cluster is always a large $|z|$; (ii) asymmetry, i.e. the leverage effect and negative skew;
  (iii) not parameter uncertainty, which adds only $3\,\mathrm{CV}^2\approx0.02$–$0.05$ of excess kurtosis.

### 7.4 Answers to the model-adequacy exercises (notebook §11)
**Q H1.** ACF$(z^2)$ tests whether the *scale* dynamics are captured, i.e. whether $E[z_d^2\mid\mathcal F_{d-1}]$ is constant.
Shape tests (kurtosis, PIT) check the *distribution* of $z$ given its scale. An iid Student-$t$ $z$ has zero
autocorrelation in $z^2$ yet is far from Gaussian, so the two properties are independent.

**Q H2.** A symmetric forecast puts mass $\alpha$ in each tail. If the true conditional distribution is negatively skewed
with the right variance, it has more mass far to the left and less far to the right. The same symmetric quantiles
are then breached too often below and too rarely above. Calibrating the variance cannot fix both at once.

**Q H3.** The 21-day realised volatility at day $d$ contains $r_d^2$ (1/21 of its weight) and 20 earlier returns.
It is a lagging, smoothed, partly *contemporaneous* quantity: scoring a one-step forecast against it would reward
forecasts that resemble a trailing average (such as EWMA) and leak the outcome into the target. QLIKE against $r_d^2$
is conditionally unbiased for $h_d$ and consistent for ranking (Patton, 2011).

### 7.5 Interview questions on model adequacy
1. **Your GARCH residuals pass Ljung–Box on $z^2$. Is the model adequate for risk management?** No. That only shows the
   variance dynamics are captured. At the 1% level, the Gaussian VaR was breached on 2.5% of days for SPY. Tail shape and
   asymmetry need separate tests (PIT, Kupiec, Christoffersen).
2. **How do you evaluate a predictive distribution out of sample?** With the PIT histogram and normal scores
   (Diebold, Gunther & Tay 1998; Berkowitz 2001), quantile exceedances with Kupiec and Christoffersen tests, and proper
   scores (log score, CRPS) for comparisons.
3. **When do volatility models fail most?** When a shock arrives from a calm regime (February 2018, February 2020): the
   predicted variance is low exactly when the first large move happens. Here SPY's 1% breaches were about 3% in the
   calm predicted-volatility tercile.
4. **EWMA or GARCH in a crisis?** GARCH reacts at a similar speed but mean-reverts. Its half-life after the 2020 peak was
   9–16 days versus EWMA's 20–33 days, so EWMA stays too high for weeks. EWMA does leave less residual clustering for TLT
   and GLD, which suggests faster forgetting helps when the variance level shifts.
5. **What would you change first?** Asymmetric dynamics (GJR/EGARCH) and a skewed fat-tailed innovation for equities;
   faster adaptation or regime-switching for bonds. These were deliberately not fitted in this project, which diagnoses
   the existing models.

Additional references for this section: Ljung & Box (1978); McLeod & Li (1983); Li & Mak (1994), *J. Time Series
Analysis* 15(6), 627–636; Diebold, Gunther & Tay (1998); Berkowitz (2001). Full citations are in `reports/research_report.pdf`, except
Li & Mak, given here.
