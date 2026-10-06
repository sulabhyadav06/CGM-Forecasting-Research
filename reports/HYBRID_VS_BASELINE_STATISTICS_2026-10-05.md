# Hybrid vs Baseline Statistical Comparison

Paired Wilcoxon signed-rank tests were performed across the 12 OhioT1DM patients at each forecast horizon, comparing the proposed hybrid model against each baseline architecture.

Two-sided tests were used for MAE and RMSE.

## Interpretation

The hybrid model shows statistically significant differences against several simpler architectures, particularly TCN and GRU at shorter horizons. However, comparisons against the stronger hybrid architectures are generally not statistically significant.

Therefore, the results do **not** support a claim that the proposed hybrid model universally outperforms all competing architectures.

The strongest defensible conclusion is that the proposed model is competitive across multiple horizons, with statistically significant improvements over selected simpler baselines, while performance differences versus stronger hybrid architectures are generally not significant.

The 12-patient paired analysis should be treated as exploratory because of the small sample size and multiple comparisons.
