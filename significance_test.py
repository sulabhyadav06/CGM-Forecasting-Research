import pandas as pd
from scipy.stats import wilcoxon

full_df = pd.read_csv("output/full_results_by_patient.csv")

for horizon in ["30min", "60min"]:
    print(f"\n=== Horizon: {horizon} ===")
    a = full_df[(full_df.Model == "A: Glucose only") & (full_df.Horizon == horizon)].sort_values("Patient")["RMSE"].values
    b = full_df[(full_df.Model == "B: Glucose + HeartRate") & (full_df.Horizon == horizon)].sort_values("Patient")["RMSE"].values
    c = full_df[(full_df.Model == "C: Glucose + Carbs(meals)") & (full_df.Horizon == horizon)].sort_values("Patient")["RMSE"].values
    d = full_df[(full_df.Model == "D: Glucose + HR + Steps + Carbs + Sleep") & (full_df.Horizon == horizon)].sort_values("Patient")["RMSE"].values

    print(f"A mean RMSE: {a.mean():.2f} | D mean RMSE: {d.mean():.2f}")
    stat_ad, p_ad = wilcoxon(a, d)
    print(f"A vs D: statistic={stat_ad:.3f}, p-value={p_ad:.4f}")

    stat_ac, p_ac = wilcoxon(a, c)
    print(f"A vs C: statistic={stat_ac:.3f}, p-value={p_ac:.4f}")

    stat_ab, p_ab = wilcoxon(a, b)
    print(f"A vs B: statistic={stat_ab:.3f}, p-value={p_ab:.4f}")