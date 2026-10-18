import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.friberg_model import FribergANCModel  # noqa: E402
from src.tumor_model import GompertzTumorModel  # noqa: E402


def check_friberg_and_tumor():
    # 1. FRIBERG BOUND HITS CHECK
    params_path = os.path.join(BASE_DIR, "data", "patient_friberg_parameters.csv")
    if not os.path.exists(params_path):
        print("data/patient_friberg_parameters.csv not found. Please run scripts/run_multi_seed_benchmark.py first.")
        return

    params_df = pd.read_csv(params_path)
    lower_b = np.array(FribergANCModel.LOWER_BOUNDS)
    upper_b = np.array(FribergANCModel.UPPER_BOUNDS)
    param_names = ["mtt", "recovery_rate", "drug_slope"]

    hit_counts = {p: {"lower": 0, "upper": 0} for p in param_names}
    total_pts = len(params_df)

    for _, row in params_df.iterrows():
        vals = [
            row["mtt"],
            row.get("raw_recovery_rate", row["recovery_rate"]),
            row["drug_slope"],
        ]
        for idx, (p, v) in enumerate(zip(param_names, vals)):
            lb, ub = lower_b[idx], upper_b[idx]
            if np.isclose(v, lb, atol=1e-3) or v <= lb:
                hit_counts[p]["lower"] += 1
            if np.isclose(v, ub, atol=1e-3) or v >= ub:
                hit_counts[p]["upper"] += 1

    print("=== 1. FRIBERG PARAMETER BOUND HITS CHECK ===")
    print(f"Total Patients Analyzed: {total_pts}")
    for p in param_names:
        l_hits = hit_counts[p]["lower"]
        u_hits = hit_counts[p]["upper"]
        print(f"  {p}: Lower bound hits = {l_hits}/{total_pts} ({l_hits/total_pts:.1%}), Upper bound hits = {u_hits}/{total_pts} ({u_hits/total_pts:.1%})")

    # 2. MONOTONICITY CHECK: Nadir rises with delay
    c_df = pd.read_csv(os.path.join(BASE_DIR, "data", "clinical_logs.csv"))
    t_df = pd.read_csv(os.path.join(BASE_DIR, "data", "timing_eval.csv"))
    friberg_engine = FribergANCModel()

    delays = [0, 3, 7, 14]
    total_pairs = 0
    monotonic_pairs = 0

    for (pid, cycle), group in t_df.groupby(["patient_id", "cycle"]):
        d_day = group["planned_day"].iloc[0]
        p_params = params_df[params_df["patient_id"] == pid]
        if p_params.empty:
            continue
        row = p_params.iloc[0]
        p_history = c_df[(c_df["patient_id"] == pid) & (c_df["day_index"] <= d_day)]

        forecasts = friberg_engine.forecast_anc_nadir(
            base_anc=row["base_anc"],
            mtt=row["mtt"],
            recovery_rate=row["recovery_rate"],
            candidate_delays=delays,
            dose_mg=100.0,
            drug_slope=row["drug_slope"],
            horizon_days=21,
            patient_history=p_history,
            decision_day=d_day,
            num_mc_samples=0,
        )
        nadirs = [forecasts[d]["predicted_nadir_anc"] for d in delays]
        for i in range(len(nadirs) - 1):
            total_pairs += 1
            if nadirs[i+1] >= nadirs[i] - 1e-4:
                monotonic_pairs += 1

    print("\n=== 2. MONOTONICITY AUDIT (NADIR RISES WITH DELAY) ===")
    print(f"Total delay step transitions (0->3d, 3->7d, 7->14d): {total_pairs}")
    print(f"Monotonic pairs (nadir rises or recovers with delay): {monotonic_pairs}/{total_pairs} ({monotonic_pairs / total_pairs:.2%})")
    print(f"Target Threshold: >= 95.0% | Status: {'PASSED' if (monotonic_pairs / total_pairs) >= 0.95 else 'FAILED (Retain Experimental Label)'}")

    # 3. TUMOR MODEL RE-RUN VERIFICATION
    print("\n=== 3. TUMOR MODEL RE-RUN BENCHMARK ===")
    tumor_engine = GompertzTumorModel()
    mape_res = tumor_engine.evaluate_forecast_mape(c_df, decision_day=21)
    cov_res = tumor_engine.evaluate_empirical_coverage(c_df)

    print(f"Forecast MAPE (Decision Day D=21, {mape_res['num_test_patients']} test patients):")
    print(f"  Patient-specific Gompertz MAPE: {mape_res['patient_specific_mape_mean']:.2f}% +/- {mape_res['patient_specific_mape_std']:.2f}%")
    print(f"  Population-average Gompertz MAPE: {mape_res['population_average_mape_mean']:.2f}% +/- {mape_res['population_average_mape_std']:.2f}%")
    print(f"Empirical Coverage of 90% Uncertainty Band ({cov_res['total_evaluations']} forward evaluations):")
    for d in [0, 3, 7, 14]:
        print(f"  Delay +{d} days: {cov_res[f'coverage_delay_{d}']:.1%} (N={cov_res['sample_counts'][d]})")
    print(f"  Overall Empirical Coverage: {cov_res['overall_coverage']:.1%} (Target nominal: 90.0%)")


if __name__ == "__main__":
    check_friberg_and_tumor()
