import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.generate_data import generate_synthetic_data  # noqa: E402
from src.features import build_toxicity_dataset, get_feature_sets  # noqa: E402
from src.train_toxicity import repeated_patient_cross_validation_with_paired_diff  # noqa: E402


def evaluate_single_run(kappa: float, seed: int, tmp_dir: str):
    # Check if pre-existing benchmark data for kappa=1.0 exists
    benchmark_dir = "data" if seed == 42 else (f"data/seed_{seed}" if os.path.exists(f"data/seed_{seed}") else f"scratch/benchmark_data_seed_{seed}")
    if kappa == 1.0 and os.path.exists(os.path.join(benchmark_dir, "processed_dataset.csv")):
        dataset_df = pd.read_csv(os.path.join(benchmark_dir, "processed_dataset.csv"))
    else:
        os.makedirs(tmp_dir, exist_ok=True)
        generate_synthetic_data(
            num_patients=150,
            days=90,
            seed=seed,
            output_dir=tmp_dir,
            coupling_kappa=kappa,
        )

        profiles_df = pd.read_csv(os.path.join(tmp_dir, "patient_profiles.csv"))
        clinical_df = pd.read_csv(os.path.join(tmp_dir, "clinical_logs.csv"))
        wearable_df = pd.read_csv(os.path.join(tmp_dir, "wearable_streams.csv"))
        timing_df = pd.read_csv(os.path.join(tmp_dir, "timing_eval.csv"))

        dataset_df = build_toxicity_dataset(
            profiles_df, clinical_df, wearable_df, timing_df=timing_df
        )

    feature_set_a, feature_set_b = get_feature_sets()

    # Identical 75/25 split as multi-seed benchmark
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, _ = next(
        splitter.split(
            dataset_df,
            dataset_df["severe_toxicity_label"],
            groups=dataset_df["patient_id"],
        )
    )
    train_data = dataset_df.iloc[train_idx]

    # Identical repeated 5x5 Group CV as benchmark
    cv_res = repeated_patient_cross_validation_with_paired_diff(
        dataset_df=train_data,
        feature_set_a=feature_set_a,
        feature_set_b=feature_set_b,
        model_class=LogisticRegression,
        model_kwargs={"random_state": seed, "max_iter": 1000},
        scale=True,
    )

    return cv_res["mean_auc_a"], cv_res["mean_auc_b"], cv_res["paired_diff_mean"]


def run_coupling_stress_test(
    kappas=None,
    seeds=None,
    output_png: str = "docs/figures/coupling_sensitivity_stress_test.png",
):
    if kappas is None:
        kappas = [0.0, 0.25, 0.50, 0.75, 1.0]
    if seeds is None:
        seeds = [42, 101, 2024, 7, 999]

    out_png_full = os.path.join(BASE_DIR, output_png)
    os.makedirs(os.path.dirname(out_png_full), exist_ok=True)
    tmp_base = os.path.join(BASE_DIR, "data", "coupling_eval_tmp")

    summary_records = []

    print(f"Running Coupling Sensitivity Stress Test across {len(kappas)} kappas and {len(seeds)} seeds...")
    for k in kappas:
        seed_diffs = []
        seed_aucs_a = []
        seed_aucs_b = []
        for s in seeds:
            tmp_dir = os.path.join(tmp_base, f"k_{k}_s_{s}")
            auc_a, auc_b, diff = evaluate_single_run(k, s, tmp_dir)
            seed_aucs_a.append(auc_a)
            seed_aucs_b.append(auc_b)
            seed_diffs.append(diff)
            shutil.rmtree(tmp_dir, ignore_errors=True)

        mean_diff = float(np.mean(seed_diffs))
        std_diff = float(np.std(seed_diffs, ddof=1))
        ci_diff = 1.96 * std_diff / np.sqrt(len(seeds))
        mean_auc_a = float(np.mean(seed_aucs_a))
        mean_auc_b = float(np.mean(seed_aucs_b))

        print(
            f"Kappa = {k:.2f} | AUC A = {mean_auc_a:.4f} | AUC B = {mean_auc_b:.4f} | "
            f"Lift = {mean_diff:+.4f} +/- {ci_diff:.4f} (std={std_diff:.4f})"
        )
        summary_records.append(
            {
                "coupling_kappa": k,
                "mean_auc_a": mean_auc_a,
                "mean_auc_b": mean_auc_b,
                "lift_mean": mean_diff,
                "lift_ci": ci_diff,
                "lift_std": std_diff,
            }
        )

    # Clean up tmp directory
    shutil.rmtree(tmp_base, ignore_errors=True)

    df = pd.DataFrame(summary_records)
    csv_path = os.path.join(BASE_DIR, "docs", "figures", "coupling_sensitivity_data.csv")
    df.to_csv(csv_path, index=False)

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Shaded zero-gain line
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="Zero Wearable Gain (Parity)")

    # Plot Lift as primary bar / errorbar
    k_vals = df["coupling_kappa"].values
    lifts = df["lift_mean"].values
    cis = df["lift_ci"].values

    ax1.errorbar(
        k_vals,
        lifts,
        yerr=cis,
        fmt="-o",
        color="#1f77b4",
        ecolor="#aec7e8",
        elinewidth=2.5,
        capsize=5,
        capthick=1.5,
        markersize=8,
        linewidth=2.2,
        label=r"Wearable Lift $\Delta$ AUROC (B - A) [95% CI]",
    )

    # Add annotations at extremes
    ax1.annotate(
        f"$\\kappa=0.0$: No Coupling\n$\\Delta = {lifts[0]:+.4f}$ (Noise Penalty)",
        xy=(k_vals[0], lifts[0]),
        xytext=(k_vals[0] + 0.05, lifts[0] + 0.003),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontweight="semibold",
        color="#d62728",
    )
    ax1.annotate(
        f"$\\kappa=1.0$: Full Upstream Coupling\n$\\Delta = {lifts[-1]:+.4f}$ (Benchmark Match)",
        xy=(k_vals[-1], lifts[-1]),
        xytext=(k_vals[-1] - 0.32, lifts[-1] - 0.005),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontweight="semibold",
        color="#2ca02c",
    )

    ax1.set_ylim(-0.015, 0.025)

    ax1.set_xlabel(r"Autonomic-Marrow Coupling Strength ($\kappa$)", fontsize=12, fontweight="semibold")
    ax1.set_ylabel("Paired AUROC Gain (Ablation B - Ablation A)", fontsize=12, fontweight="semibold", color="#1f77b4")
    ax1.set_title("Wearable Multimodal Lift vs. Biological Coupling Strength ($\kappa$)\n5-Seed Repeated Group CV Stress Test", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(k_vals)
    ax1.set_xticklabels([f"{k:.2f}" for k in k_vals])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png_full, dpi=300)
    plt.close()
    print(f"Saved stress test figure to: {out_png_full}")
    print(f"Saved numerical data to: {csv_path}")


if __name__ == "__main__":
    run_coupling_stress_test()
