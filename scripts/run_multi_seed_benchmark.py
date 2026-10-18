import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score, precision_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.generate_data import generate_synthetic_data  # noqa: E402
from src.features import (  # noqa: E402
    build_toxicity_dataset,
    build_timing_evaluation_dataset,
    get_feature_sets,
)
from src.train_toxicity import (  # noqa: E402
    repeated_patient_cross_validation_with_paired_diff,
    evaluate_intra_patient_timing_ranking,
)
from src.friberg_model import fit_all_patient_friberg_models  # noqa: E402
from src.evaluate import generate_evaluation_figures_and_report  # noqa: E402


def run_multi_seed_benchmark(seeds=None, coupling_kappa: float = 1.0):
    if seeds is None:
        seeds = [42, 101, 2024, 7, 999]

    feature_set_a, feature_set_b = get_feature_sets()

    seed_results = []
    seed_model_summaries = {
        "Logistic Regression": [],
        "Random Forest": [],
        "Gradient Boosting": [],
    }

    print(f"=== Starting Multi-Seed Benchmark across {seeds} (kappa={coupling_kappa}) ===")

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")
        tmp_dir = "data" if seed == 42 else f"data/seed_{seed}"
        os.makedirs(tmp_dir, exist_ok=True)

        # Generate synthetic data if not already present
        if not (
            os.path.exists(os.path.join(tmp_dir, "patient_profiles.csv"))
            and os.path.exists(os.path.join(tmp_dir, "clinical_logs.csv"))
            and os.path.exists(os.path.join(tmp_dir, "wearable_streams.csv"))
            and os.path.exists(os.path.join(tmp_dir, "timing_eval.csv"))
        ):
            generate_synthetic_data(
                num_patients=150,
                days=90,
                seed=seed,
                output_dir=tmp_dir,
                coupling_kappa=coupling_kappa,
            )

        profiles_df = pd.read_csv(os.path.join(tmp_dir, "patient_profiles.csv"))
        clinical_df = pd.read_csv(os.path.join(tmp_dir, "clinical_logs.csv"))
        wearable_df = pd.read_csv(os.path.join(tmp_dir, "wearable_streams.csv"))
        timing_df = pd.read_csv(os.path.join(tmp_dir, "timing_eval.csv"))

        dataset_df = build_toxicity_dataset(
            profiles_df, clinical_df, wearable_df, timing_df=timing_df
        )
        timing_feat_df = build_timing_evaluation_dataset(
            profiles_df, clinical_df, wearable_df, timing_df
        )

        dataset_df.to_csv(os.path.join(tmp_dir, "processed_dataset.csv"), index=False)
        timing_feat_df.to_csv(os.path.join(tmp_dir, "timing_features.csv"), index=False)

        # Split train/test (75/25 GroupShuffleSplit by patient_id)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, test_idx = next(
            splitter.split(
                dataset_df,
                dataset_df["severe_toxicity_label"],
                groups=dataset_df["patient_id"],
            )
        )

        train_data = dataset_df.iloc[train_idx]
        test_data = dataset_df.iloc[test_idx]
        test_patient_ids = set(test_data["patient_id"].unique())

        y_train = train_data["severe_toxicity_label"].values
        y_test = test_data["severe_toxicity_label"].values

        models_config = {
            "Logistic Regression": {
                "cls": LogisticRegression,
                "kwargs": {"random_state": seed, "max_iter": 1000},
                "scale": True,
            },
            "Random Forest": {
                "cls": RandomForestClassifier,
                "kwargs": {"n_estimators": 100, "max_depth": 5, "random_state": seed},
                "scale": False,
            },
            "Gradient Boosting": {
                "cls": GradientBoostingClassifier,
                "kwargs": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "random_state": seed},
                "scale": False,
            },
        }

        current_seed_summary = []
        for model_name, cfg in models_config.items():
            cv_res = repeated_patient_cross_validation_with_paired_diff(
                dataset_df=train_data,
                feature_set_a=feature_set_a,
                feature_set_b=feature_set_b,
                model_class=cfg["cls"],
                model_kwargs=cfg["kwargs"],
                scale=cfg["scale"],
            )

            X_tr = train_data[feature_set_b].values
            X_te = test_data[feature_set_b].values

            scaler = None
            if cfg["scale"]:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

            model = cfg["cls"](**cfg["kwargs"])
            model.fit(X_tr, y_train)

            test_probs = model.predict_proba(X_te)[:, 1]
            test_auc = roc_auc_score(y_test, test_probs)
            test_brier = brier_score_loss(y_test, test_probs)

            threshold = cv_res["val_tuned_threshold"]
            test_preds = (test_probs >= threshold).astype(int)
            test_rec = recall_score(y_test, test_preds, zero_division=0)
            test_prec = precision_score(y_test, test_preds, zero_division=0)

            tau_res = evaluate_intra_patient_timing_ranking(
                model=model,
                scaler=scaler,
                timing_feat_df=timing_feat_df,
                features=feature_set_b,
                test_patient_ids=test_patient_ids,
            )

            res_entry = {
                "seed": seed,
                "model": model_name,
                "cv_auc_a": cv_res["mean_auc_a"],
                "cv_auc_b": cv_res["mean_auc_b"],
                "paired_diff": cv_res["paired_diff_mean"],
                "paired_diff_ci": cv_res["paired_diff_ci"],
                "test_auc": test_auc,
                "test_brier": test_brier,
                "threshold": threshold,
                "test_recall": test_rec,
                "test_precision": test_prec,
                "tau_mean": tau_res["mean"],
                "tau_std": tau_res["std"],
                "tau_iqr": tau_res["iqr"],
            }
            current_seed_summary.append(res_entry)
            seed_model_summaries[model_name].append(res_entry)

            print(
                f"  [{model_name}] CV B-A: {cv_res['paired_diff_mean']:+.4f} | "
                f"Test AUC: {test_auc:.4f} | Recall: {test_rec:.1%} | Prec: {test_prec:.1%} | "
                f"Tau: {tau_res['mean']:.4f}"
            )

        # Track LR primary metrics for multi-seed overall table
        lr_entry = [e for e in current_seed_summary if e["model"] == "Logistic Regression"][0]
        seed_results.append(
            {
                "seed": seed,
                "cv_auc_a": lr_entry["cv_auc_a"],
                "cv_auc_b": lr_entry["cv_auc_b"],
                "paired_diff": lr_entry["paired_diff"],
                "test_auc": lr_entry["test_auc"],
                "test_brier": lr_entry["test_brier"],
                "threshold": lr_entry["threshold"],
                "test_recall": lr_entry["test_recall"],
                "test_precision": lr_entry["test_precision"],
                "tau_mean": lr_entry["tau_mean"],
            }
        )

    # Multi-seed aggregate stats for Logistic Regression
    lr_df = pd.DataFrame(seed_results)
    print("\n=== MULTI-SEED LOGISTIC REGRESSION RESULTS ===")
    print(lr_df.to_string(index=False))

    agg_table = []
    metrics = [
        ("CV AUROC Ablation A (Clin)", "cv_auc_a", "{:.4f}"),
        ("CV AUROC Ablation B (+Wear)", "cv_auc_b", "{:.4f}"),
        ("Paired Gain (B - A)", "paired_diff", "{:+.4f}"),
        ("Holdout Test AUROC", "test_auc", "{:.4f}"),
        ("Holdout Test Brier Score", "test_brier", "{:.4f}"),
        ("CV-Tuned Threshold", "threshold", "{:.3f}"),
        ("Untouched Test Recall", "test_recall", "{:.2%}"),
        ("Untouched Test Precision", "test_precision", "{:.2%}"),
        ("Timing Ranking Kendall Tau", "tau_mean", "{:.4f}"),
    ]

    ref_row = lr_df[lr_df["seed"] == 42].iloc[0]
    for label, col, fmt in metrics:
        vals = lr_df[col].values
        m = np.mean(vals)
        mi, ma = np.min(vals), np.max(vals)
        ref_v = ref_row[col]
        agg_table.append(
            {
                "Metric": label,
                "5-Seed Mean": fmt.format(m),
                "5-Seed Range": f"{fmt.format(mi)} to {fmt.format(ma)}",
                "Reference Run (Seed 42)": fmt.format(ref_v),
            }
        )

    agg_df = pd.DataFrame(agg_table)
    print("\n=== AGGREGATE 5-SEED TABLE ===")
    print(agg_df.to_markdown(index=False))
    agg_df.to_csv(os.path.join(BASE_DIR, "docs", "figures", "multi_seed_aggregate_table.csv"), index=False)

    # Print Model Comparison multi-seed summary
    print("\n=== MODEL COMPARISON ACROSS 5 SEEDS ===")
    for model_name, entries in seed_model_summaries.items():
        m_df = pd.DataFrame(entries)
        print(f"\n{model_name}:")
        print(f"  CV AUROC A: {m_df['cv_auc_a'].mean():.4f} ({m_df['cv_auc_a'].min():.4f} - {m_df['cv_auc_a'].max():.4f})")
        print(f"  CV AUROC B: {m_df['cv_auc_b'].mean():.4f} ({m_df['cv_auc_b'].min():.4f} - {m_df['cv_auc_b'].max():.4f})")
        print(f"  Paired Diff (B - A): {m_df['paired_diff'].mean():+.4f} ({m_df['paired_diff'].min():+.4f} - {m_df['paired_diff'].max():+.4f})")
        print(f"  Test AUROC: {m_df['test_auc'].mean():.4f} ({m_df['test_auc'].min():.4f} - {m_df['test_auc'].max():.4f})")
        print(f"  Test Recall: {m_df['test_recall'].mean():.2%} ({m_df['test_recall'].min():.2%} - {m_df['test_recall'].max():.2%})")
        print(f"  Test Precision: {m_df['test_precision'].mean():.2%} ({m_df['test_precision'].min():.2%} - {m_df['test_precision'].max():.2%})")
        print(f"  Timing Tau: {m_df['tau_mean'].mean():.4f} ({m_df['tau_mean'].min():.4f} - {m_df['tau_mean'].max():.4f})")

    # Fit Friberg parameters for Seed 42
    print("\n--- Fitting Friberg ODE parameters for Seed 42 ---")
    p_df = pd.read_csv(os.path.join(BASE_DIR, "data", "patient_profiles.csv"))
    c_df = pd.read_csv(os.path.join(BASE_DIR, "data", "clinical_logs.csv"))
    w_df = pd.read_csv(os.path.join(BASE_DIR, "data", "wearable_streams.csv"))
    params_df = fit_all_patient_friberg_models(clinical_df=c_df, profiles_df=p_df, wearable_df=w_df)
    params_df.to_csv(os.path.join(BASE_DIR, "data", "patient_friberg_parameters.csv"), index=False)
    print("Saved data/patient_friberg_parameters.csv")

    # Regenerate all figures in docs/figures
    print("\n--- Generating Evaluation Figures ---")
    generate_evaluation_figures_and_report(
        data_dir=os.path.join(BASE_DIR, "data"),
        models_dir=os.path.join(BASE_DIR, "models"),
        docs_dir=os.path.join(BASE_DIR, "docs", "figures"),
    )
    print("All figures regenerated successfully.")


if __name__ == "__main__":
    run_multi_seed_benchmark()
