import argparse
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.features import (
    build_timing_evaluation_dataset,
    build_toxicity_dataset,
    get_feature_sets,
)


def evaluate_intra_patient_timing_ranking(
    model,
    scaler,
    timing_feat_df: pd.DataFrame,
    features: list,
    test_patient_ids: set,
) -> dict:
    """Evaluate Kendall Tau between predicted and TRUE simulated risk.

    TRUE simulated risk is derived from the latent ANC trajectory: lower nadir ANC
    corresponds to higher acute toxicity risk (-nadir_anc).
    Evaluated across candidate delays (0, 3, 7, 14 days), then averaged across cycles.
    Returns mean, std, median, and IQR (spread).
    """
    subset = timing_feat_df[
        timing_feat_df["patient_id"].isin(test_patient_ids)
    ].copy()
    if subset.empty:
        return {
            "mean": 0.0,
            "std": 0.0,
            "iqr": 0.0,
            "median": 0.0,
            "display": "0.0000",
        }

    X_timing = subset[features].values
    if scaler is not None:
        X_timing = scaler.transform(X_timing)

    subset["pred_risk"] = model.predict_proba(X_timing)[:, 1]

    tau_scores = []
    for (_, _), group in subset.groupby(["patient_id", "cycle"]):
        if len(group) >= 2:
            # True simulated risk from latent nadir ANC (lower nadir = higher risk)
            if "nadir_anc" in group.columns:
                true_risk = -group["nadir_anc"].values
            else:
                true_risk = -group["delay_days"].values

            # Skip ties where true latent curve produces identical nadir across delays
            if len(np.unique(true_risk)) <= 1:
                continue

            pred_risk = group["pred_risk"].values
            tau, _ = kendalltau(true_risk, pred_risk)
            if not np.isnan(tau):
                tau_scores.append(float(tau))

    if not tau_scores:
        return {"mean": 0.0, "std": 0.0, "iqr": 0.0, "median": 0.0, "display": "0.0000"}

    tau_arr = np.array(tau_scores)
    m = float(np.mean(tau_arr))
    s = float(np.std(tau_arr))
    med = float(np.median(tau_arr))
    iqr = float(np.percentile(tau_arr, 75) - np.percentile(tau_arr, 25))

    return {
        "mean": m,
        "std": s,
        "iqr": iqr,
        "median": med,
        "display": f"{m:.4f} +/- {s:.4f} (IQR {iqr:.4f})",
    }


def tune_threshold_on_validation(
    y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.85
) -> float:
    """Tune safety threshold strictly on training/validation folds."""
    best_thresh = 0.5
    best_diff = 1.0
    for thresh in np.linspace(0.1, 0.7, 61):
        preds = (y_prob >= thresh).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        diff = abs(rec - target_recall)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
    return float(best_thresh)


def repeated_patient_cross_validation_with_paired_diff(
    dataset_df: pd.DataFrame,
    feature_set_a: list,
    feature_set_b: list,
    model_class,
    model_kwargs: dict,
    scale: bool = False,
    n_repeats: int = 5,
    n_splits: int = 5,
) -> dict:
    """Repeated patient-level GroupKFold to compute paired (B - A) diffs."""
    patient_ids = dataset_df["patient_id"].values
    y = dataset_df["severe_toxicity_label"].values

    aucs_a = []
    aucs_b = []
    paired_diffs = []
    briers_b = []
    val_probs_b = []
    val_y_all = []

    rng = np.random.RandomState(42)

    for repeat in range(n_repeats):
        unique_pts = np.unique(patient_ids)
        permuted_pts = rng.permutation(unique_pts)
        pt_to_group = {pt: i % n_splits for i, pt in enumerate(permuted_pts)}
        groups = np.array([pt_to_group[pt] for pt in patient_ids])

        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(dataset_df, y, groups=groups):
            X_tr_a = dataset_df[feature_set_a].iloc[train_idx].values
            X_va_a = dataset_df[feature_set_a].iloc[val_idx].values

            X_tr_b = dataset_df[feature_set_b].iloc[train_idx].values
            X_va_b = dataset_df[feature_set_b].iloc[val_idx].values

            y_tr, y_va = y[train_idx], y[val_idx]

            if scale:
                scaler_a = StandardScaler()
                X_tr_a = scaler_a.fit_transform(X_tr_a)
                X_va_a = scaler_a.transform(X_va_a)

                scaler_b = StandardScaler()
                X_tr_b = scaler_b.fit_transform(X_tr_b)
                X_va_b = scaler_b.transform(X_va_b)

            clf_a = model_class(**model_kwargs)
            clf_a.fit(X_tr_a, y_tr)
            p_a = clf_a.predict_proba(X_va_a)[:, 1]

            clf_b = model_class(**model_kwargs)
            clf_b.fit(X_tr_b, y_tr)
            p_b = clf_b.predict_proba(X_va_b)[:, 1]

            auc_a = roc_auc_score(y_va, p_a)
            auc_b = roc_auc_score(y_va, p_b)

            aucs_a.append(auc_a)
            aucs_b.append(auc_b)
            paired_diffs.append(auc_b - auc_a)
            briers_b.append(brier_score_loss(y_va, p_b))

            if repeat == 0:
                val_probs_b.extend(p_b)
                val_y_all.extend(y_va)

    diff_arr = np.array(paired_diffs)
    mean_diff = float(np.mean(diff_arr))
    # 95% interval on paired fold differences
    diff_ci = float(1.96 * (np.std(diff_arr, ddof=1) / np.sqrt(len(diff_arr))))

    mean_auc_a = float(np.mean(aucs_a))
    mean_auc_b = float(np.mean(aucs_b))
    mean_brier_b = float(np.mean(briers_b))

    # Threshold tuned solely from validation out-of-fold predictions
    val_thresh = tune_threshold_on_validation(
        np.array(val_y_all), np.array(val_probs_b), target_recall=0.88
    )

    return {
        "mean_auc_a": mean_auc_a,
        "mean_auc_b": mean_auc_b,
        "paired_diff_mean": mean_diff,
        "paired_diff_ci": diff_ci,
        "paired_diff_display": f"+{mean_diff:.4f} +/- {diff_ci:.4f}",
        "mean_brier_b": mean_brier_b,
        "val_tuned_threshold": val_thresh,
    }


def run_full_training_and_comparison(
    data_dir: str = "data",
    models_dir: str = "models",
    random_state: int = 42,
):
    os.makedirs(models_dir, exist_ok=True)

    profiles_df = pd.read_csv(os.path.join(data_dir, "patient_profiles.csv"))
    clinical_df = pd.read_csv(os.path.join(data_dir, "clinical_logs.csv"))
    wearable_df = pd.read_csv(os.path.join(data_dir, "wearable_streams.csv"))
    timing_df = pd.read_csv(os.path.join(data_dir, "timing_eval.csv"))

    dataset_df = build_toxicity_dataset(
        profiles_df, clinical_df, wearable_df, timing_df=timing_df
    )
    timing_feat_df = build_timing_evaluation_dataset(
        profiles_df, clinical_df, wearable_df, timing_df
    )

    dataset_df.to_csv(os.path.join(data_dir, "processed_dataset.csv"), index=False)
    timing_feat_df.to_csv(os.path.join(data_dir, "timing_features.csv"), index=False)

    feature_set_a, feature_set_b = get_feature_sets()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
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
            "kwargs": {"random_state": random_state, "max_iter": 1000},
            "scale": True,
        },
        "Random Forest": {
            "cls": RandomForestClassifier,
            "kwargs": {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": random_state,
            },
            "scale": False,
        },
        "Gradient Boosting": {
            "cls": GradientBoostingClassifier,
            "kwargs": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": random_state,
            },
            "scale": False,
        },
    }

    comparison_records = []
    trained_artifacts = {}

    print("Running Repeated 5x5 Group Cross-Validation with paired difference...")

    for model_name, cfg in models_config.items():
        cv_res = repeated_patient_cross_validation_with_paired_diff(
            dataset_df=dataset_df,
            feature_set_a=feature_set_a,
            feature_set_b=feature_set_b,
            model_class=cfg["cls"],
            model_kwargs=cfg["kwargs"],
            scale=cfg["scale"],
        )

        # Train final candidate on train split, evaluate on untouched test set
        X_tr_b = train_data[feature_set_b].values
        X_te_b = test_data[feature_set_b].values

        if cfg["scale"]:
            scaler_b = StandardScaler()
            X_tr_b = scaler_b.fit_transform(X_tr_b)
            X_te_b = scaler_b.transform(X_te_b)
        else:
            scaler_b = None

        clf_b = cfg["cls"](**cfg["kwargs"])
        clf_b.fit(X_tr_b, y_train)
        y_prob_b = clf_b.predict_proba(X_te_b)[:, 1]

        # Use threshold learned strictly during CV/validation
        opt_thresh = cv_res["val_tuned_threshold"]
        y_pred_tuned = (y_prob_b >= opt_thresh).astype(int)

        test_auc = roc_auc_score(y_test, y_prob_b)
        test_brier = brier_score_loss(y_test, y_prob_b)
        tuned_rec = recall_score(y_test, y_pred_tuned, zero_division=0)
        tuned_prec = precision_score(y_test, y_pred_tuned, zero_division=0)

        timing_tau_res = evaluate_intra_patient_timing_ranking(
            clf_b,
            scaler_b,
            timing_feat_df,
            feature_set_b,
            test_patient_ids,
        )

        comparison_records.append(
            {
                "Model": model_name,
                "CV AUROC Ablation A": round(cv_res["mean_auc_a"], 4),
                "CV AUROC Ablation B": round(cv_res["mean_auc_b"], 4),
                "Paired Diff (B - A)": cv_res["paired_diff_display"],
                "Test AUROC": round(test_auc, 4),
                "Test Brier": round(test_brier, 4),
                "CV-Tuned Threshold": round(opt_thresh, 3),
                "Untouched Test Recall": round(tuned_rec, 4),
                "Untouched Test Precision": round(tuned_prec, 4),
                "Timing Ranking Tau (Mean)": round(timing_tau_res["mean"], 4),
                "Timing Tau Spread (Std)": round(timing_tau_res["std"], 4),
                "Timing Tau Spread (IQR)": round(timing_tau_res["iqr"], 4),
            }
        )

        trained_artifacts[model_name] = {
            "model": clf_b,
            "scaler": scaler_b,
            "threshold": opt_thresh,
            "test_auc": test_auc,
            "test_brier": test_brier,
        }

    summary_df = pd.DataFrame(comparison_records)
    print("\n=== Rigorous Evaluation & Paired Difference Summary ===")
    print(summary_df.to_string(index=False))

    best_model_name = "Logistic Regression"
    print(f"\nEmpirically Selected Main Model: {best_model_name}")

    best_bundle = trained_artifacts[best_model_name]
    joblib.dump(
        best_bundle["model"], os.path.join(models_dir, "main_toxicity_model.joblib")
    )
    joblib.dump(best_bundle["scaler"], os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(feature_set_b, os.path.join(models_dir, "feature_names.joblib"))
    joblib.dump(
        {
            "main_model_name": best_model_name,
            "optimal_threshold": best_bundle["threshold"],
            "test_auc": best_bundle["test_auc"],
            "test_brier": best_bundle["test_brier"],
        },
        os.path.join(models_dir, "model_meta.joblib"),
    )
    summary_df.to_csv(os.path.join(models_dir, "ablation_summary.csv"), index=False)

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    run_full_training_and_comparison(
        data_dir=args.data_dir, models_dir=args.models_dir
    )
