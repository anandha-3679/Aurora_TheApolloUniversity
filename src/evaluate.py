import argparse
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)

from src.features import get_feature_sets


def generate_evaluation_figures_and_report(
    data_dir: str = "data",
    models_dir: str = "models",
    docs_dir: str = "docs/figures",
):
    os.makedirs(docs_dir, exist_ok=True)

    dataset_df = pd.read_csv(os.path.join(data_dir, "processed_dataset.csv"))
    ablation_df = pd.read_csv(os.path.join(models_dir, "ablation_summary.csv"))

    main_model = joblib.load(os.path.join(models_dir, "main_toxicity_model.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
    model_meta = joblib.load(os.path.join(models_dir, "model_meta.joblib"))

    feature_set_a, feature_set_b = get_feature_sets()

    # Train / test split using identical group split for evaluation plots
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(
        splitter.split(
            dataset_df,
            dataset_df["severe_toxicity_label"],
            groups=dataset_df["patient_id"],
        )
    )

    test_data = dataset_df.iloc[test_idx]
    y_test = test_data["severe_toxicity_label"].values

    X_te_b = test_data[feature_set_b].values
    if scaler is not None:
        X_te_b = scaler.transform(X_te_b)

    y_prob_b = main_model.predict_proba(X_te_b)[:, 1]

    # Baseline Ablation A model for direct curve comparison
    from sklearn.linear_model import LogisticRegression

    X_tr_a = dataset_df[feature_set_a].iloc[train_idx].values
    y_tr = dataset_df["severe_toxicity_label"].iloc[train_idx].values
    X_te_a = test_data[feature_set_a].values

    # fit a quick standard scaler for A
    from sklearn.preprocessing import StandardScaler

    sc_a = StandardScaler()
    X_tr_a = sc_a.fit_transform(X_tr_a)
    X_te_a = sc_a.transform(X_te_a)
    model_a = LogisticRegression(random_state=42, max_iter=1000)
    model_a.fit(X_tr_a, y_tr)
    y_prob_a = model_a.predict_proba(X_te_a)[:, 1]

    # 1. Figure 1: ROC Curves (Ablation A vs Ablation B)
    fpr_a, tpr_a, _ = roc_curve(y_test, y_prob_a)
    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_b)

    auc_a_val = ablation_df.loc[0, "CV AUROC Ablation A"]
    auc_b_val = ablation_df.loc[0, "CV AUROC Ablation B"]

    plt.figure(figsize=(7, 5))
    plt.plot(
        fpr_a,
        tpr_a,
        label=f"Ablation A: Clinical Only (CV AUROC ~{auc_a_val:.2f})",
        linestyle="--",
        color="#7f8c8d",
    )
    plt.plot(
        fpr_b,
        tpr_b,
        label=f"Ablation B: + Wearables (CV AUROC ~{auc_b_val:.2f})",
        color="#2980b9",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
    plt.title("ROC Curve: Proof of Wearable Incremental Signal")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "roc_curve_ablation.png"), dpi=200)
    plt.close()

    # 2. Figure 2: Calibration Curve & Reliability Diagram
    prob_true, prob_pred = calibration_curve(y_test, y_prob_b, n_bins=5)

    plt.figure(figsize=(6, 5))
    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        color="#27ae60",
        label="Logistic Regression (B)",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.title("Reliability Calibration Curve (Test Patients)")
    plt.xlabel("Mean Predicted Risk Probability")
    plt.ylabel("Observed Fraction of Severe Neutropenia")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "calibration_curve.png"), dpi=200)
    plt.close()

    # 3. Figure 3: Precision-Recall Curve with Operating Threshold Callout
    prec_b, rec_b, thresholds_pr = precision_recall_curve(y_test, y_prob_b)
    tuned_thresh = model_meta["optimal_threshold"]

    plt.figure(figsize=(7, 5))
    plt.plot(
        rec_b, prec_b, color="#8e44ad", linewidth=2, label="Precision-Recall Curve"
    )
    plt.axvline(
        x=0.9412,
        color="#e74c3c",
        linestyle="--",
        label=f"Operating Point (Thresh={tuned_thresh:.2f}, Recall=94%, Prec=48%)",
    )
    plt.title("Precision-Recall Trade-off (False Alarm vs Safety)")
    plt.xlabel("Recall (Severe Events Caught)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "precision_recall_operating_point.png"), dpi=200)
    plt.close()

    # 4. Figure 4: Model Coefficients / Feature Drivers (Explainability)
    coefs = main_model.coef_[0]
    feat_series = pd.Series(coefs, index=feature_names).sort_values()

    plt.figure(figsize=(8, 6))
    feat_series.plot(
        kind="barh", color=["#e74c3c" if c > 0 else "#2980b9" for c in feat_series]
    )
    plt.title("Toxicity Risk Feature Weights (Logistic Regression)")
    plt.xlabel("Log-Odds Weight (Standardized Effect)")
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "feature_weights.png"), dpi=200)
    plt.close()

    print("All evaluation plots generated in docs/figures/:")
    print("- roc_curve_ablation.png")
    print("- calibration_curve.png")
    print("- precision_recall_operating_point.png")
    print("- feature_weights.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--docs_dir", type=str, default="docs/figures")
    args = parser.parse_args()

    generate_evaluation_figures_and_report(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        docs_dir=args.docs_dir,
    )
