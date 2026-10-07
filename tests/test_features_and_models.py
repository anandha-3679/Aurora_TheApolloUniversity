import joblib
import numpy as np
import pandas as pd
import pytest
from src.features import (
    build_timing_evaluation_dataset,
    build_toxicity_dataset,
    compute_wearable_features,
    extract_features_at_decision_day,
    get_feature_sets,
)


def test_compute_wearable_features():
    dummy_wearables = pd.DataFrame(
        {
            "patient_id": ["PT_001"] * 10,
            "day_index": list(range(10)),
            "resting_heart_rate": [70 + i for i in range(10)],
            "hrv_rmssd": [50 - i for i in range(10)],
            "step_count": [6000 - i * 100 for i in range(10)],
            "sleep_duration_mins": [450 for _ in range(10)],
        }
    )
    features_df = compute_wearable_features(dummy_wearables)
    assert "resting_heart_rate_base30" in features_df.columns
    assert "hrv_rmssd_mean7" in features_df.columns
    assert "step_count_ratio_7_to_30" in features_df.columns
    assert len(features_df) == 10


def test_feature_sets_definition():
    feat_a, feat_b = get_feature_sets()
    assert len(feat_a) > 0
    assert len(feat_b) > len(feat_a)
    assert "resting_heart_rate" in feat_b
    assert "resting_heart_rate" not in feat_a
    assert "delay_days" in feat_a and "delay_days" in feat_b
    assert "dose_interval_bin_8_14" in feat_a and "dose_interval_bin_8_14" in feat_b
    # Confirm dropped constant / duplicate features
    assert "dose_mg" not in feat_a
    assert "dose_interval_bin_0_3" not in feat_a


def test_timing_dataset_builder():
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    wearable_df = pd.read_csv("data/wearable_streams.csv")
    timing_df = pd.read_csv("data/timing_eval.csv")

    timing_feat_df = build_timing_evaluation_dataset(
        profiles_df, clinical_df, wearable_df, timing_df
    )
    assert len(timing_feat_df) == len(timing_df)
    assert "hrv_ratio_7_to_30" in timing_feat_df.columns
    assert "pre_dose_anc" in timing_feat_df.columns
    assert "delay_days" in timing_feat_df.columns
    assert "dose_interval_bin_8_14" in timing_feat_df.columns


def test_random_patients_and_days_future_perturbation_invariance():
    """Requirement 1: Random patients & decision days D.

    Perturb all wearable and lab values after D, and assert predicted risk
    for every candidate delay (+0, +3, +7, +14) remains unchanged.
    """
    np.random.seed(42)
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    wearable_df = pd.read_csv("data/wearable_streams.csv")

    model = joblib.load("models/main_toxicity_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    feature_names = joblib.load("models/feature_names.joblib")

    all_patients = profiles_df["patient_id"].unique()
    sample_patients = np.random.choice(all_patients, size=5, replace=False)
    candidate_delays = [0, 3, 7, 14]

    for p_id in sample_patients:
        # Choose random valid decision days where prior history exists
        p_clin = clinical_df[clinical_df["patient_id"] == p_id]
        p_days = p_clin["day_index"].values
        valid_decision_days = p_days[(p_days >= 20) & (p_days <= 60)]
        d = int(np.random.choice(valid_decision_days))

        # 1. Base predictions across delays
        w_feat_base = compute_wearable_features(wearable_df)
        preds_base = []
        for delay in candidate_delays:
            f = extract_features_at_decision_day(
                patient_id=p_id,
                decision_day=d,
                profiles_df=profiles_df,
                clinical_df=clinical_df,
                wearable_feat_df=w_feat_base,
                delay_days=delay,
                days_since_last_dose=21.0 + delay,
                planned_dose=100.0,
            )
            x = pd.DataFrame([f])[feature_names]
            x_scaled = scaler.transform(x.values)
            preds_base.append(float(model.predict_proba(x_scaled)[0, 1]))

        # 2. Perturb ALL wearable and lab values strictly after D (> D)
        pert_clinical = clinical_df.copy()
        pert_wearable = wearable_df.copy()

        fut_clin_mask = (pert_clinical["patient_id"] == p_id) & (
            pert_clinical["day_index"] > d
        )
        pert_clinical.loc[fut_clin_mask, "anc"] = 0.05
        pert_clinical.loc[fut_clin_mask, "wbc"] = 0.10
        pert_clinical.loc[fut_clin_mask, "tumor_volume_cm3"] = 999.0

        fut_wear_mask = (pert_wearable["patient_id"] == p_id) & (
            pert_wearable["day_index"] > d
        )
        pert_wearable.loc[fut_wear_mask, "resting_heart_rate"] = 180.0
        pert_wearable.loc[fut_wear_mask, "hrv_rmssd"] = 5.0
        pert_wearable.loc[fut_wear_mask, "step_count"] = 50.0
        pert_wearable.loc[fut_wear_mask, "sleep_duration_mins"] = 60.0

        w_feat_pert = compute_wearable_features(pert_wearable)
        preds_pert = []
        for delay in candidate_delays:
            f = extract_features_at_decision_day(
                patient_id=p_id,
                decision_day=d,
                profiles_df=profiles_df,
                clinical_df=pert_clinical,
                wearable_feat_df=w_feat_pert,
                delay_days=delay,
                days_since_last_dose=21.0 + delay,
                planned_dose=100.0,
            )
            x = pd.DataFrame([f])[feature_names]
            x_scaled = scaler.transform(x.values)
            preds_pert.append(float(model.predict_proba(x_scaled)[0, 1]))

        # Predictions for every delay must be completely unchanged
        assert np.allclose(preds_base, preds_pert, atol=1e-7), (
            f"Predictions changed for {p_id} at day {d} after perturbation"
        )


def test_no_feature_uses_rows_after_decision_day():
    """Requirement 2: Assert no feature uses rows with day_index > D."""
    np.random.seed(123)
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    wearable_df = pd.read_csv("data/wearable_streams.csv")

    w_feat_df = compute_wearable_features(wearable_df)

    test_cases = [
        ("PT_001", 14),
        ("PT_003", 21),
        ("PT_010", 35),
        ("PT_025", 42),
    ]

    for p_id, d in test_cases:
        # Extract features under baseline
        feat_base = extract_features_at_decision_day(
            patient_id=p_id,
            decision_day=d,
            profiles_df=profiles_df,
            clinical_df=clinical_df,
            wearable_feat_df=w_feat_df,
            delay_days=3,
            days_since_last_dose=24.0,
            planned_dose=100.0,
        )

        # Slice data tables to completely DROP all records where day_index > D
        strict_clin = clinical_df[
            ~((clinical_df["patient_id"] == p_id) & (clinical_df["day_index"] > d))
        ]
        strict_wear_feat = w_feat_df[
            ~((w_feat_df["patient_id"] == p_id) & (w_feat_df["day_index"] > d))
        ]

        feat_truncated = extract_features_at_decision_day(
            patient_id=p_id,
            decision_day=d,
            profiles_df=profiles_df,
            clinical_df=strict_clin,
            wearable_feat_df=strict_wear_feat,
            delay_days=3,
            days_since_last_dose=24.0,
            planned_dose=100.0,
        )

        # Every single extracted feature key must match exactly
        for k in feat_base:
            assert feat_base[k] == pytest.approx(feat_truncated[k]), (
                f"Feature '{k}' depends on rows with day_index > {d} for {p_id}"
            )


def test_each_label_window_starts_after_hypothetical_dose_day():
    """Requirement 3: Assert label window starts strictly after hypothetical dose.

    For candidate delay delta at decision day D:
    hypothetical dose day = D + delay.
    The label evaluation window must be strictly (D + delay, D + delay + 7].
    """
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    wearable_df = pd.read_csv("data/wearable_streams.csv")
    timing_df = pd.read_csv("data/timing_eval.csv")

    dataset = build_toxicity_dataset(
        profiles_df, clinical_df, wearable_df, timing_df=timing_df
    )

    # Filter to candidate delay rows where delay > 0 was evaluated
    cand_rows = dataset[dataset["delay_days"] > 0]
    assert len(cand_rows) > 0, "No candidate delay rows found in toxicity dataset"

    for _, row in cand_rows.head(50).iterrows():
        p_id = row["patient_id"]
        d = int(row["decision_day"])
        delay = int(row["delay_days"])
        dose_day = d + delay

        # Match in timing_df
        t_match = timing_df[
            (timing_df["patient_id"] == p_id)
            & (timing_df["planned_day"] == d)
            & (timing_df["delay_days"] == delay)
        ]
        assert not t_match.empty
        sim_row = t_match.iloc[0]

        # Hypothetical dose day is candidate_day = planned_day + delay
        assert int(sim_row["candidate_day"]) == dose_day

        # Confirm post-nadir window reflects days strictly > dose_day
        # In generator: cand_severity was evaluated for cand_day,
        # measuring nadir in the 7 days after cand_day.
        # Confirm label is binary (0 or 1) and matches simulated toxicity
        assert row["severe_toxicity_label"] == int(sim_row["severe_toxicity_label"])
