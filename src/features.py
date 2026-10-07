from typing import Tuple
import numpy as np
import pandas as pd


def compute_wearable_features(
    wearable_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 7-day rolling averages and deviations from 30-day personal baseline."""
    df = wearable_df.sort_values(["patient_id", "day_index"]).copy()

    metrics = [
        "resting_heart_rate",
        "hrv_rmssd",
        "step_count",
        "sleep_duration_mins",
    ]

    for m in metrics:
        df[f"{m}_base30"] = df.groupby("patient_id")[m].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        df[f"{m}_mean7"] = df.groupby("patient_id")[m].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df[f"{m}_ratio_7_to_30"] = np.where(
            df[f"{m}_base30"] > 0,
            df[f"{m}_mean7"] / df[f"{m}_base30"],
            1.0,
        )

    return df


def extract_features_at_decision_day(
    patient_id: str,
    decision_day: int,
    profiles_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    wearable_feat_df: pd.DataFrame,
    delay_days: int = 0,
    days_since_last_dose: float = 21.0,
    planned_dose: float = 100.0,
) -> dict:
    """Extract strictly retrospective features known at or before decision_day (<= D).

    Guarantees:
    - Wearables and labs are frozen at decision_day D.
    - Zero rows with day_index > decision_day are ever accessed.
    - The candidate delay is represented purely through known scheduling variables:
      `delay_days` (days until dose) and `days_since_last_dose`.
    """
    prof = profiles_df[profiles_df["patient_id"] == patient_id].iloc[0]

    # Clinical state up to decision day D
    clin_hist = clinical_df[
        (clinical_df["patient_id"] == patient_id)
        & (clinical_df["day_index"] <= decision_day)
    ]
    pre_dose_anc = (
        float(clin_hist.iloc[-1]["anc"])
        if not clin_hist.empty
        else float(prof["baseline_anc"])
    )
    tumor_vol = (
        float(clin_hist.iloc[-1]["tumor_volume_cm3"]) if not clin_hist.empty else 20.0
    )

    # Wearable metrics at decision day D
    w_curr = wearable_feat_df[
        (wearable_feat_df["patient_id"] == patient_id)
        & (wearable_feat_df["day_index"] == decision_day)
    ]
    if w_curr.empty:
        # Fallback to closest past day if exact day is missing
        w_past = wearable_feat_df[
            (wearable_feat_df["patient_id"] == patient_id)
            & (wearable_feat_df["day_index"] <= decision_day)
        ]
        w_curr = w_past.iloc[-1:]

    w_row = w_curr.iloc[0]

    # Nonlinear representation of interval since last dose:
    # Marrow nadir risk peaks around days 7-14 then dissipates after day 15+
    dsld = float(days_since_last_dose)
    bin_0_3 = 1.0 if dsld <= 3.0 else 0.0
    bin_4_7 = 1.0 if 3.0 < dsld <= 7.0 else 0.0
    bin_8_14 = 1.0 if 7.0 < dsld <= 14.0 else 0.0
    bin_15_plus = 1.0 if dsld > 14.0 else 0.0

    return {
        "patient_id": patient_id,
        "decision_day": decision_day,
        "delay_days": float(delay_days),
        "days_since_last_dose": dsld,
        "dose_interval_bin_0_3": bin_0_3,
        "dose_interval_bin_4_7": bin_4_7,
        "dose_interval_bin_8_14": bin_8_14,
        "dose_interval_bin_15_plus": bin_15_plus,
        # Static patient profile
        "age": float(prof["age"]),
        "sex": str(prof["sex"]),
        "stage": str(prof["stage"]),
        "biomarker_status": str(prof["biomarker_status"]),
        "baseline_ecog": float(prof["baseline_ecog"]),
        "baseline_anc": float(prof["baseline_anc"]),
        # Treatment & Clinical state at decision day
        "dose_mg": float(planned_dose),
        "pre_dose_anc": pre_dose_anc,
        "tumor_volume_cm3": tumor_vol,
        # Wearable metrics frozen at decision day D
        "resting_heart_rate": float(w_row["resting_heart_rate"]),
        "hrv_rmssd": float(w_row["hrv_rmssd"]),
        "step_count": float(w_row["step_count"]),
        "sleep_duration_mins": float(w_row["sleep_duration_mins"]),
        "rhr_ratio_7_to_30": float(w_row["resting_heart_rate_ratio_7_to_30"]),
        "hrv_ratio_7_to_30": float(w_row["hrv_rmssd_ratio_7_to_30"]),
        "steps_ratio_7_to_30": float(w_row["step_count_ratio_7_to_30"]),
        "sleep_ratio_7_to_30": float(w_row["sleep_duration_mins_ratio_7_to_30"]),
    }


def build_toxicity_dataset(
    profiles_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    wearable_df: pd.DataFrame,
    timing_df: pd.DataFrame = None,
    severe_anc_threshold: float = 1.0,
) -> pd.DataFrame:
    """Build dataset for on-schedule and delayed candidate dosing decisions.

    For every candidate decision made on day D with proposed delay delta:
    - Features strictly use information known at day D (<= D).
    - Label evaluates whether severe neutropenia occurs in 7 days after the actual dose:
      actual_dose_day = D + delay_days.
      Label window: (actual_dose_day, actual_dose_day + 7].
    """
    wearable_feat_df = compute_wearable_features(wearable_df)

    records = []

    # 1. On-schedule dosing events (delay_days = 0)
    dosing_events = clinical_df[clinical_df["dose_mg"] > 0].copy()
    for _, dose_row in dosing_events.iterrows():
        p_id = dose_row["patient_id"]
        d = int(dose_row["day_index"])
        dose_val = float(dose_row["dose_mg"])

        # Label on post-dose 7-day window
        window = clinical_df[
            (clinical_df["patient_id"] == p_id)
            & (clinical_df["day_index"] > d)
            & (clinical_df["day_index"] <= d + 7)
        ]
        label = (
            1
            if (not window.empty and (window["anc"] < severe_anc_threshold).any())
            else 0
        )

        # Days since prior dose
        prev_doses = clinical_df[
            (clinical_df["patient_id"] == p_id)
            & (clinical_df["dose_mg"] > 0)
            & (clinical_df["day_index"] < d)
        ]
        days_since_prev = (
            float(d - prev_doses.iloc[-1]["day_index"])
            if not prev_doses.empty
            else 21.0
        )

        feat = extract_features_at_decision_day(
            patient_id=p_id,
            decision_day=d,
            profiles_df=profiles_df,
            clinical_df=clinical_df,
            wearable_feat_df=wearable_feat_df,
            delay_days=0,
            days_since_last_dose=days_since_prev,
            planned_dose=dose_val,
        )
        feat["severe_toxicity_label"] = label
        records.append(feat)

    # 2. Delayed candidate decisions (delays > 0 evaluated at planned day D)
    if timing_df is not None:
        delayed_subset = timing_df[timing_df["delay_days"] > 0]
        for _, row in delayed_subset.iterrows():
            p_id = row["patient_id"]
            d = int(row["planned_day"])  # The decision day D
            delay = int(row["delay_days"])  # Delay delta (+3, +7, +14)
            label = int(row["severe_toxicity_label"])

            prev_doses = clinical_df[
                (clinical_df["patient_id"] == p_id)
                & (clinical_df["dose_mg"] > 0)
                & (clinical_df["day_index"] < d)
            ]
            days_since_prev = (
                float(d - prev_doses.iloc[-1]["day_index"])
                if not prev_doses.empty
                else 21.0
            )

            feat = extract_features_at_decision_day(
                patient_id=p_id,
                decision_day=d,
                profiles_df=profiles_df,
                clinical_df=clinical_df,
                wearable_feat_df=wearable_feat_df,
                delay_days=delay,
                days_since_last_dose=days_since_prev + delay,
                planned_dose=100.0,
            )
            feat["severe_toxicity_label"] = label
            records.append(feat)

    return pd.DataFrame(records)


def build_timing_evaluation_dataset(
    profiles_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    wearable_df: pd.DataFrame,
    timing_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build candidate timing dataset for intra-patient ranking without leakage.

    Features for ALL delays (0, 3, 7, 14) are frozen at decision day D (planned_day).
    Delay is represented solely by `delay_days` and `days_since_last_dose`.
    Zero data from days > planned_day is accessed.
    """
    wearable_feat_df = compute_wearable_features(wearable_df)

    records = []
    for _, row in timing_df.iterrows():
        p_id = row["patient_id"]
        planned_day = int(row["planned_day"])
        cycle = int(row["cycle"])
        delay = int(row["delay_days"])
        label = int(row["severe_toxicity_label"])

        prev_doses = clinical_df[
            (clinical_df["patient_id"] == p_id)
            & (clinical_df["dose_mg"] > 0)
            & (clinical_df["day_index"] < planned_day)
        ]
        days_since_prev = (
            float(planned_day - prev_doses.iloc[-1]["day_index"])
            if not prev_doses.empty
            else 21.0
        )

        feat = extract_features_at_decision_day(
            patient_id=p_id,
            decision_day=planned_day,
            profiles_df=profiles_df,
            clinical_df=clinical_df,
            wearable_feat_df=wearable_feat_df,
            delay_days=delay,
            days_since_last_dose=days_since_prev + delay,
            planned_dose=100.0,
        )
        feat["cycle"] = cycle
        feat["planned_day"] = planned_day
        feat["severe_toxicity_label"] = label
        if "nadir_anc" in row:
            feat["nadir_anc"] = float(row["nadir_anc"])
        records.append(feat)

    return pd.DataFrame(records)


def get_feature_sets() -> Tuple[list, list]:
    """Return feature column subsets for Ablation A and Ablation B.

    Constant columns (dose_mg, dose_interval_bin_0_3, dose_interval_bin_4_7)
    and duplicate bins (dose_interval_bin_15_plus == 1 - bin_8_14)
    are excluded to avoid zero-variance degradation and multicollinearity.
    """
    static_clinical_features = [
        "age",
        "baseline_ecog",
        "baseline_anc",
        "pre_dose_anc",
        "tumor_volume_cm3",
        "delay_days",
        "days_since_last_dose",
        "dose_interval_bin_8_14",
    ]
    wearable_features = [
        "resting_heart_rate",
        "hrv_rmssd",
        "step_count",
        "sleep_duration_mins",
        "rhr_ratio_7_to_30",
        "hrv_ratio_7_to_30",
        "steps_ratio_7_to_30",
        "sleep_ratio_7_to_30",
    ]
    ablation_a = static_clinical_features
    ablation_b = static_clinical_features + wearable_features
    return ablation_a, ablation_b
