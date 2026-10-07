import os
import pandas as pd
from src.generate_data import generate_synthetic_data


def test_synthetic_data_generation(tmp_path):
    output_dir = str(tmp_path / "data")
    generate_synthetic_data(num_patients=10, days=30, seed=123, output_dir=output_dir)

    profiles_file = os.path.join(output_dir, "patient_profiles.csv")
    clinical_file = os.path.join(output_dir, "clinical_logs.csv")
    wearable_file = os.path.join(output_dir, "wearable_streams.csv")

    assert os.path.exists(profiles_file)
    assert os.path.exists(clinical_file)
    assert os.path.exists(wearable_file)

    profiles_df = pd.read_csv(profiles_file)
    clinical_df = pd.read_csv(clinical_file)
    wearable_df = pd.read_csv(wearable_file)

    assert len(profiles_df) == 10
    assert len(clinical_df) == 10 * 30
    assert len(wearable_df) == 10 * 30

    # Ensure required columns exist
    assert "patient_id" in profiles_df.columns
    assert "baseline_ecog" in profiles_df.columns
    assert "anc" in clinical_df.columns
    assert "hrv_rmssd" in wearable_df.columns
    assert "resting_heart_rate" in wearable_df.columns
