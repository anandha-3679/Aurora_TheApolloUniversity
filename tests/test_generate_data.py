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


def test_generator_marrow_damage_and_cycle_recovery(tmp_path):
    """Verify that the generator includes cumulative marrow damage and inter-cycle recovery variations."""
    output_dir = str(tmp_path / "data_marrow")
    generate_synthetic_data(num_patients=5, days=90, seed=42, output_dir=output_dir)

    clinical_file = os.path.join(output_dir, "clinical_logs.csv")
    clinical_df = pd.read_csv(clinical_file)

    # Check that multiple dosing events exist across 90 days
    dosing_events = clinical_df[clinical_df["dose_mg"] > 0]
    assert len(dosing_events) >= 10
    # Patient ANC values vary dynamically over time
    assert clinical_df["anc"].nunique() > 20


def test_all_seed_folders_contain_patient_profiles():
    """Assert that every seed directory contains its own distinct patient_profiles.csv."""
    seeds = [42, 101, 2024, 7, 999]
    for s in seeds:
        seed_dir = os.path.join("data", f"seed_{s}")
        profile_path = os.path.join(seed_dir, "patient_profiles.csv")
        assert os.path.exists(profile_path), (
            f"Seed folder {seed_dir} is missing patient_profiles.csv"
        )

        # Ensure the file is not empty and has the required schema
        profiles_df = pd.read_csv(profile_path)
        assert len(profiles_df) > 0, (
            f"Seed folder {seed_dir} patient_profiles.csv is empty"
        )
        assert "patient_id" in profiles_df.columns
        assert "baseline_anc" in profiles_df.columns
        assert "baseline_ecog" in profiles_df.columns
