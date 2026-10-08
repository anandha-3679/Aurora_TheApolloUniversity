import numpy as np
import pandas as pd
from src.tumor_model import GompertzTumorModel


def test_gompertz_step_growth():
    model = GompertzTumorModel(growth_rate=0.042, carrying_capacity=100.0)
    initial_volume = 15.0
    grown_volume = model.step_growth(initial_volume)
    assert grown_volume > initial_volume
    assert grown_volume < model.carrying_capacity


def test_simulate_cost_of_delay_tradeoff_summary():
    model = GompertzTumorModel()
    initial_volume = 15.0
    delays = [0, 3, 7, 14]

    summary = model.forecast_with_uncertainty(
        current_volume=initial_volume, candidate_delays=delays
    )
    assert summary[0]["excess_volume_cm3"] == 0.0
    assert summary[0]["regrowth_penalty_pct"] == 0.0
    assert summary[0]["lower_bound_90_cm3"] <= summary[0]["volume_at_dose_cm3"]
    assert summary[0]["upper_bound_90_cm3"] >= summary[0]["volume_at_dose_cm3"]

    # Longer delay leads to strictly greater tumor regrowth before treatment
    assert summary[3]["excess_volume_cm3"] > 0
    assert summary[7]["excess_volume_cm3"] > summary[3]["excess_volume_cm3"]
    assert summary[14]["excess_volume_cm3"] > summary[7]["excess_volume_cm3"]
    assert summary[14]["regrowth_penalty_pct"] > summary[7]["regrowth_penalty_pct"]
    assert (
        summary[14]["upper_bound_90_cm3"] > summary[14]["lower_bound_90_cm3"]
    )


def test_perturb_tumor_rows_after_d_invariance():
    """Assert perturbing tumor rows after day D does not change the forecast."""
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    model = GompertzTumorModel()

    # Test across multiple representative patients and decision days
    test_cases = [("PT_001", 21), ("PT_002", 28), ("PT_005", 14)]

    for p_id, d in test_cases:
        p_data = clinical_df[clinical_df["patient_id"] == p_id].copy()

        # 1. Base fit and forecast
        base_fit = model.fit_patient_parameters(p_data, decision_day=d)
        base_forecast = model.forecast_with_uncertainty(
            current_volume=base_fit["volume_at_d"],
            candidate_delays=[0, 3, 7, 14],
            growth_rate=base_fit["growth_rate"],
            kill_fraction=base_fit["kill_fraction"],
            residual_std=base_fit["residual_std"],
        )

        # 2. Perturb all clinical and tumor rows strictly after day D
        pert_data = p_data.copy()
        fut_mask = pert_data["day_index"] > d
        pert_data.loc[fut_mask, "tumor_volume_cm3"] = 999.0
        pert_data.loc[fut_mask, "anc"] = 0.01
        pert_data.loc[fut_mask, "dose_mg"] = 500.0

        pert_fit = model.fit_patient_parameters(pert_data, decision_day=d)
        pert_forecast = model.forecast_with_uncertainty(
            current_volume=pert_fit["volume_at_d"],
            candidate_delays=[0, 3, 7, 14],
            growth_rate=pert_fit["growth_rate"],
            kill_fraction=pert_fit["kill_fraction"],
            residual_std=pert_fit["residual_std"],
        )

        # 3. Assert fitted parameters and forecast bands are identical
        assert np.isclose(base_fit["growth_rate"], pert_fit["growth_rate"])
        assert np.isclose(base_fit["kill_fraction"], pert_fit["kill_fraction"])
        assert np.isclose(base_fit["volume_at_d"], pert_fit["volume_at_d"])
        assert np.isclose(base_fit["residual_std"], pert_fit["residual_std"])

        for delay in [0, 3, 7, 14]:
            assert (
                base_forecast[delay]["volume_at_dose_cm3"]
                == pert_forecast[delay]["volume_at_dose_cm3"]
            )
            assert (
                base_forecast[delay]["lower_bound_90_cm3"]
                == pert_forecast[delay]["lower_bound_90_cm3"]
            )
            assert (
                base_forecast[delay]["upper_bound_90_cm3"]
                == pert_forecast[delay]["upper_bound_90_cm3"]
            )


def test_evaluate_forecast_mape_outperforms_population():
    """Assert patient Gompertz fit achieves lower MAPE than population-average."""
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    model = GompertzTumorModel()
    mape_results = model.evaluate_forecast_mape(clinical_df, decision_day=21)

    assert mape_results["num_test_patients"] > 0
    # Patient-specific forecast should achieve lower mean error than population average
    assert (
        mape_results["patient_specific_mape_mean"]
        < mape_results["population_average_mape_mean"]
    )
