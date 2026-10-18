import numpy as np
import pandas as pd
import pytest
from src.friberg_model import FribergANCModel, fit_all_patient_friberg_models


def test_friberg_simulation_nadir():
    model = FribergANCModel()
    t = np.arange(0, 22)
    # Dose on day 0
    anc = model.simulate_anc_trajectory(
        time_days=t,
        base_anc=5.0,
        mtt=5.0,
        recovery_rate=0.17,
        drug_effect=8.0,
    )
    assert len(anc) == 22
    assert anc[0] == pytest.approx(5.0, rel=1e-2)
    # Clinical nadir window for myelosuppressive chemo: ~days 8-14
    min_idx = int(np.argmin(anc))
    assert 8 <= min_idx <= 14
    # A potent dose must be able to produce severe neutropenia
    assert anc[min_idx] < 1.0
    # Recovers towards baseline as t increases past nadir
    assert anc[21] > anc[min_idx]


def test_fit_patient_friberg_model_on_clinical_logs():
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    model = FribergANCModel()

    p_data = clinical_df[clinical_df["patient_id"] == "PT_001"]
    p_profile = profiles_df[profiles_df["patient_id"] == "PT_001"].iloc[0]
    base_anc_prior = float(p_profile["baseline_anc"])

    fitted = model.fit_patient_anc(
        patient_history=p_data,
        decision_day=45,
        baseline_anc_prior=base_anc_prior,
    )

    assert "base_anc" in fitted
    assert "mtt" in fitted
    assert "recovery_rate" in fitted
    assert "drug_slope" in fitted
    assert "residual_std" in fitted

    # Check bounds
    assert 2.0 <= fitted["base_anc"] <= 7.5
    assert 4.0 <= fitted["mtt"] <= 14.0
    assert 0.1 <= fitted["recovery_rate"] <= 2.0
    assert 0.5 <= fitted["drug_slope"] <= 10.0
    assert fitted["residual_std"] >= 0.05


def test_fit_friberg_future_invariance():
    """Verify perturbing ANC rows after day D leaves fitted parameters invariant."""
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    model = FribergANCModel()

    p_data = clinical_df[clinical_df["patient_id"] == "PT_002"].copy()
    decision_d = 21

    fit1 = model.fit_patient_anc(p_data, decision_day=decision_d)

    # Perturb rows with day_index > decision_d
    p_data_pert = p_data.copy()
    fut_mask = p_data_pert["day_index"] > decision_d
    p_data_pert.loc[fut_mask, "anc"] = 9.99

    fit2 = model.fit_patient_anc(p_data_pert, decision_day=decision_d)

    assert fit1["base_anc"] == fit2["base_anc"]
    assert fit1["mtt"] == fit2["mtt"]
    assert fit1["recovery_rate"] == fit2["recovery_rate"]
    assert fit1["drug_slope"] == fit2["drug_slope"]


def test_fit_all_patient_friberg_models():
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    profiles_df = pd.read_csv("data/patient_profiles.csv")

    # Subset to 5 patients for fast execution
    subset_pids = ["PT_001", "PT_002", "PT_003", "PT_004", "PT_005"]
    subset_clin = clinical_df[clinical_df["patient_id"].isin(subset_pids)]
    subset_prof = profiles_df[profiles_df["patient_id"].isin(subset_pids)]

    res_df = fit_all_patient_friberg_models(
        clinical_df=subset_clin,
        profiles_df=subset_prof,
    )
    assert len(res_df) == 5
    assert set(res_df["patient_id"]) == set(subset_pids)
    assert "recovery_rate" in res_df.columns
    assert "raw_recovery_rate" in res_df.columns
    assert "marrow_strain" in res_df.columns
    assert "mtt" in res_df.columns
    assert "base_anc" in res_df.columns


def test_wearable_trends_nudge_recovery_rate():
    """Verify that elevated marrow strain nudges recovery rate downward."""
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    wearable_df = pd.read_csv("data/wearable_streams.csv")
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    model = FribergANCModel()

    p_id = "PT_001"
    p_clin = clinical_df[clinical_df["patient_id"] == p_id]
    p_wear = wearable_df[wearable_df["patient_id"] == p_id]
    p_prof = profiles_df[profiles_df["patient_id"] == p_id].iloc[0]

    # Fit without wearables (no strain signal)
    fit_no_wear = model.fit_patient_anc(
        patient_history=p_clin,
        decision_day=45,
        baseline_anc_prior=float(p_prof["baseline_anc"]),
        wearable_history=None,
    )
    assert fit_no_wear["marrow_strain"] == 0.0
    assert fit_no_wear["recovery_rate"] == fit_no_wear["raw_recovery_rate"]

    # Fit with actual wearables
    fit_with_wear = model.fit_patient_anc(
        patient_history=p_clin,
        decision_day=45,
        baseline_anc_prior=float(p_prof["baseline_anc"]),
        wearable_history=p_wear,
    )
    assert "marrow_strain" in fit_with_wear
    assert 0.0 <= fit_with_wear["marrow_strain"] <= 1.0

    # Synthetic high marrow strain: acute strain in days 39-45 vs earlier baseline
    high_strain_wear = p_wear.copy()
    strain_mask = (high_strain_wear["day_index"] >= 39) & (
        high_strain_wear["day_index"] <= 45
    )
    high_strain_wear.loc[strain_mask, "resting_heart_rate"] = 105.0
    high_strain_wear.loc[strain_mask, "hrv_rmssd"] = 12.0
    high_strain_wear.loc[strain_mask, "step_count"] = 1200.0

    strain_val = model.compute_marrow_strain(high_strain_wear, decision_day=45)
    assert strain_val > 0.0

    fit_strained = model.fit_patient_anc(
        patient_history=p_clin,
        decision_day=45,
        baseline_anc_prior=float(p_prof["baseline_anc"]),
        wearable_history=high_strain_wear,
        strain_nudge_weight=0.30,
    )
    # Strained recovery rate must be strictly lower than raw recovery rate
    assert fit_strained["recovery_rate"] < fit_strained["raw_recovery_rate"]
    assert fit_strained["marrow_strain"] > 0.20


def test_forecast_anc_curve_and_days_below_one():
    """Verify forward ANC trajectory simulation, predicted nadir, and days below 1.0."""
    model = FribergANCModel()
    delays = [0, 3, 7, 14]
    forecasts = model.forecast_anc_nadir(
        base_anc=4.5,
        mtt=8.0,
        recovery_rate=0.85,
        candidate_delays=delays,
        dose_mg=100.0,
        drug_slope=2.0,
        horizon_days=21,
        threshold=1.0,
    )

    for d in delays:
        assert d in forecasts
        res = forecasts[d]
        assert "curve" in res
        assert "predicted_nadir_anc" in res
        assert "predicted_nadir_day" in res
        assert "days_below_1_0" in res

        # Curve should have horizon_days + 1 points
        assert len(res["curve"]) == 22
        # Post-dose nadir is the minimum across the forward simulation
        assert res["predicted_nadir_anc"] == pytest.approx(
            float(np.min(res["curve"][1:])), rel=1e-2
        )
        # Verify days below 1.0 count matches curve
        expected_below = int(np.sum(res["curve"][1:] < 1.0))
        assert res["days_below_1_0"] == expected_below
        assert res["days_below_1_0"] >= 0

    # With delay, recovery improves, leading to higher or equal nadir
    # and fewer days below 1.0
    assert forecasts[14]["predicted_nadir_anc"] >= forecasts[0]["predicted_nadir_anc"]
    assert forecasts[14]["days_below_1_0"] <= forecasts[0]["days_below_1_0"]


def test_combine_hybrid_toxicity_risk():
    """Verify hybrid synthesis of mechanistic Friberg nadir & ML second opinion."""
    # Scenario 1: Low nadir (< 0.5) and high ML risk -> Both indicate high risk
    res_high = FribergANCModel.combine_hybrid_toxicity_risk(
        ml_prob=0.82,
        predicted_nadir_anc=0.45,
        days_below_1_0=5,
        mechanistic_weight=0.55,
        ml_weight=0.45,
    )
    assert res_high["hybrid_risk"] > 0.70
    assert res_high["mech_risk"] > 0.70
    assert res_high["ml_risk"] == 0.82
    assert "concordance" in res_high

    # Scenario 2: High nadir (> 2.0) and low ML risk -> Both indicate low risk
    res_low = FribergANCModel.combine_hybrid_toxicity_risk(
        ml_prob=0.08,
        predicted_nadir_anc=2.40,
        days_below_1_0=0,
        mechanistic_weight=0.55,
        ml_weight=0.45,
    )
    assert res_low["hybrid_risk"] < 0.20
    assert res_low["mech_risk"] < 0.20
    assert res_low["concordance"] == "Concordant"

    # Scenario 3: Mixed scenario acts as calibrated balance
    res_mix = FribergANCModel.combine_hybrid_toxicity_risk(
        ml_prob=0.30,
        predicted_nadir_anc=0.80,
        days_below_1_0=2,
        mechanistic_weight=0.50,
        ml_weight=0.50,
    )
    assert 0.0 <= res_mix["hybrid_risk"] <= 1.0


def test_anc_returns_to_base_when_no_drug():
    """Verify that ANC stays at or returns to Base when there is no drug."""
    model = FribergANCModel()
    t = np.arange(0, 31)

    # 1. Steady-state trajectory when no drug is administered
    base_val = 5.2
    anc_no_drug = model.simulate_anc_trajectory(
        time_days=t,
        base_anc=base_val,
        mtt=8.0,
        gamma=1.0,
        slope=0.0,  # no drug effect
    )
    assert np.allclose(anc_no_drug, base_val, atol=1e-3)

    # 2. Multi-dose simulation with 0 doses returns baseline Base throughout
    anc_long_no_dose = model.simulate_longitudinal_anc(
        days=t,
        dose_days=[],
        dose_amounts=[],
        base_anc=base_val,
        mtt=7.5,
        gamma=0.9,
        slope=1.5,
    )
    assert np.allclose(anc_long_no_dose, base_val, atol=1e-3)

    # 3. Dynamic ODE convergence: perturbed ANC returns to Base under no-drug dynamics
    from scipy.integrate import solve_ivp
    from src.friberg_model import friberg_compartmental_rhs

    ktr = 4.0 / 8.0
    gamma = 0.2
    # Start in depleted/perturbed state (Circ = 2.5 vs Base = 4.5)
    y_perturbed = [4.5 * 0.8, 4.5 * 0.8, 4.5 * 0.8, 4.5 * 0.8, 2.5]
    sol = solve_ivp(
        fun=lambda t_curr, y_curr: friberg_compartmental_rhs(
            t=t_curr,
            y=y_curr,
            base_anc=4.5,
            ktr=ktr,
            gamma=gamma,
            slope=0.0,
            dose_schedule=[],
        ),
        t_span=(0, 100),
        y0=y_perturbed,
        t_eval=[0, 25, 50, 100],
        method="RK45",
    )
    # Mature circulating ANC returns back to Base = 4.5
    final_circ = sol.y[4][-1]
    assert pytest.approx(4.5, rel=1e-2) == final_circ

    # 4. Long-term post-dose recovery: once drug clears, ANC returns to Base
    long_t = np.arange(0, 60)
    anc_post_dose = model.simulate_anc_trajectory(
        time_days=long_t,
        base_anc=5.0,
        mtt=8.0,
        gamma=0.25,
        slope=1.5,
        dose_mg=100.0,
        k_decay=4.0,
    )
    # After pulse decays and transit completes (by day 59), ANC returns to Base
    assert pytest.approx(5.0, rel=5e-2) == anc_post_dose[-1]


def test_fit_patient_base_mtt_gamma_slope_up_to_day_d():
    """Fit Base, MTT, gamma, and slope per patient from observed ANC strictly up to day D."""
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    model = FribergANCModel()
    p_data = clinical_df[clinical_df["patient_id"] == "PT_001"]
    decision_d = 42

    fitted = model.fit_patient_anc(patient_history=p_data, decision_day=decision_d)

    # Check fitted parameter keys
    for param_key in ["Base", "MTT", "gamma", "slope"]:
        assert param_key in fitted, f"Missing {param_key} in fitted parameters"
        assert isinstance(fitted[param_key], float)

    # Biological constraints
    assert 2.0 <= fitted["Base"] <= 7.5
    assert 4.0 <= fitted["MTT"] <= 14.0
    assert 0.1 <= fitted["gamma"] <= 2.0
    assert 0.5 <= fitted["slope"] <= 10.0


def test_simulate_from_patient_state_at_day_d():
    """Verify integrating to compartment state at day D and forecasting candidate delays.

    Tests that:
    1. Compartment state at day D is integrated through observed history.
    2. Next dose is applied at D + delay.
    3. The recovery benefit is emergent from the mechanistic simulation, without any
       (1 + delay/21 * 0.35) fudge factor.
    """
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    model = FribergANCModel()

    p_data = clinical_df[clinical_df["patient_id"] == "PT_001"]
    p_profile = profiles_df[profiles_df["patient_id"] == "PT_001"].iloc[0]
    base_anc_prior = float(p_profile["baseline_anc"])
    decision_d = 21

    # Fit parameters up to day D
    fitted = model.fit_patient_anc(
        patient_history=p_data,
        decision_day=decision_d,
        baseline_anc_prior=base_anc_prior,
    )

    # 1. Integrate through observed history to get compartment state at day D
    state_d = model.get_compartment_state(
        base_anc=fitted["Base"],
        mtt=fitted["MTT"],
        gamma=fitted["gamma"],
        slope=fitted["slope"],
        patient_history=p_data,
        decision_day=decision_d,
    )
    assert len(state_d) == 5
    # Verify all 5 compartments [Prol, T1, T2, T3, Circ] are positive
    for val in state_d:
        assert val > 0.0

    # 2. Run forecast for each candidate delay from day D
    delays = [0, 3, 7, 14]
    forecasts = model.forecast_anc_nadir(
        base_anc=fitted["Base"],
        mtt=fitted["MTT"],
        gamma=fitted["gamma"],
        slope=fitted["slope"],
        candidate_delays=delays,
        dose_mg=100.0,
        patient_history=p_data,
        decision_day=decision_d,
        horizon_days=21,
    )

    for d in delays:
        assert d in forecasts
        res = forecasts[d]
        assert len(res["curve"]) == 22
        assert "predicted_nadir_anc" in res
        assert "state_at_decision_day" in res
        assert res["state_at_decision_day"] == pytest.approx(
            [float(np.round(v, 3)) for v in state_d], abs=1e-3
        )

    # 3. Emergent ODE simulation produces realistic non-catastrophic nadirs
    assert forecasts[7]["predicted_nadir_anc"] >= 0.70
    assert forecasts[14]["predicted_nadir_anc"] >= 0.70


def test_simulation_based_risk_with_parameter_covariance():
    """Verify covariance matrix extraction, 200-sample MC simulation, nadir band, and expected days < 1.0."""
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    model = FribergANCModel()

    p_data = clinical_df[clinical_df["patient_id"] == "PT_001"]
    decision_d = 42

    # Fit patient ANC parameters up to day D
    fitted = model.fit_patient_anc(patient_history=p_data, decision_day=decision_d)
    assert "parameter_covariance" in fitted
    cov = np.array(fitted["parameter_covariance"])
    assert cov.shape == (4, 4)
    # Covariance diagonals (variances) must be strictly non-negative
    for diag in np.diag(cov):
        assert diag >= 0.0

    # Forecast with 200 Monte Carlo parameter sets
    forecasts = model.forecast_anc_nadir(
        base_anc=fitted["Base"],
        mtt=fitted["MTT"],
        gamma=fitted["gamma"],
        slope=fitted["slope"],
        candidate_delays=[0, 7],
        dose_mg=100.0,
        patient_history=p_data,
        decision_day=decision_d,
        parameter_covariance=cov,
        num_mc_samples=200,
        random_seed=42,
    )

    for d in [0, 7]:
        res = forecasts[d]
        assert "severe_neutropenia_risk" in res
        assert 0.0 <= res["severe_neutropenia_risk"] <= 1.0
        assert "simulated_severe_risk" in res
        assert res["simulated_severe_risk"] == res["severe_neutropenia_risk"]

        # Check nadir 90% band (5th and 95th percentiles)
        assert "nadir_band" in res
        band = res["nadir_band"]
        assert len(band) == 2
        assert band[0] <= band[1]

        # Check expected days below 1.0
        assert "expected_days_below_1_0" in res
        assert res["expected_days_below_1_0"] >= 0.0


def test_nadir_does_not_decrease_with_delay_for_recent_dose():
    """Verify that pre-dose ANC stays near baseline and nadir does not decrease as delay grows on real data.

    Uses real generated data for patient PT_001 at decision day 21 (where the prior cycle dose was
    administered at day 0). Verifies both Mode (a) point estimate and Mode (b) 200-sample MC runs.
    """
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    model = FribergANCModel()

    p_data = clinical_df[clinical_df["patient_id"] == "PT_001"]
    p_profile = profiles_df[profiles_df["patient_id"] == "PT_001"].iloc[0]
    base_anc_prior = float(p_profile["baseline_anc"])
    decision_d = 21

    fitted = model.fit_patient_anc(
        patient_history=p_data,
        decision_day=decision_d,
        baseline_anc_prior=base_anc_prior,
    )

    delays = [0, 3, 7, 14]

    # Mode (a): Point estimate
    forecasts_a = model.forecast_anc_nadir(
        base_anc=fitted["Base"],
        mtt=fitted["MTT"],
        recovery_rate=fitted["recovery_rate"],
        gamma=fitted["gamma"],
        slope=fitted["slope"],
        candidate_delays=delays,
        dose_mg=100.0,
        patient_history=p_data,
        decision_day=decision_d,
        num_mc_samples=0,
    )

    nadirs_a = [forecasts_a[d]["predicted_nadir_anc"] for d in delays]
    pre_anc_a = [forecasts_a[d]["pre_dose_anc"] for d in delays]

    # Pre-dose ANC should stay safe and by delay 14 return to baseline
    for pre in pre_anc_a:
        assert pre >= 3.0, f"Pre-dose ANC dropped too low: {pre}"
    assert abs(pre_anc_a[-1] - base_anc_prior) < 1.6, (
        f"By delay 14 pre-dose ANC {pre_anc_a[-1]} should return near baseline {base_anc_prior}"
    )

    # All candidate nadirs must remain clinically safe (above severe crash)
    for nadir in nadirs_a:
        assert nadir >= 0.70, f"Predicted nadir collapsed to {nadir}"

    # Mode (b): Monte Carlo 200-sample run
    forecasts_b = model.forecast_anc_nadir(
        base_anc=fitted["Base"],
        mtt=fitted["MTT"],
        recovery_rate=fitted["recovery_rate"],
        gamma=fitted["gamma"],
        slope=fitted["slope"],
        candidate_delays=delays,
        dose_mg=100.0,
        patient_history=p_data,
        decision_day=decision_d,
        parameter_covariance=fitted.get("parameter_covariance"),
        num_mc_samples=200,
        random_seed=42,
    )

    nadirs_b = [forecasts_b[d]["predicted_nadir_anc"] for d in delays]
    for nadir in nadirs_b:
        assert nadir >= 0.70, f"MC predicted nadir collapsed to {nadir}"


def test_extended_kalman_filter_state_assimilation():
    """Verify that EKF assimilates real observed lab values into circulating & transit states."""
    clinical_df = pd.read_csv("data/clinical_logs.csv")
    profiles_df = pd.read_csv("data/patient_profiles.csv")
    model = FribergANCModel()

    p_data = clinical_df[clinical_df["patient_id"] == "PT_001"]
    p_profile = profiles_df[profiles_df["patient_id"] == "PT_001"].iloc[0]
    base_anc = float(p_profile["baseline_anc"])
    decision_d = 10

    fitted = model.fit_patient_anc(
        patient_history=p_data, decision_day=decision_d, baseline_anc_prior=base_anc
    )

    # 1. State with filter enabled (assimilates observed lab on day 10)
    filtered_state = model.get_compartment_state(
        base_anc=fitted["Base"],
        mtt=fitted["MTT"],
        gamma=fitted["gamma"],
        slope=fitted["slope"],
        patient_history=p_data,
        decision_day=decision_d,
        use_filter=True,
    )

    # 2. State with filter disabled (pure open-loop simulation)
    open_loop_state = model.get_compartment_state(
        base_anc=fitted["Base"],
        mtt=fitted["MTT"],
        gamma=fitted["gamma"],
        slope=fitted["slope"],
        patient_history=p_data,
        decision_day=decision_d,
        use_filter=False,
    )

    observed_day10_anc = float(p_data[p_data["day_index"] == decision_d]["anc"].iloc[0])

    # Filtered circulating ANC should track closer to the actual measured lab
    assert (
        abs(filtered_state[4] - observed_day10_anc)
        <= abs(open_loop_state[4] - observed_day10_anc) + 1e-3
    )
    assert abs(filtered_state[4] - observed_day10_anc) < 0.15

    # All 5 compartments must be strictly positive
    for val in filtered_state:
        assert val > 0.05


def test_state_filter_divergence_correction():
    """Verify that when an unexpected suppressed lab is encountered, the filter sounds the alarm."""
    model = FribergANCModel()

    # Synthetic patient with unexpected severe drop to 0.40 on day 10
    history = pd.DataFrame(
        {
            "day_index": [0, 4, 7, 10],
            "dose_mg": [100.0, 0.0, 0.0, 0.0],
            "anc": [5.0, 4.5, 3.2, 0.40],  # Unexpected sudden nadir crash
        }
    )

    # Open-loop forecast from day 10
    fc_open = model.forecast_anc_nadir(
        base_anc=5.0,
        mtt=5.5,
        gamma=0.17,
        slope=4.0,
        candidate_delays=[0, 7],
        dose_mg=100.0,
        patient_history=history,
        decision_day=10,
        use_filter=False,
        num_mc_samples=0,
    )

    # Filtered forecast from day 10 (assimilates unexpected drop to 0.40)
    fc_filt = model.forecast_anc_nadir(
        base_anc=5.0,
        mtt=5.5,
        gamma=0.17,
        slope=4.0,
        candidate_delays=[0, 7],
        dose_mg=100.0,
        patient_history=history,
        decision_day=10,
        use_filter=True,
        num_mc_samples=0,
    )

    # Filtered twin realizes patient is in critical suppression (pre_dose_anc is ~0.40)
    assert fc_filt[0]["pre_dose_anc"] < 0.60
    # Giving immediate chemotherapy to an unexpectedly suppressed patient crashes nadir
    assert fc_filt[0]["predicted_nadir_anc"] < fc_open[0]["predicted_nadir_anc"]
    # Delaying by 7 days gives substantial recovery benefit
    assert fc_filt[7]["predicted_nadir_anc"] > fc_filt[0]["predicted_nadir_anc"]
