import argparse
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.friberg_model import ODE_KW, friberg_compartmental_rhs  # noqa: E402


# Severe-neutropenia label window: the whole cycle after a dose. With a
# maturation delay the nadir falls on ~days 10-14, so the old 7-day window
# would miss it.
LABEL_WINDOW_DAYS = 21

# Sparse lab schedule relative to each dose day (CBC draws).
LAB_OFFSETS = (0, 4, 7, 10, 14, 17)


def _simulate_segment(
    state, t_start, t_end, base, mtt, gamma, slope, dose_day, k_decay=0.7
):
    """Integrate the Friberg marrow from t_start to t_end (integer days).

    Returns (daily_states[ (t_end - t_start + 1), 5 ], final_state).
    """
    t_eval = np.arange(t_start, t_end + 1, dtype=float)
    doses = [(float(dose_day), 100.0)] if dose_day is not None else []
    if t_end <= t_start:
        return np.array([state]), list(state)
    sol = solve_ivp(
        lambda t, y: friberg_compartmental_rhs(
            t, y, base, 4.0 / mtt, gamma, slope, doses, k_decay
        ),
        (float(t_start), float(t_end)),
        list(state),
        t_eval=t_eval,
        **ODE_KW,
    )
    states = sol.y.T
    return states, list(states[-1])


def sample_virtual_patient_marrow(age: float, ecog: int) -> dict:
    """Draw hidden, patient-specific Friberg parameters (never exported)."""
    mtt = float(np.clip(np.random.normal(5.5, 0.6), 4.2, 8.0))
    gamma = float(np.clip(np.random.normal(0.17, 0.025), 0.12, 0.24))
    sens = float(
        np.clip(
            2.6
            * np.exp(0.5 * (age - 55.0) / 20.0 + 0.25 * ecog)
            * np.random.lognormal(0.0, 0.30),
            0.8,
            8.0,
        )
    )
    return {"mtt": mtt, "gamma": gamma, "slope": sens}


def cycle_slope(p_slope: float, dose_idx: int, vuln: float) -> float:
    """Effective drug sensitivity for a given cycle (cumulative damage + vulnerability)."""
    return float(
        np.clip(p_slope * (1.0 + 0.12 * dose_idx) * np.exp(0.35 * vuln), 0.3, 25.0)
    )


def generate_synthetic_data(
    num_patients: int = 150,
    days: int = 90,
    seed: int = 42,
    output_dir: str = "data",
    coupling_kappa: float = 1.0,
):
    """Generate synthetic oncology digital twin longitudinal datasets.

    Wearable biometrics serve as noisy lagged proxies of each patient's hidden
    marrow state: per-patient recovery rate r_i and cycle-specific biological
    vulnerability.
    - Autonomic strain (RHR rises, HRV dips, steps drop) reflects hidden marrow state.
    - Sleep duration contains uncoupled non-treatment confounders.
    - Ground-truth labels and latent ANC are never directly exposed as features.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    patient_ids = [f"PT_{i + 1:03d}" for i in range(num_patients)]
    ages = np.random.randint(30, 76, size=num_patients)
    sexes = np.random.choice(["F", "M"], size=num_patients, p=[0.7, 0.3])
    cancer_types = ["Breast Cancer"] * num_patients
    stages = np.random.choice(
        ["II", "III", "IV"], size=num_patients, p=[0.4, 0.45, 0.15]
    )
    biomarkers = np.random.choice(
        ["HER2+", "ER+/PR+", "Triple Negative"],
        size=num_patients,
        p=[0.25, 0.55, 0.20],
    )
    baseline_ecogs = np.random.choice(
        [0, 1, 2], size=num_patients, p=[0.45, 0.45, 0.10]
    )
    baseline_ancs = np.round(np.random.uniform(2.8, 6.5, size=num_patients), 2)

    profiles_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
            "sex": sexes,
            "cancer_type": cancer_types,
            "stage": stages,
            "biomarker_status": biomarkers,
            "baseline_ecog": baseline_ecogs,
            "baseline_anc": baseline_ancs,
        }
    )

    profiles_path = os.path.join(output_dir, "patient_profiles.csv")
    profiles_df.to_csv(profiles_path, index=False)

    clinical_rows = []
    wearable_rows = []
    timing_rows = []
    start_date = datetime(2025, 1, 1)

    candidate_delays = [0, 3, 7, 14]

    for idx, p_id in enumerate(patient_ids):
        age = ages[idx]
        ecog = baseline_ecogs[idx]
        base_anc = baseline_ancs[idx]

        # Individual baseline physiology
        p_base_hr = np.random.normal(70, 5)
        p_base_hrv = np.random.normal(48, 8)
        p_base_steps = np.random.normal(6500, 1000)
        p_base_sleep = np.random.normal(440, 35)

        # Baseline tumor volume and patient-specific Gompertzian kinetics
        initial_vol = (
            12.0 if stages[idx] == "II" else (22.0 if stages[idx] == "III" else 35.0)
        )
        curr_tumor_vol = initial_vol
        # True biological parameters vary per patient
        gomp_a = float(np.clip(np.random.normal(0.042, 0.007), 0.025, 0.065))
        gomp_k = 100.0
        gomp_kill = float(np.clip(np.random.normal(0.35, 0.05), 0.20, 0.50))

        # Hidden per-patient marrow physiology (Friberg parameters)
        marrow = sample_virtual_patient_marrow(age, ecog)
        p_mtt, p_gamma, p_slope = marrow["mtt"], marrow["gamma"], marrow["slope"]

        # Randomized real dosing intervals:
        c1_day = 0
        real_delta1 = int(np.random.randint(0, 15))
        c2_day = 21 + real_delta1

        real_delta2 = int(np.random.randint(0, 15))
        c3_day = c2_day + 21 + real_delta2

        c4_day = c3_day + 21

        dose_schedule = [c1_day, c2_day, c3_day]
        if c4_day < days:
            dose_schedule.append(c4_day)

        regimen = "Regimen A"
        planned_dose = 100.0

        # Pre-assign cycle-varying latent recovery state for each cycle
        cycle_vulnerabilities = {}
        for d in dose_schedule:
            cycle_vulnerabilities[d] = float(np.random.normal(0.0, 1.0))

        # Planned nominal evaluation days for counterfactual timing
        eval_cycles = [(1, 21), (2, 42)]
        for _, p_day in eval_cycles:
            if p_day not in cycle_vulnerabilities:
                cycle_vulnerabilities[p_day] = float(np.random.normal(0.0, 1.0))

        latent_anc = np.full(days, base_anc, dtype=float)
        marrow_states = np.tile(np.full(5, base_anc), (days, 1))
        daily_physio_stress = np.zeros(days, dtype=float)
        gamma_deficit = (0.20 - p_gamma) / 0.10  # >0 = slower-recovering marrow

        # 1. Simulate the marrow continuously through the real dosing schedule.
        # Cumulative damage: each cycle lowers marrow reserve (feedback target)
        # and raises effective drug sensitivity.
        real_doses = [d for d in dose_schedule if d < days]
        seg_bounds = real_doses + [days - 1]
        state = [base_anc] * 5
        for cycle_idx, d in enumerate(real_doses):
            c_vuln = cycle_vulnerabilities[d]
            cumulative_marrow_damage = 0.04 * cycle_idx
            cycle_base_anc = max(1.8, base_anc * (1.0 - cumulative_marrow_damage))
            c_slope = cycle_slope(p_slope, cycle_idx, c_vuln)
            seg_end = seg_bounds[cycle_idx + 1]
            states, state = _simulate_segment(
                state, d, seg_end, cycle_base_anc, p_mtt, p_gamma, c_slope, d
            )
            # Keep the pre-dose state on the next dose day (dose acts after t)
            n_fill = seg_end - d + (1 if cycle_idx == len(real_doses) - 1 else 0)
            marrow_states[d : d + n_fill] = states[:n_fill]

            # Pre-dose biological stress manifesting in the 7 days before dose
            pre_dose_impairment = max(
                0.0,
                (c_vuln * 0.75) + gamma_deficit * 0.5 + cumulative_marrow_damage * 2.0,
            )
            for pre_t in range(max(0, d - 7), d):
                daily_physio_stress[pre_t] += pre_dose_impairment * 0.55

        latent_anc = np.maximum(0.05, marrow_states[:, 4])
        # Upstream progenitor & early transit exhaustion (hidden from circulating blood)
        progenitor_pool = 0.6 * marrow_states[:, 0] + 0.4 * marrow_states[:, 1]
        upstream_depletion = np.clip(
            (base_anc - progenitor_pool) / base_anc, 0.0, 1.2
        )
        daily_physio_stress += 2.2 * upstream_depletion

        lab_days = {0}
        for d in real_doses:
            lab_days.update(d + o for o in LAB_OFFSETS if d + o < days)

        # Also inject pre-decision stress for counterfactual evaluation decision days
        for cycle_num, eval_day in eval_cycles:
            if eval_day not in dose_schedule and eval_day < days:
                eval_vuln = cycle_vulnerabilities[eval_day]
                cum_damage = 0.04 * cycle_num
                eval_impairment = max(
                    0.0,
                    (eval_vuln * 0.75) + gamma_deficit * 0.5 + cum_damage * 2.0,
                )
                for pre_t in range(max(0, eval_day - 7), eval_day):
                    daily_physio_stress[pre_t] += eval_impairment * 0.55

        # Scaled by autonomic coupling strength kappa (0.0 = completely uncoupled, 1.0 = physiological coupling)
        daily_physio_stress = float(coupling_kappa) * daily_physio_stress

        # 2. Record daily clinical & wearable observations
        # Autonomic signals reflect personal baseline + hidden marrow stress state
        for day in range(days):
            date_str = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")
            is_dosing_day = day in dose_schedule
            tx_admin = regimen if is_dosing_day else "None"
            dose_val = planned_dose if is_dosing_day else 0.0

            # True latent tumor progression
            growth = (
                gomp_a * curr_tumor_vol * np.log(max(1.001, gomp_k / curr_tumor_vol))
            )
            curr_tumor_vol = min(gomp_k, max(0.5, curr_tumor_vol + growth))
            if is_dosing_day:
                curr_tumor_vol = max(0.2, curr_tumor_vol * (1.0 - gomp_kill))

            # Observed tumor volume with measurement noise (e.g. imaging variance)
            obs_tumor_vol = max(
                0.2,
                np.round(curr_tumor_vol + np.random.normal(0, 0.45), 2),
            )

            # Lab measurement error ~ 8% CV + small absolute noise
            obs_anc = max(
                0.05,
                np.round(
                    latent_anc[day] * np.random.lognormal(0.0, 0.08)
                    + np.random.normal(0, 0.05),
                    2,
                ),
            )
            obs_wbc = max(0.8, np.round(obs_anc * 2.2 + np.random.normal(0, 0.25), 2))

            clinical_rows.append(
                {
                    "patient_id": p_id,
                    "day_index": day,
                    "treatment_administered": tx_admin,
                    "dose_mg": dose_val,
                    "anc": obs_anc,
                    "wbc": obs_wbc,
                    "tumor_volume_cm3": obs_tumor_vol,
                    "is_lab_day": int(day in lab_days),
                }
            )

            # Wearable signals: Noisy lagged proxies of hidden physiological strain
            stress = daily_physio_stress[day]
            # RHR rises when hidden marrow reserve is strained
            daily_hr = p_base_hr + (stress * 12.0) + np.random.normal(0, 2.5)
            # HRV dips when hidden marrow reserve is strained
            daily_hrv = max(
                8.0, p_base_hrv - (stress * 15.0) + np.random.normal(0, 3.0)
            )
            # Daily steps drop with physical fatigue / biological stress
            daily_steps = max(
                300.0, p_base_steps - (stress * 2200.0) + np.random.normal(0, 450.0)
            )
            # Sleep duration acts as an uncoupled confounder (unrelated to marrow state)
            confounder_noise = np.random.normal(0, 35.0)
            daily_sleep = max(180.0, p_base_sleep + confounder_noise)

            wearable_rows.append(
                {
                    "patient_id": p_id,
                    "date": date_str,
                    "day_index": day,
                    "resting_heart_rate": np.round(daily_hr, 1),
                    "hrv_rmssd": np.round(daily_hrv, 1),
                    "step_count": int(daily_steps),
                    "sleep_duration_mins": int(daily_sleep),
                }
            )

        # 3. Counterfactual timing: start from the patient's TRUE marrow state at
        # the decision day and simulate each candidate delay with the same ODE.
        for cycle_num, planned_day in eval_cycles:
            c_vuln = cycle_vulnerabilities[planned_day]
            dose_idx = cycle_num  # day-21 decision = 2nd administration
            cum_marrow_damage = 0.04 * dose_idx
            eval_base_anc = max(1.8, base_anc * (1.0 - cum_marrow_damage))
            state_D = list(marrow_states[min(planned_day, days - 1)])
            prev_base = max(1.8, base_anc * (1.0 - 0.04 * (dose_idx - 1)))

            for delay in candidate_delays:
                cand_day = planned_day + delay
                # Waiting lets transient vulnerability fade (plus natural regrowth via the ODE)
                effective_vuln = max(
                    -1.5,
                    c_vuln - (delay / 14.0) * 0.9 + float(np.random.normal(0.0, 0.08)),
                )
                c_slope = cycle_slope(p_slope, dose_idx, effective_vuln)
                # Recover undisturbed until the candidate dose day
                _, st = _simulate_segment(
                    state_D, planned_day, cand_day, prev_base, p_mtt, p_gamma, 0.0, None
                )
                post_states, _ = _simulate_segment(
                    st,
                    cand_day,
                    cand_day + LABEL_WINDOW_DAYS,
                    eval_base_anc,
                    p_mtt,
                    p_gamma,
                    c_slope,
                    cand_day,
                )
                post_anc = post_states[1:, 4] * np.random.lognormal(
                    0.0, 0.08, LABEL_WINDOW_DAYS
                )
                post_nadir_anc = float(np.min(post_anc))
                toxicity_occurred = 1 if post_nadir_anc < 1.0 else 0

                timing_rows.append(
                    {
                        "patient_id": p_id,
                        "cycle": cycle_num,
                        "planned_day": planned_day,
                        "delay_days": delay,
                        "candidate_day": cand_day,
                        "severe_toxicity_label": toxicity_occurred,
                        "nadir_anc": round(post_nadir_anc, 2),
                    }
                )

    clinical_df = pd.DataFrame(clinical_rows)
    wearable_df = pd.DataFrame(wearable_rows)
    timing_df = pd.DataFrame(timing_rows)

    clinical_df.to_csv(os.path.join(output_dir, "clinical_logs.csv"), index=False)
    wearable_df.to_csv(os.path.join(output_dir, "wearable_streams.csv"), index=False)
    timing_df.to_csv(os.path.join(output_dir, "timing_eval.csv"), index=False)

    # Severe neutropenia incidence over the post-dose cycle window
    dose_events = clinical_df[clinical_df["dose_mg"] > 0]
    total_doses = len(dose_events)
    severe_events = 0

    for _, row in dose_events.iterrows():
        p_id = row["patient_id"]
        d = int(row["day_index"])
        window = clinical_df[
            (clinical_df["patient_id"] == p_id)
            & (clinical_df["day_index"] > d)
            & (clinical_df["day_index"] <= d + LABEL_WINDOW_DAYS)
        ]
        if (window["anc"] < 1.0).any():
            severe_events += 1

    incidence_rate = severe_events / total_doses if total_doses > 0 else 0

    print("Data generation with noisy lagged proxies complete.")
    print(f"Patients: {num_patients}, Days: {days}")
    print(f"Total Dosing Events: {total_doses}")
    print(f"Severe Toxicity Count: {severe_events} ({incidence_rate:.2%})")
    print(f"Timing Evaluation Dosing Choices Generated: {len(timing_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=int, default=150)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--coupling_kappa", type=float, default=1.0)
    args = parser.parse_args()

    generate_synthetic_data(
        num_patients=args.patients,
        days=args.days,
        seed=args.seed,
        output_dir=args.out,
        coupling_kappa=args.coupling_kappa,
    )
