import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def compute_post_dose_anc_trajectory(
    dose_day: int,
    base_anc: float,
    total_severity: float,
    recovery_rate: float,
    horizon_days: int = 21,
    noise_std: float = 0.1,
) -> np.ndarray:
    """Compute post-dose daily ANC trajectory using the standard nadir-curve function.

    Shared by BOTH real doses and counterfactual delayed doses:
    - Peak marrow suppression occurs around days 7-10 (nadir).
    - Recovery rate per patient governs rebound back toward baseline reserves.
    """
    anc_curve = np.zeros(horizon_days, dtype=float)
    for dt in range(horizon_days):
        # Nadir response function peaking around day 8
        nadir_shape = (dt / 8.0) * np.exp(1.0 - (dt / 8.0))
        # Drop governed by severity, tempered by patient recovery rate
        drop = (total_severity / max(0.5, recovery_rate)) * nadir_shape
        noise = np.random.normal(0, noise_std) if noise_std > 0 else 0.0
        anc_curve[dt] = max(0.15, base_anc - drop + noise)
    return anc_curve


def generate_synthetic_data(
    num_patients: int = 150,
    days: int = 90,
    seed: int = 42,
    output_dir: str = "data",
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

    patient_ids = [f"PT_{i+1:03d}" for i in range(num_patients)]
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

        # Each patient has their own random physiological marrow recovery rate
        p_recovery_rate = float(np.clip(np.random.normal(1.0, 0.18), 0.65, 1.45))

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
        daily_physio_stress = np.zeros(days, dtype=float)

        clinical_factor = (
            (planned_dose / 100.0) * 1.5 + (age / 75.0) * 0.7 + ecog * 0.5
        )

        # 1. Simulate real doses on patient's timeline using ONE shared nadir curve
        for d in dose_schedule:
            if d < days:
                c_vuln = cycle_vulnerabilities[d]

                # Pre-dose biological stress manifesting in 7 days before dose:
                # Modulated by cycle vulnerability and recovery rate (r_i)
                pre_dose_impairment = max(
                    0.0, (c_vuln * 0.75) + (1.0 - p_recovery_rate) * 1.2
                )
                for pre_t in range(max(0, d - 7), d):
                    daily_physio_stress[pre_t] += pre_dose_impairment * 0.55

                total_severity = max(
                    0.4,
                    clinical_factor
                    + (c_vuln * 0.85)
                    + (1.0 - p_recovery_rate) * 0.8
                    + np.random.normal(0, 0.15),
                )

                # Generate post-dose trajectory using the unified function
                horizon = min(21, days - d)
                post_anc = compute_post_dose_anc_trajectory(
                    dose_day=d,
                    base_anc=base_anc,
                    total_severity=total_severity,
                    recovery_rate=p_recovery_rate,
                    horizon_days=horizon,
                    noise_std=0.08,
                )

                for dt in range(horizon):
                    t = d + dt
                    drop = max(0.0, base_anc - post_anc[dt])
                    latent_anc[t] = min(latent_anc[t], post_anc[dt])
                    daily_physio_stress[t] += (drop / base_anc) * 1.5

        # Also inject pre-decision stress for counterfactual evaluation decision days
        for _, eval_day in eval_cycles:
            if eval_day not in dose_schedule and eval_day < days:
                eval_vuln = cycle_vulnerabilities[eval_day]
                eval_impairment = max(
                    0.0, (eval_vuln * 0.75) + (1.0 - p_recovery_rate) * 1.2
                )
                for pre_t in range(max(0, eval_day - 7), eval_day):
                    daily_physio_stress[pre_t] += eval_impairment * 0.55

        # 2. Record daily clinical & wearable observations
        # Autonomic signals reflect personal baseline + hidden marrow stress state
        for day in range(days):
            date_str = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")
            is_dosing_day = day in dose_schedule
            tx_admin = regimen if is_dosing_day else "None"
            dose_val = planned_dose if is_dosing_day else 0.0

            # True latent tumor progression
            growth = gomp_a * curr_tumor_vol * np.log(
                max(1.001, gomp_k / curr_tumor_vol)
            )
            curr_tumor_vol = min(gomp_k, max(0.5, curr_tumor_vol + growth))
            if is_dosing_day:
                curr_tumor_vol = max(0.2, curr_tumor_vol * (1.0 - gomp_kill))

            # Observed tumor volume with measurement noise (e.g. imaging variance)
            obs_tumor_vol = max(
                0.2,
                np.round(curr_tumor_vol + np.random.normal(0, 0.45), 2),
            )

            obs_anc = max(0.1, np.round(latent_anc[day] + np.random.normal(0, 0.1), 2))
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

        # 3. Simulate counterfactual intra-patient timing choices for cycles 1 & 2
        # Uses the unified nadir-curve function
        for cycle_num, planned_day in eval_cycles:
            c_vuln = cycle_vulnerabilities[planned_day]

            for delay in candidate_delays:
                cand_day = planned_day + delay

                # Delay allows recovery proportional to patient rate r_i
                stochastic_noise = float(np.random.normal(0.0, 0.08))
                base_rec = (delay / 14.0) * p_recovery_rate * 0.95
                effective_recovery = base_rec + stochastic_noise
                effective_vuln = max(-1.5, c_vuln - effective_recovery)

                cand_severity = max(
                    0.3,
                    clinical_factor
                    + (effective_vuln * 0.85)
                    + (1.0 - p_recovery_rate) * 0.8
                    + np.random.normal(0, 0.12),
                )

                # Generate simulated daily ANC in 7 days after candidate dose day
                cand_post_anc = compute_post_dose_anc_trajectory(
                    dose_day=cand_day,
                    base_anc=base_anc,
                    total_severity=cand_severity,
                    recovery_rate=p_recovery_rate,
                    horizon_days=8,
                    noise_std=0.08,
                )

                # Post-dose 7-day window nadir ANC (days 1 to 7 after candidate dose)
                post_7d_anc = cand_post_anc[1:8]
                post_nadir_anc = float(np.min(post_7d_anc))
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

    # Calculate post-dose 7-day severe neutropenia label incidence rate
    dose_events = clinical_df[clinical_df["dose_mg"] > 0]
    total_doses = len(dose_events)
    severe_events = 0

    for _, row in dose_events.iterrows():
        p_id = row["patient_id"]
        d = int(row["day_index"])
        window = clinical_df[
            (clinical_df["patient_id"] == p_id)
            & (clinical_df["day_index"] > d)
            & (clinical_df["day_index"] <= d + 7)
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
    args = parser.parse_args()

    generate_synthetic_data(
        num_patients=args.patients,
        days=args.days,
        seed=args.seed,
        output_dir=args.out,
    )
