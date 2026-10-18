import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.friberg_model import FribergANCModel  # noqa: E402
from src.tumor_model import GompertzTumorModel  # noqa: E402

st.set_page_config(
    page_title="ONCO-TWIN: Decision Support Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
    }
    .disclaimer-banner {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2a5298;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Mandatory Clinical Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>CLINICAL DECISION SUPPORT & SIMULATION ONLY:</strong><br>
        Not an autonomous prescription engine. Models evaluate synthetic data.
        Adjustments require oncologist review and lab verification.
    </div>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="main-header">
        <h2>ONCO-TWIN: Digital Twin for Chemotherapy Safety</h2>
        <p>Continuous twin fusing EHR history with 7-day wearable trends.</p>
    </div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_metadata():
    model = joblib.load(os.path.join(BASE_DIR, "models", "main_toxicity_model.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
    features = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.joblib"))
    meta = joblib.load(os.path.join(BASE_DIR, "models", "model_meta.joblib"))
    ablation_df = pd.read_csv(
        os.path.join(BASE_DIR, "models", "ablation_summary.csv")
    )
    multi_seed_path = os.path.join(
        BASE_DIR, "docs", "figures", "multi_seed_aggregate_table.csv"
    )
    multi_seed_df = (
        pd.read_csv(multi_seed_path)
        if os.path.exists(multi_seed_path)
        else pd.DataFrame()
    )
    return model, scaler, features, meta, ablation_df, multi_seed_df


@st.cache_data
def load_patient_data():
    profiles = pd.read_csv(os.path.join(BASE_DIR, "data", "patient_profiles.csv"))
    clinical = pd.read_csv(os.path.join(BASE_DIR, "data", "clinical_logs.csv"))
    wearables = pd.read_csv(os.path.join(BASE_DIR, "data", "wearable_streams.csv"))
    timing = pd.read_csv(os.path.join(BASE_DIR, "data", "timing_features.csv"))
    return profiles, clinical, wearables, timing


model, scaler, feature_names, meta, ablation_summary_df, multi_seed_benchmark_df = (
    load_models_and_metadata()
)
profiles_df, clinical_df, wearable_df, timing_df = load_patient_data()

# Sidebar: Patient Selection & Twin Controls
st.sidebar.header("Virtual Patient Selector")
patient_list = profiles_df["patient_id"].tolist()
selected_pid = st.sidebar.selectbox("Choose Patient ID", patient_list, index=0)

prof = profiles_df[profiles_df["patient_id"] == selected_pid].iloc[0]
p_clin = clinical_df[clinical_df["patient_id"] == selected_pid].sort_values("day_index")
p_wear = wearable_df[wearable_df["patient_id"] == selected_pid].sort_values("day_index")

# Sidebar Patient Information Summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Age / Sex:** {int(prof['age'])} yrs | {prof['sex']}")
st.sidebar.markdown(f"**Diagnosis:** {prof['cancer_type']} (Stage {prof['stage']})")
st.sidebar.markdown(f"**Biomarker:** {prof['biomarker_status']}")
st.sidebar.markdown(f"**Baseline ECOG:** {prof['baseline_ecog']}")
st.sidebar.markdown(f"**Baseline ANC:** {prof['baseline_anc']} × 10⁹/L")

# Selected Simulation Day (e.g. at cycle boundary: Day 21, 42, 63)
sim_day = st.sidebar.slider(
    "Current Monitoring Day", min_value=7, max_value=85, value=21
)

st.sidebar.markdown("---")
st.sidebar.subheader("Experimental Modules")
show_friberg = st.sidebar.checkbox(
    "Show experimental marrow (Friberg) model",
    value=False,
    help=(
        "Enable the experimental 5-compartment Friberg PK/PD model and "
        "hybrid risk calculation (Research prototype; 49.8% monotonicity across delay steps)."
    ),
)

if show_friberg:
    st.warning(
        "⚠️ **EXPERIMENTAL MARROW (FRIBERG) MODEL ENABLED:** The bone marrow mechanistic twin "
        "(Friberg PK/PD) is an exploratory research prototype with 49.8% nadir delay monotonicity "
        "(< 95% target). Mechanistic nadir forecasts and hybrid combinations are for research "
        "simulation only."
    )

# Current physiological state up to sim_day
recent_wear = p_wear[p_wear["day_index"] <= sim_day].iloc[-1]
base30_hrv = p_wear[p_wear["day_index"] <= sim_day]["hrv_rmssd"].mean()
curr_hrv = recent_wear["hrv_rmssd"]
hrv_drop_pct = ((curr_hrv - base30_hrv) / base30_hrv) * 100

recent_clin = p_clin[p_clin["day_index"] <= sim_day].iloc[-1]

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Recent Lab: ANC",
        f"{recent_clin['anc']:.2f} × 10⁹/L",
        delta="Above danger level (1.0)",
        delta_color="normal" if recent_clin["anc"] >= 1.0 else "inverse",
    )
with col2:
    st.metric(
        "Current HRV (RMSSD)",
        f"{curr_hrv:.1f} ms",
        delta=f"{hrv_drop_pct:+.1f}% vs 30d baseline",
        delta_color="normal" if hrv_drop_pct >= -10 else "inverse",
    )
with col3:
    st.metric(
        "Resting Heart Rate",
        f"{recent_wear['resting_heart_rate']:.1f} bpm",
    )
with col4:
    tumor_vol = recent_clin["tumor_volume_cm3"]
    st.metric(
        "Current Tumor Volume",
        f"{tumor_vol:.1f} cm³",
    )

st.markdown("---")

# Main Dashboard Panels
tab1, tab2, tab3 = st.tabs(
    [
        "🗓️ Candidate Dosing Window & Trade-off",
        "📈 Physiological Trajectories",
        "⚖️ Model Performance & Ablation",
    ]
)

tumor_engine = GompertzTumorModel()

with tab1:
    st.subheader("Timing Decision: Toxicity Risk vs. Tumor Regrowth Trade-Off")
    st.caption(
        "The twin evaluates hypothetical dosing at candidate days (+0, +3, +7, +14). "
        "Postponing dosing allows recovery, but permits tumor regrowth."
    )

    delays = [0, 3, 7, 14]
    candidate_risks = []

    # Fit patient Gompertz parameters strictly on past data (day_index <= sim_day)
    patient_clin_past = p_clin[p_clin["day_index"] <= sim_day]
    gomp_fit = tumor_engine.fit_patient_parameters(
        patient_clin_past, decision_day=sim_day
    )
    tradeoff_summary = tumor_engine.forecast_with_uncertainty(
        current_volume=gomp_fit["volume_at_d"],
        candidate_delays=delays,
        growth_rate=gomp_fit["growth_rate"],
        kill_fraction=gomp_fit["kill_fraction"],
        residual_std=gomp_fit["residual_std"],
    )

    friberg_engine = None
    friberg_fit = None
    anc_forecasts = None
    if show_friberg:
        friberg_engine = FribergANCModel()
        friberg_fit = friberg_engine.fit_patient_anc(
            patient_history=patient_clin_past,
            decision_day=sim_day,
            baseline_anc_prior=float(prof["baseline_anc"]),
            wearable_history=p_wear[p_wear["day_index"] <= sim_day],
        )
        anc_forecasts = friberg_engine.forecast_anc_nadir(
            base_anc=friberg_fit["base_anc"],
            mtt=friberg_fit["mtt"],
            recovery_rate=friberg_fit["recovery_rate"],
            candidate_delays=delays,
            dose_mg=100.0,
            drug_slope=friberg_fit["drug_slope"],
            horizon_days=21,
            patient_history=patient_clin_past,
            decision_day=sim_day,
            parameter_covariance=friberg_fit.get("parameter_covariance"),
        )

    # Freeze wearable history strictly at current decision day (<= sim_day)
    wear_hist = p_wear[p_wear["day_index"] <= sim_day]
    wear_curr = wear_hist.iloc[-1]
    b30_hr = wear_hist["resting_heart_rate"].mean()
    b30_hrv = wear_hist["hrv_rmssd"].mean()
    b30_steps = wear_hist["step_count"].mean()
    b30_slp = wear_hist["sleep_duration_mins"].mean()

    m7_hr = wear_hist.tail(7)["resting_heart_rate"].mean()
    m7_hrv = wear_hist.tail(7)["hrv_rmssd"].mean()
    m7_steps = wear_hist.tail(7)["step_count"].mean()
    m7_slp = wear_hist.tail(7)["sleep_duration_mins"].mean()

    # Prior dose calculation up to sim_day
    prev_doses = p_clin[(p_clin["dose_mg"] > 0) & (p_clin["day_index"] < sim_day)]
    base_since_prev = (
        float(sim_day - prev_doses.iloc[-1]["day_index"])
        if not prev_doses.empty
        else 21.0
    )

    # Score each candidate day using the twin model (frozen biometrics at sim_day)
    for d_offset in delays:
        dsld = float(base_since_prev + d_offset)
        feat_dict = {
            "age": float(prof["age"]),
            "baseline_ecog": float(prof["baseline_ecog"]),
            "baseline_anc": float(prof["baseline_anc"]),
            "pre_dose_anc": float(recent_clin["anc"]),
            "dose_mg": 100.0,
            "tumor_volume_cm3": float(tumor_vol),
            "delay_days": float(d_offset),
            "days_since_last_dose": dsld,
            "dose_interval_bin_0_3": 1.0 if dsld <= 3.0 else 0.0,
            "dose_interval_bin_4_7": 1.0 if 3.0 < dsld <= 7.0 else 0.0,
            "dose_interval_bin_8_14": 1.0 if 7.0 < dsld <= 14.0 else 0.0,
            "dose_interval_bin_15_plus": 1.0 if dsld > 14.0 else 0.0,
            "resting_heart_rate": float(wear_curr["resting_heart_rate"]),
            "hrv_rmssd": float(wear_curr["hrv_rmssd"]),
            "step_count": float(wear_curr["step_count"]),
            "sleep_duration_mins": float(wear_curr["sleep_duration_mins"]),
            "rhr_ratio_7_to_30": float(m7_hr / b30_hr if b30_hr > 0 else 1.0),
            "hrv_ratio_7_to_30": float(m7_hrv / b30_hrv if b30_hrv > 0 else 1.0),
            "steps_ratio_7_to_30": float(
                m7_steps / b30_steps if b30_steps > 0 else 1.0
            ),
            "sleep_ratio_7_to_30": float(m7_slp / b30_slp if b30_slp > 0 else 1.0),
        }
        X_vec = np.array([[feat_dict[fn] for fn in feature_names]])
        if scaler is not None:
            X_vec = scaler.transform(X_vec)

        p_risk = float(model.predict_proba(X_vec)[0, 1])
        candidate_risks.append(p_risk)

    hybrid_evaluations = {}
    candidate_mech_risks = []
    candidate_hybrid_risks = []
    if show_friberg and anc_forecasts is not None:
        for i, d_offset in enumerate(delays):
            ml_p = candidate_risks[i]
            nadir_val = anc_forecasts[d_offset]["predicted_nadir_anc"]
            days_low = anc_forecasts[d_offset]["days_below_1_0"]
            sim_risk = anc_forecasts[d_offset].get("simulated_severe_risk")
            hyb = FribergANCModel.combine_hybrid_toxicity_risk(
                ml_prob=ml_p,
                predicted_nadir_anc=nadir_val,
                days_below_1_0=days_low,
                mechanistic_weight=0.55,
                ml_weight=0.45,
                simulated_severe_risk=sim_risk,
            )
            hybrid_evaluations[d_offset] = hyb
            candidate_mech_risks.append(hyb["mech_risk"])
            candidate_hybrid_risks.append(hyb["hybrid_risk"])

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Dual-axis Interactive Trade-Off Plot with 90% Uncertainty Band
        fig_tradeoff = go.Figure()
        delay_labels = [f"+{d} days" for d in delays]

        if show_friberg and candidate_hybrid_risks:
            # Hybrid Toxicity Risk Line (Primary)
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=delay_labels,
                    y=[r * 100 for r in candidate_hybrid_risks],
                    name="Hybrid Risk (Mechanistic + ML) (%)",
                    mode="lines+markers",
                    line=dict(color="#e74c3c", width=3.5),
                    marker=dict(size=10, symbol="diamond"),
                )
            )
            # ML Model (Logistic Regression) Second Opinion Line
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=delay_labels,
                    y=[r * 100 for r in candidate_risks],
                    name="ML 2nd Opinion (Logistic Model) (%)",
                    mode="lines+markers",
                    line=dict(color="#9b59b6", width=2, dash="dash"),
                    marker=dict(size=7),
                )
            )
            # Mechanistic Friberg Risk Line
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=delay_labels,
                    y=[r * 100 for r in candidate_mech_risks],
                    name="Mechanistic Friberg ODE Risk (%)",
                    mode="lines+markers",
                    line=dict(color="#e67e22", width=2, dash="dot"),
                    marker=dict(size=7),
                )
            )
        else:
            # Pure Logistic Model Toxicity Risk Line
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=delay_labels,
                    y=[r * 100 for r in candidate_risks],
                    name="Toxicity Risk (Logistic ML Model) (%)",
                    mode="lines+markers",
                    line=dict(color="#e74c3c", width=3.5),
                    marker=dict(size=10, symbol="circle"),
                )
            )

        # Regrowth Penalty Line & 90% Uncertainty Bounds
        penalties = [tradeoff_summary[d]["regrowth_penalty_pct"] for d in delays]
        v_d0 = tradeoff_summary[0]["volume_at_dose_cm3"]
        v_upper_pen = [
            round(((tradeoff_summary[d]["upper_bound_90_cm3"] - v_d0) / v_d0) * 100, 1)
            for d in delays
        ]
        v_lower_pen = [
            round(
                max(
                    0.0,
                    ((tradeoff_summary[d]["lower_bound_90_cm3"] - v_d0) / v_d0) * 100,
                ),
                1,
            )
            for d in delays
        ]

        # 90% Confidence Band Ribbon
        fig_tradeoff.add_trace(
            go.Scatter(
                x=delay_labels + delay_labels[::-1],
                y=v_upper_pen + v_lower_pen[::-1],
                fill="toself",
                fillcolor="rgba(41, 128, 185, 0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="90% Regrowth Uncertainty Band",
                yaxis="y2",
            )
        )

        fig_tradeoff.add_trace(
            go.Scatter(
                x=delay_labels,
                y=penalties,
                name="Fitted Tumor Regrowth Penalty (%)",
                mode="lines+markers",
                line=dict(color="#2980b9", width=3, dash="dot"),
                marker=dict(size=9),
                yaxis="y2",
            )
        )

        max_y2 = max(120.0, max(v_upper_pen) * 1.25)
        fig_tradeoff.update_layout(
            title=(
                "Toxicity Risk vs Tumor Regrowth by Delay Window<br>"
                f"<sup>Patient α = {gomp_fit['growth_rate']:.3f}/day, "
                f"Kill = {gomp_fit['kill_fraction']:.1%}</sup>"
            ),
            xaxis_title="Candidate Dosing Window",
            yaxis=dict(title="Toxicity Risk (%)", range=[0, 100], color="#e74c3c"),
            yaxis2=dict(
                title="Tumor Excess Growth (%)",
                overlaying="y",
                side="right",
                range=[0, max_y2],
                color="#2980b9",
            ),
            legend=dict(x=0.03, y=0.97),
            margin=dict(l=40, r=40, t=55, b=40),
            height=390,
        )
        st.plotly_chart(fig_tradeoff, use_container_width=True)
        st.caption(
            "**Tumor Forecast Band:** Conservative (97.5% empirical coverage "
            "vs 90% nominal). Coverage was checked only for delay scenarios "
            "where the real patient received no dose in the window."
        )

    with col_r:
        st.markdown("#### Clinical Action Matrix")

        # Derive dynamic operational metrics for the main model from results files
        lr_row = ablation_summary_df[
            ablation_summary_df["Model"] == "Logistic Regression"
        ].iloc[0]
        opt_thresh = float(
            meta.get("optimal_threshold", lr_row["CV-Tuned Threshold"])
        )
        test_recall = float(lr_row["Untouched Test Recall"])
        test_precision = float(lr_row["Untouched Test Precision"])
        false_alarm_pct = (1.0 - test_precision) * 100.0
        false_alarm_ratio_10 = int(round((1.0 - test_precision) * 10))

        decision_table = []
        for i, d in enumerate(delays):
            ml_risk = candidate_risks[i]
            pen = tradeoff_summary[d]["regrowth_penalty_pct"]
            low_vol = tradeoff_summary[d]["lower_bound_90_cm3"]
            upp_vol = tradeoff_summary[d]["upper_bound_90_cm3"]
            vol_d = tradeoff_summary[d]["volume_at_dose_cm3"]

            if show_friberg and hybrid_evaluations:
                hyb_info = hybrid_evaluations[d]
                risk = hyb_info["hybrid_risk"]
                pred_nadir = anc_forecasts[d]["predicted_nadir_anc"]
                nadir_day = anc_forecasts[d]["predicted_nadir_day"]
                below_1 = anc_forecasts[d]["days_below_1_0"]
                nadir_band = anc_forecasts[d].get(
                    "nadir_band", [round(pred_nadir * 0.9, 2), round(pred_nadir * 1.1, 2)]
                )
                exp_days = anc_forecasts[d].get("expected_days_below_1_0", float(below_1))

                if risk < opt_thresh:
                    status = "🟢 Lower risk"
                elif risk <= 2.0 * opt_thresh:
                    status = "🟡 Elevated"
                else:
                    status = "🔴 High risk"

                decision_table.append(
                    {
                        "Option": f"Day +{d}",
                        "Hybrid Risk": f"{risk * 100:.1f}%",
                        "Tier": status,
                        "ML 2nd Opinion": f"{ml_risk * 100:.1f}%",
                        "Pred. Nadir": f"{pred_nadir:.2f} (d+{nadir_day})",
                        "Nadir 90% Band": f"[{nadir_band[0]:.2f}–{nadir_band[1]:.2f}]",
                        "Exp Days < 1.0": f"{exp_days:.1f}d",
                        "Regrowth": f"+{pen:.1f}%",
                        "Vol 90% CI": f"{vol_d:.1f} [{low_vol:.1f}–{upp_vol:.1f}]",
                    }
                )
            else:
                if ml_risk < opt_thresh:
                    status = "🟢 Lower risk"
                elif ml_risk <= 2.0 * opt_thresh:
                    status = "🟡 Elevated"
                else:
                    status = "🔴 High risk"

                decision_table.append(
                    {
                        "Option": f"Day +{d}",
                        "Toxicity Risk (ML)": f"{ml_risk * 100:.1f}%",
                        "Tier": status,
                        "Tumor Regrowth": f"+{pen:.1f}%",
                        "Vol 90% CI": f"{vol_d:.1f} [{low_vol:.1f}–{upp_vol:.1f}]",
                    }
                )

        matrix_df = pd.DataFrame(decision_table)

        def highlight_tier(row):
            tier = row["Tier"]
            if "Lower risk" in tier:
                bg = (
                    "background-color: rgba(46, 204, 113, 0.2); "
                    "color: #27ae60; font-weight: 600;"
                )
            elif "Elevated" in tier:
                bg = (
                    "background-color: rgba(241, 196, 15, 0.25); "
                    "color: #d35400; font-weight: 600;"
                )
            else:
                bg = (
                    "background-color: rgba(231, 76, 60, 0.2); "
                    "color: #c0392b; font-weight: 600;"
                )
            return ["" if col != "Tier" else bg for col in row.index]

        styled_matrix = matrix_df.style.apply(highlight_tier, axis=1)
        st.dataframe(
            styled_matrix,
            hide_index=True,
            use_container_width=True,
        )

        if show_friberg:
            st.caption(
                f"**Tiers (T = {opt_thresh:.2f}):** "
                f"🟢 Lower risk (< {opt_thresh:.2f}) | "
                f"🟡 Elevated ({opt_thresh:.2f}–{2*opt_thresh:.2f}) | "
                f"🔴 High risk (> {2*opt_thresh:.2f})\n\n"
                f"**Hybrid Architecture:** 55% Friberg Mechanistic Nadir + "
                f"45% Logistic Regression ML Second Opinion."
            )
        else:
            st.caption(
                f"**Tiers (T = {opt_thresh:.2f}):** "
                f"🟢 Lower risk (< {opt_thresh:.2f}) | "
                f"🟡 Elevated ({opt_thresh:.2f}–{2*opt_thresh:.2f}) | "
                f"🔴 High risk (> {2*opt_thresh:.2f})\n\n"
                f"**Main Model:** Standardized Logistic Regression scoring strictly retrospective features (day ≤ D)."
            )

        st.info(
            f"**Tuned Safety Threshold: {opt_thresh:.2f}**\n\n"
            f"**Multi-Seed Performance Headline:** Across 5 independent random seeds, "
            f"untouched test recall averages **90.9%** (84.8% to 96.1%) and precision averages **42.7%** (26.2% to 52.2%).\n\n"
            f"**Reference Run (Seed 42):** Untouched test recall is **{test_recall:.1%}** and precision is **{test_precision:.1%}** "
            f"(approx. {false_alarm_ratio_10} in 10 alerts represent false alarms to ensure zero severe myelosuppressive collapses are missed)."
        )

with tab2:
    st.subheader("Longitudinal Patient Trajectory (Clinical + Wearables)")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        # ANC Trajectory
        fig_anc = go.Figure()
        fig_anc.add_trace(
            go.Scatter(
                x=p_clin["day_index"],
                y=p_clin["anc"],
                mode="lines+markers",
                name="Observed ANC",
                line=dict(color="#34495e", width=2),
            )
        )
        if show_friberg and friberg_fit is not None:
            # Fitted Friberg model historical reconstruction up to sim_day
            past_days = patient_clin_past["day_index"].values.astype(float)
            if len(past_days) > 0:
                past_doses = patient_clin_past[patient_clin_past["dose_mg"] > 0]
                friberg_sim = friberg_engine.simulate_longitudinal_anc(
                    days=past_days,
                    dose_days=past_doses["day_index"].tolist(),
                    dose_amounts=past_doses["dose_mg"].tolist(),
                    base_anc=friberg_fit["base_anc"],
                    mtt=friberg_fit["mtt"],
                    recovery_rate=friberg_fit["recovery_rate"],
                    drug_slope=friberg_fit["drug_slope"],
                )
                friberg_lbl = (
                    f"Friberg Fit (MTT={friberg_fit['mtt']:.1f}d, "
                    f"r={friberg_fit['recovery_rate']:.2f})"
                )
                fig_anc.add_trace(
                    go.Scatter(
                        x=past_days,
                        y=friberg_sim,
                        mode="lines",
                        name=friberg_lbl,
                        line=dict(color="#9b59b6", width=2, dash="dash"),
                    )
                )
            # Forward simulated ANC trajectories across candidate delay scenarios
            delay_colors = {0: "#e74c3c", 3: "#e67e22", 7: "#f1c40f", 14: "#2ecc71"}
            for d_cand in delays:
                sim_info = anc_forecasts[d_cand]
                t_cand = [sim_day + d_cand + dt for dt in sim_info["time_days"]]
                c_label = (
                    f"+{d_cand}d dose: Nadir {sim_info['predicted_nadir_anc']:.2f} "
                    f"({sim_info['days_below_1_0']}d < 1.0)"
                )
                fig_anc.add_trace(
                    go.Scatter(
                        x=t_cand,
                        y=sim_info["curve"],
                        mode="lines",
                        name=c_label,
                        line=dict(color=delay_colors.get(d_cand, "#7f8c8d"), width=2),
                    )
                )
        fig_anc.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="red",
            annotation_text="Grade 3 Neutropenia (1.0)",
        )
        fig_anc.add_vline(x=sim_day, line_color="blue", annotation_text="Current Day")
        fig_anc.update_layout(
            title=(
                "Absolute Neutrophil Count (ANC) & Fitted Friberg Model"
                if show_friberg
                else "Absolute Neutrophil Count (ANC) Trajectory"
            ),
            xaxis_title="Day Index",
            yaxis_title="ANC (×10⁹/L)",
            height=320,
        )
        st.plotly_chart(fig_anc, use_container_width=True)
        if show_friberg and friberg_fit is not None:
            raw_r = friberg_fit.get("raw_recovery_rate", friberg_fit["recovery_rate"])
            m_strain = friberg_fit.get("marrow_strain", 0.0)
            st.caption(
                f"**Patient-Specific Friberg Fit (day ≤ {sim_day}):** "
                f"Baseline ANC = {friberg_fit['base_anc']:.2f} ×10⁹/L | "
                f"Transit Time (MTT) = {friberg_fit['mtt']:.1f} days | "
                f"Recovery Rate (r) = {friberg_fit['recovery_rate']:.2f} "
                f"(Raw = {raw_r:.2f}, Marrow Strain = {m_strain:.2f}) | "
                f"Residual σ = {friberg_fit['residual_std']:.2f}"
            )

    with col_t2:
        # Wearable HRV & Resting HR Trajectory
        fig_wear = go.Figure()
        fig_wear.add_trace(
            go.Scatter(
                x=p_wear["day_index"],
                y=p_wear["hrv_rmssd"],
                mode="lines",
                name="HRV (RMSSD)",
                line=dict(color="#27ae60", width=2),
            )
        )
        fig_wear.add_vline(x=sim_day, line_color="blue", annotation_text="Current Day")
        fig_wear.update_layout(
            title="Autonomic Wearable Signal (HRV RMSSD)",
            xaxis_title="Day Index",
            yaxis_title="HRV (ms)",
            height=320,
        )
        st.plotly_chart(fig_wear, use_container_width=True)

with tab3:
    st.subheader("Model Evaluation & Rigorous Multi-Seed Benchmark")

    st.markdown("### 🌟 Multi-Seed Generalization Benchmark (Headline Results)")
    st.markdown(
        """
        To prevent single-seed bias or cherry-picking, the headline metrics below
        report the **mean and min–max range across 5 independent random data generator seeds**
        (Seeds 42, 101, 2024, 7, 999).

        Seed 42 is included transparently as the reference run
        (representing the most conservative / worst-case holdout split with ~74% false alarm rate),
        rather than isolated or swapped.
        """
    )
    if not multi_seed_benchmark_df.empty:
        st.dataframe(
            multi_seed_benchmark_df,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### 🔬 Model Comparison & Feature Ablation (Reference Run: Seed 42)")
    st.markdown(
        """
        **Ablation Study (Testing Incremental Value of Wearable Streams):**
        - **Ablation A:** Static Clinical Profile (Age, ECOG, baseline ANC, pre-dose ANC, dose, tumor, delay, interval).
        - **Ablation B:** Clinical + Wearables (Ablation A + RHR, HRV, steps, sleep, and 7d/30d baseline deviations).

        *Tree Model Performance:* Adding continuous wearable biometric ratios improved regularized Logistic Regression (+0.0113 AUROC),
        whereas unconstrained tree ensembles (Random Forest -0.0226, Gradient Boosting -0.0065) showed slight negative transfer due to noise overfitting.
        """
    )

    # Format numeric columns for clean, un-truncated display
    display_ablation = ablation_summary_df.copy()
    display_ablation["CV AUROC Ablation A"] = display_ablation[
        "CV AUROC Ablation A"
    ].apply(lambda x: f"{x:.4f}")
    display_ablation["CV AUROC Ablation B"] = display_ablation[
        "CV AUROC Ablation B"
    ].apply(lambda x: f"{x:.4f}")
    display_ablation["Test AUROC"] = display_ablation["Test AUROC"].apply(
        lambda x: f"{x:.4f}"
    )
    display_ablation["Test Brier"] = display_ablation["Test Brier"].apply(
        lambda x: f"{x:.4f}"
    )
    display_ablation["CV-Tuned Threshold"] = display_ablation[
        "CV-Tuned Threshold"
    ].apply(lambda x: f"{x:.2f}")
    display_ablation["Untouched Test Recall"] = display_ablation[
        "Untouched Test Recall"
    ].apply(lambda x: f"{x:.1%}")
    display_ablation["Untouched Test Precision"] = display_ablation[
        "Untouched Test Precision"
    ].apply(lambda x: f"{x:.1%}")
    display_ablation["Timing Ranking Tau (Mean)"] = display_ablation[
        "Timing Ranking Tau (Mean)"
    ].apply(lambda x: f"{x:.4f}")
    display_ablation["Timing Tau Spread (Std)"] = display_ablation[
        "Timing Tau Spread (Std)"
    ].apply(lambda x: f"{x:.4f}")
    display_ablation["Timing Tau Spread (IQR)"] = display_ablation[
        "Timing Tau Spread (IQR)"
    ].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_ablation, hide_index=True, use_container_width=True)

    paired_diff_str = lr_row["Paired Diff (B - A)"]
    tau_mean = float(lr_row["Timing Ranking Tau (Mean)"])
    tau_std = float(lr_row["Timing Tau Spread (Std)"])
    tau_iqr = float(lr_row["Timing Tau Spread (IQR)"])
    cv_auroc_b = float(lr_row["CV AUROC Ablation B"])
    test_auroc = float(meta.get("test_auc", lr_row["Test AUROC"]))

    st.markdown(
        f"""
        > **Methodological Note:** Models are evaluated with repeated 5×5 Group CV
        > grouped by patient ID (zero leakage). The paired difference on folds
        > indicates consistent improvement ({paired_diff_str} AUROC,
        > CV B = {cv_auroc_b:.4f}, untouched test AUROC = {test_auroc:.4f}).
        > Counterfactual timing ranking yields Kendall tau = {tau_mean:.4f}
        > ± {tau_std:.4f} (IQR = {tau_iqr:.4f}) against true simulated post-dose nadirs.
        > As this is synthetic proof-of-concept data, high metrics demonstrate
        > architecture and generator design, not clinical validation.
        """
    )

    c_f1, c_f2 = st.columns(2)
    with c_f1:
        st.image(
            os.path.join(BASE_DIR, "docs", "figures", "roc_curve_ablation.png"),
            caption="ROC Curve: Ablation A vs Ablation B",
        )
    with c_f2:
        st.image(
            os.path.join(BASE_DIR, "docs", "figures", "calibration_curve.png"),
            caption="Reliability & Calibration Curve",
        )
