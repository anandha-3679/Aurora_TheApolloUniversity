# ONCO-TWIN: A Digital Twin for Predicting Severe Chemotherapy Toxicity and Choosing the Safest Dosing Window

> **For clinical decision support and simulation only. Not an autonomous prescription engine.**
> All data in this project is synthetic. This is a proof-of-concept and has not been clinically validated.

**Happiest Health Digital Twin Challenge 2026**

---

## 0. Team & Submission Details

| Item | Details |
|------|---------|
| Team name | Aurora |
| Team leader | Anandha Lakshmi RM (solo entry) |
| College / incubator | The Apollo University |
| Email / phone | anandhalakshmi005@gmail.com / 7799301372 |
| Project title | ONCO-TWIN |
| Repository folder name | Aurora_TheApolloUniversity |
| License | MIT (see `LICENSE`) |

**Submission checklist**
- [x] Public GitHub repository, all files and links accessible
- [x] This README with all required details
- [x] Architecture diagram (PDF/PPT) in `docs/`
- [x] Project presentation (PDF/PPT) in `docs/`
- [ ] Demo video, **minimum 20 minutes** (PowerPoint recording or unlisted YouTube link): [Insert 20-min YouTube / Drive Link Here]
- [x] Open-source license file (`LICENSE`)

---

## 1. Healthcare Use Case & Problem Statement

### Why cancer fits "chronic condition prevalent in India"
Cancer is classified as a chronic disease by major health bodies, including the WHO (as a leading noncommunicable disease) and the US National Cancer Institute, because it is long-lasting and needs ongoing monitoring and management. In India it is a major and growing burden.
According to the Indian Council of Medical Research - National Cancer Registry Programme (ICMR-NCRP Report 2020–2025; Mathur et al., *JCO Global Oncology* 2020), **breast cancer is the single most common cancer in Indian women**, accounting for **28.2% of all female cancer diagnoses** with over **200,000 new cases diagnosed annually**. The projected lifetime cumulative risk is 1 in 28 for Indian females, with over 50% presenting at locally advanced stages requiring intensive cytotoxic chemotherapy regimens that frequently induce life-threatening myelosuppression.

**Chosen cancer type:** **Breast Cancer (Invasive Ductal Carcinoma / Stage II–IV)** under myelosuppressive chemotherapy regimens (e.g., Doxorubicin + Cyclophosphamide / Paclitaxel).

### The problem
Cancer treatment is retrospective. Chemotherapy toxicity (especially severe neutropenia) is usually caught after it happens, through lab checks at scheduled visits, while the patient's real-time condition goes unseen between visits. Dosing decisions rely on population averages, not on how resilient this patient is today.

### What ONCO-TWIN does
ONCO-TWIN maintains a continuously updated virtual patient by fusing:
- **Static/historical data:** simulated EHR (demographics, cancer type and stage, biomarkers, ECOG, baseline labs, treatment history)
- **Dynamic data:** simulated wearable time series (HRV, resting heart rate, step count, sleep)

and uses that state to predict an adverse event before it happens.

### Headline prediction (adverse event)
> Probability of **severe toxicity (grade >= 3 neutropenia, ANC < 1.0 x10^9/L) within 7 days after a dose**, scored for each candidate dosing day.

### Supporting panel: cost of delay
A small Gompertz tumor module shows how much the tumor may regrow if the dose is delayed by 0, 3, 7 or 14 days, so a clinician sees the toxicity-versus-tumor-control trade-off. Without it, "safest day" would always be "later".

---

## 2. Concept

```
 Patient data (incl. tumor)  ──┐
 Treatment data              ──┼──> Digital Twin State ──> Main model: toxicity risk per dosing day
 Wearable data               ──┘                      └─> Supporting module: tumor regrowth cost of delay
```

| Component | Question | Inputs | Output |
|-----------|----------|--------|--------|
| Toxicity model (main) | When is severe toxicity least likely? | Patient + treatment + wearable data | Risk per candidate dose day, with uncertainty and drivers |
| Cost-of-delay module (supporting) | What does waiting cost in tumor control? | Patient + treatment data | Tumor regrowth curve per delay (0/3/7/14 days) |

---

## 3. Data (Synthetic)

Teams may use synthetic or open-source data under the challenge rules. We use a Python generator for the clinical and wearable streams. `<OPTIONAL: Synthea for baseline EHR attributes and/or an open wearable dataset to calibrate realism>`

### Tier 1: `patient_profiles.csv` (one row per patient)
`patient_id`, `age`, `sex`, `cancer_type`, `stage`, `biomarker_status`, `baseline_ecog`, `baseline_anc`

### Tier 2: `clinical_logs.csv` (longitudinal)
`patient_id`, `day_index`, `treatment_administered` (Regimen A / Regimen B / None), `dose_mg`, `anc`, `wbc`, `tumor_volume_cm3`

### Tier 3: `wearable_streams.csv` (daily)
`patient_id`, `date`, `resting_heart_rate`, `hrv_rmssd`, `step_count`, `sleep_duration_mins`

### Generator design (avoids circular validation)
The toxicity label is **not** computed from the wearable features.
1. A **latent ANC curve** is simulated per patient. It drops after each dose and reaches a nadir roughly 7-14 days later, with depth driven by regimen, dose, age and ECOG.
2. **Label** = latent ANC falls below 1.0 x10^9/L in the 7-day window after dosing.
3. **Wearables are noisy, lagged proxies** of the same hidden state (HRV and steps dip, resting heart rate rises).
4. **Confounders** are added (poor sleep or activity changes unrelated to treatment) plus per-patient baseline variability.
5. **Tumor volume** follows a Gompertz growth model with a regimen-specific, biomarker-dependent kill term.

Parameters are anchored to published literature (`docs/references.md`).

---

## 4. Models & Evaluation

### Empirically Selected Main Model
- **Main Model:** **Standardized Logistic Regression** (empirically selected over Random Forest and Gradient Boosting due to superior generalization, lower Brier calibration error, and highest CV AUROC).
- **Benchmark Models:** Random Forest and Gradient Boosting.
- **Features:** Patient profile + treatment attributes + 7-day wearable trends versus personal 30-day rolling baseline.

### Rigorous Evaluation & Ablation Results

Evaluation strictly groups by patient ID (`GroupShuffleSplit` & repeated 5×5 `GroupKFold`), preventing longitudinal row-level leakage:

| Model | CV AUROC Ablation A (Clinical only) | CV AUROC Ablation B (+ Wearables) | Paired Diff $(B - A)$ Fold Mean $\pm$ 95% CI | Test AUROC | Test Brier Score | Untouched Test Recall (at Thresh=0.11) | Untouched Test Precision | Timing Ranking Tau Mean $\pm$ Std (IQR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Main)** | **0.8684** | **0.8960** | **+0.0276 ± 0.0092** | **0.9165** | **0.0660** | **90.91%** | **35.09%** | **0.6782 ± 0.3911 (IQR 0.3333)** |
| Random Forest | 0.8304 | 0.8541 | +0.0237 ± 0.0094 | 0.8022 | 0.0951 | 72.73% | 22.38% | 0.4896 ± 0.4592 (IQR 0.4252) |
| Gradient Boosting | 0.8279 | 0.8700 | +0.0421 ± 0.0082 | 0.8663 | 0.0889 | 84.09% | 33.04% | 0.6216 ± 0.4212 (IQR 0.6667) |

> **Methodological Honesty Note on Metrics:**
> - Rather than claiming "statistical significance" on reused patient folds, we report the paired difference fold interval $(+0.0276 \pm 0.0092)$, demonstrating consistent improvement across splits.
> - Holdout test performance (Test AUROC 0.9165) reflects our simulated cohort; we lean on the cross-validation paired differences (+0.0276 AUROC) rather than raw holdout numbers alone to evaluate incremental wearable utility.

### Clinical Safety Threshold & Operational Limitation
- Rather than an arbitrary 0.50 cutoff, the decision threshold was tuned strictly inside validation folds to **0.11** to prioritize patient safety.
- **Operational Limitation:** At the tuned threshold (0.11), the model catches about 91% of severe events with about 35% precision on a small synthetic test set, so it is a risk flag to support clinician judgment, not a safety guarantee. Because the test set is small, the recall estimate has wide uncertainty. Approximately 6 to 7 out of 10 alerts represent false alarms.

### Timing Ranking Evaluation: Definition & Empirical Spread
- **Kendall Tau Definition:** Computed strictly per patient and treatment cycle between the model's **predicted toxicity risk** and the **TRUE simulated risk** across candidate delays (+0, +3, +7, +14 days). The TRUE simulated risk is derived from the latent ANC trajectory (where lower nadir ANC corresponds to greater acute toxicity risk: $\text{true\_risk} = -\text{nadir\_anc}$). The concordance coefficients are then averaged across all test cycles.
- **Empirical Distribution (Mean & Spread):**
  - **Logistic Regression (Main):** Mean Tau = **0.6782**, Std = **0.3911**, Median = **0.8165**, IQR = **0.3333**.
  - **Random Forest:** Mean Tau = **0.4896**, Std = **0.4592**, IQR = **0.4252**.
  - **Gradient Boosting:** Mean Tau = **0.6216**, Std = **0.4212**, IQR = **0.6667**.
- **Honest Framing on Timing & Ordering:** We do **not** claim perfect monotonic ordering (+1.0000). Biological noise, nonlinear marrow recovery rates ($r_i$), and stochastic stress spikes produce realistic rank inversions across candidate delays, accurately captured by the spread metrics.
- **Prospective Daily Re-scoring:** All candidate delays are evaluated strictly with retrospective features frozen at decision day $D$. In clinical practice, the twin operates by **daily morning re-scoring**, updating risk as real-world biometrics reveal whether marrow recovery is actually underway.

### Explainability & Upgraded Cost-of-Delay Module
- Standardized log-odds coefficients and feature importance weights reveal top risk drivers (e.g., severe drops in 7-day HRV and step count relative to personal baseline).
- **Patient-Specific Gompertz Estimation:** At decision day $D$, the module fits patient-specific biological kinetics (growth rate $\alpha$ and chemotherapy kill effect $\kappa$) using bounded nonlinear least squares strictly on retrospective observations ($\text{day\_index} \le D$, zero future lookahead).
- **Forecast Uncertainty (90% Band):** For candidate delays (0, 3, 7, 14 days), the engine generates prospective tumor volume trajectories with a calibrated 90% confidence ribbon ($\pm 1.645 \cdot \sigma \sqrt{\text{delay} + 1}$) accounting for imaging measurement noise.
- **Held-Out Forecast Accuracy (MAPE Benchmark):**
  - Evaluated on held-out post-decision measurements ($\text{day\_index} > D$) across test patients ($N = 37$):
  - **Patient-Specific Gompertz Fit:** **$4.77\% \pm 3.33\%$ MAPE**
  - **Population-Average Gompertz Curve:** **$8.19\% \pm 6.63\%$ MAPE**
  - Patient-tailored fitting reduces future trajectory forecast error by **41.8%** over static population averages while strictly preserving zero lookahead invariance.

---

## 5. Doctor Dashboard (Streamlit)

- Patient selector and current twin state (latest labs, wearable trends vs 30-day baseline)
- Dual-axis interactive trade-off panel: severe toxicity risk curve vs Gompertz tumor regrowth penalty across candidate dosing windows (+0, +3, +7, +14 days)
- Longitudinal curves for Absolute Neutrophil Count (ANC) and autonomic HRV trajectories
- Action matrix with 3 risk tiers based on tuned threshold T (0.11), 90% forecast CI, and risk flag characteristics
- Model performance tab embedding ROC, calibration, PR operating point, and coefficient plots
- Permanent banner: *"For Clinical Decision Support and Simulation Only. Not an Autonomous Prescription Engine."*

---

## 6. Demo Walkthrough

1. Load one synthetic patient on Regimen A.
2. Wearables show an HRV dip and resting heart rate rise; predicted risk climbs.
3. Twin scores dosing today versus +3, +7 and +14 days.
4. Dashboard shows risk falling with delay, while tumor regrowth cost rises.
5. Driver panel explains why. The clinician makes the call.

---

## 7. Build Order (solo plan)

| Step | Task | Notes |
|------|------|-------|
| 1 | Synthetic data generator | Foundation. **Hard cap about 1 day** for a first working version. Check distributions and label rate. |
| 2 | Cost-of-delay Gompertz module | Small, rule-based, single regimen. |
| 3 | Toxicity model, patient-level split, ablation | Baseline first, then stronger model. |
| 4 | Evaluation: metrics, calibration, uncertainty | Save plots for the presentation. |
| 5 | Streamlit dashboard | Call Python directly; no separate API. |
| 6 | SHAP explanations | Mostly library calls once the model works. |
| 7 | Architecture diagram and presentation | PDF/PPT, required for submission. |
| 8 | Demo video (20+ minutes) | Script it from Section 6, add model results and limitations. |
| 9 | Final repo check | README, license, folder name, public access. |

**Cut first if time runs short:** FastAPI, a second regimen, any ML in the cost-of-delay module.

---

## 8. Repository Structure

```
Aurora_TheApolloUniversity/
├── README.md
├── LICENSE
├── data/                 # generated CSVs
├── src/
│   ├── generate_data.py
│   ├── tumor_model.py    # cost-of-delay (Gompertz)
│   ├── features.py
│   ├── train_toxicity.py # toxicity model
│   └── evaluate.py
├── app/
│   └── dashboard.py      # Streamlit
├── docs/
│   ├── architecture.pdf
│   ├── presentation.pdf
│   ├── references.md
│   ├── limitations.md
│   └── figures/
└── requirements.txt
```

---

## 9. Technical Stack

Python, Pandas, NumPy, Scikit-Learn, SHAP, Plotly, Streamlit.

---

## 10. Limitations & Safety

> [!WARNING]
> **Essential Methodological Limitations & Non-Clinical Status:**
> - **Synthetic Proxy Assumption:** The observed paired wearable gain ($+0.0510 \pm 0.0106$ CV AUROC) directly depends on the synthetic generator's assumption that autonomic wearable trends (RHR, HRV, steps) serve as noisy proxies for an unobserved biological recovery rate ($r_i$) and latent marrow vulnerability. If real-world wearable signals are decoupled from bone marrow nadir dynamics, this incremental utility may not generalize.
> - **Risk Flag, Not Safety Guarantee:** At the tuned threshold (0.11), the model catches about 91% of severe events with about 35% precision on a small synthetic test set, meaning it serves as an advisory risk flag to support clinician judgment rather than a definitive safety guarantee. Because the test set is small, the recall estimate has wide uncertainty.
> - **Synthetic Data Only:** All findings and performance figures reflect simulated data, not clinically validated patient outcomes.
> - **Not Monotonically Perfect:** Dosing delay ranking exhibits meaningful variance (Tau spread $\sigma = 0.2218$, IQR = $0.3333$) due to individual recovery kinetics and noise; perfect ordering should never be claimed.
> - **Decision Support Only:** The system generates informational risk trajectories for simulation and pair-programming research. It is **not** an autonomous prescribing device.
> - **Regulatory Prerequisites:** Real-world translational application mandates clinical trial validation, formal prospective auditing, and full regulatory clearance (DPDP Act, HIPAA, FDA SaMD).

---

## 11. License

MIT License. See `LICENSE`.
