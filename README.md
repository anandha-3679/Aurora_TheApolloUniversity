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
| Email | anandhalakshmi005@gmail.com |
| Project title | ONCO-TWIN |
| Repository folder name | Aurora_TheApolloUniversity |
| License | MIT (see [LICENSE](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/LICENSE)) |

**Submission checklist**
- [x] Public GitHub repository, all files and links accessible
- [x] This README with all required details and exact current state
- [x] Architecture diagram (PDF/PPT) in [`docs/architecture.pdf`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/docs/)
- [x] Project presentation (PDF/PPT) in [`docs/presentation.pdf`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/docs/)
- [ ] Demo video, **minimum 20 minutes** (PowerPoint recording or unlisted YouTube link): [Insert 20-min YouTube / Drive Link Here]
- [x] Open-source license file ([`LICENSE`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/LICENSE))
- [x] Comprehensive test suite passing with zero-leakage assertions (12 passing tests in [`tests/`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/tests/))
- [x] Strict linting passing clean ([`ruff check .`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/ruff.toml))

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
A patient-specific Gompertz tumor module shows how much the tumor may regrow if the dose is delayed by 0, 3, 7 or 14 days, with a conservative 90% forecast uncertainty band, so a clinician sees the toxicity-versus-tumor-control trade-off. Without it, "safest day" would always be "later".

---

## 2. Concept & Information Flow

```
 Patient data (incl. tumor)  ──┐
 Treatment data              ──┼──> Digital Twin State (<= D) ──> Main model: toxicity risk per candidate delay
 Wearable data               ──┘                                └─> Supporting module: Gompertz regrowth cost of delay
                                   [Zero Future Lookahead Guarantee]
```

| Component | Clinical Question | Inputs (strictly $\le D$) | Output |
|-----------|-------------------|---------------------------|--------|
| **Toxicity model (Main)** | When is severe toxicity least likely? | Patient profile + treatment state + wearable deviations | Risk per candidate dose day (+0, +3, +7, +14d), with safety tiers and feature drivers |
| **Cost-of-delay module (Supporting)** | What does waiting cost in tumor control? | Patient-specific Gompertz fit ($\alpha, \kappa$) on retrospective scans | Prospective tumor volume curve per delay (0/3/7/14 days) with 90% uncertainty band |

---

## 3. Data Architecture (Synthetic & Leakage-Free)

Teams may use synthetic or open-source data under challenge rules. We engineered a multi-tier Python generator anchored to clinical oncology kinetics ([`docs/references.md`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/docs/references.md)).

### Data Files Overview (`data/`)

| File | Rows | Schema / Columns | Clinical Role |
| :--- | :--- | :--- | :--- |
| **`patient_profiles.csv`** | 150 | `patient_id`, `age`, `sex`, `cancer_type`, `stage`, `biomarker_status`, `baseline_ecog`, `baseline_anc` | Tier 1: Static demographics, cancer staging, baseline biology |
| **`clinical_logs.csv`** | 13,500 | `patient_id`, `day_index`, `treatment_administered`, `dose_mg`, `anc`, `wbc`, `tumor_volume_cm3` | Tier 2: Longitudinal clinical labs (90 days/pt), dosing events (600 total doses), tumor measurements |
| **`wearable_streams.csv`** | 13,500 | `patient_id`, `date`, `day_index`, `resting_heart_rate`, `hrv_rmssd`, `step_count`, `sleep_duration_mins` | Tier 3: Daily biometric time series (90 days/pt) |
| **`timing_eval.csv`** | 1,200 | `patient_id`, `cycle`, `planned_day`, `delay_days`, `candidate_day`, `severe_toxicity_label`, `nadir_anc` | Intra-patient counterfactual timing evaluation choices (+0, +3, +7, +14d across cycles 1 & 2) |
| **`processed_dataset.csv`** | 1,500 | 26 columns (engineered features + label) | Dosing decisions at day $D$ for training and cross-validation |
| **`timing_features.csv`** | 1,200 | 27 columns (engineered features + cycle + nadir ANC) | Prospective candidate delay feature matrix evaluated at day $D$ |

### Generator Design (Avoids Circular Validation)
The toxicity label is **not** computed from wearable features:
1. **Latent ANC Nadir Curve:** A hidden hematopoietic reserve is simulated per patient using Friberg-style myelosuppression kinetics. It reaches nadir roughly 7–10 days post-dose, with rebound rate $r_i$ varying across patients ($0.65 \le r_i \le 1.45$).
2. **Ground Truth Label:** `severe_toxicity_label` = 1 if latent ANC drops below $1.0 \times 10^9/\text{L}$ (Grade $\ge 3$ neutropenia) in the post-dose window, 0 otherwise.
3. **Noisy Lagged Proxies:** Autonomic wearable streams are generated as noisy, lagged manifestations of marrow strain (RHR elevates $+12 \text{ bpm} \cdot \text{stress}$, HRV dips $-15 \text{ ms} \cdot \text{stress}$, step count drops $-2200 \text{ steps} \cdot \text{stress}$).
4. **Independent Confounders:** Sleep duration includes uncoupled non-treatment variance ($\pm 35\text{ mins}$ normal noise).
5. **Gompertzian Tumor Dynamics:** Latent tumor volume evolves via differential equation $\frac{dV}{dt} = \alpha V \ln(K/V) - \kappa \cdot \text{Dose} \cdot V$, observed with imaging measurement error ($\sigma \approx 0.45 \text{ cm}^3$).

---

## 4. Models, Evaluation & Leakage-Free Design

### Zero Future Lookahead Guarantee (Verified by Automated Testing)
A core architectural requirement of ONCO-TWIN is **strict prospective invariance**:
- When evaluating candidate dosing days (+0, +3, +7, +14 days) at decision day $D$, **wearable signals and clinical labs are strictly frozen at $D$ ($\le D$)**.
- The candidate delay is parameterized purely through known scheduling variables: `delay_days`, `days_since_last_dose`, and interval non-linearity bins (`dose_interval_bin_8_14`).
- **Invariance Test:** In [`tests/test_features_and_models.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/tests/test_features_and_models.py), test suites perturb all post-decision wearable and lab data ($> D$) with extreme values ($\text{ANC} = 0.05, \text{RHR} = 180, \text{HRV} = 5$). The model outputs for all candidate delays remain **mathematically identical ($\text{atol} = 10^{-7}$)**.
- **Label Window:** The label evaluation window for a proposed dose at $D + \Delta$ strictly measures outcomes post-infusion: $(D + \Delta, D + \Delta + 7]$.

### Feature Sets (Ablation A vs Ablation B)
Features are defined in [`src/features.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/features.py):
- **Ablation A (Clinical Only - 8 features):** `age`, `baseline_ecog`, `baseline_anc`, `pre_dose_anc`, `tumor_volume_cm3`, `delay_days`, `days_since_last_dose`, `dose_interval_bin_8_14`.
- **Ablation B (Clinical + Wearables - 16 features):** Ablation A + `resting_heart_rate`, `hrv_rmssd`, `step_count`, `sleep_duration_mins`, `rhr_ratio_7_to_30`, `hrv_ratio_7_to_30`, `steps_ratio_7_to_30`, `sleep_ratio_7_to_30`.

### Empirically Selected Main Model
- **Main Model:** **Standardized Logistic Regression** (`C=1.0, max_iter=1000`, `StandardScaler`). Empirically chosen over Random Forest and Gradient Boosting due to superior out-of-fold generalization, lowest Brier calibration error, and highest holdout AUROC.
- **Benchmark Models:** Random Forest (`n_estimators=100, max_depth=5`) and Gradient Boosting (`n_estimators=100, max_depth=3`).

### Rigorous Multi-Seed Evaluation & Ablation Results

Evaluation strictly groups by patient ID (`GroupShuffleSplit` & repeated 5×5 `GroupKFold`), preventing longitudinal row-level leakage.

To ensure empirical stability beyond a single dataset realization, models were evaluated across **5 distinct data generator seeds** (Seeds 42, 101, 2024, 7, 999). We report the **mean and min–max range** across these runs alongside reference point estimates:

| Metric | Multi-Seed Mean | Multi-Seed Range (Min – Max) | Reference Run (Seed 42) |
| :--- | :--- | :--- | :--- |
| **Paired Wearable Gain $(B - A)$** | **+0.0354 AUROC** | **+0.0266 to +0.0478** | +0.0276 ± 0.0092 |
| **CV AUROC Ablation B (+ Wearables)** | **0.8978** | **0.8702 to 0.9184** | 0.8960 |
| **CV AUROC Ablation A (Clinical only)**| **0.8623** | **0.8345 to 0.8838** | 0.8684 |
| **Holdout Test AUROC** | **0.9084** | **0.8718 to 0.9258** | 0.9165 |
| **Holdout Test Brier Score** | **0.0682** | **0.0610 to 0.0754** | 0.0660 |
| **CV-Tuned Threshold** | **0.114** | **0.100 to 0.140** | 0.110 |
| **Untouched Test Recall** | **88.88%** | **82.35% to 94.55%** | 90.91% |
| **Untouched Test Precision** | **37.44%** | **34.21% to 45.03%** | 35.09% |
| **Timing Ranking Kendall Tau** | **0.7251** | **0.6384 to 0.8281** | 0.6782 |

> **Methodological Honesty Note on Multi-Seed Metrics:**
> - Across all 5 seeds, adding wearable signals consistently improved cross-validation discrimination over clinical features alone, with a mean paired gain of **+0.0354 AUROC** (ranging from **+0.0266 to +0.0478**).
> - Rather than relying on a single lucky holdout split, holdout test performance averages **0.9084 AUROC** (range: **0.8718 to 0.9258**).

### Model Comparison Table (`models/ablation_summary.csv`)

| Model | CV AUROC A | CV AUROC B | Paired Gain (B - A) | Test AUROC | Test Brier | Untouched Recall | Untouched Precision | Kendall Tau (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Main)** | **0.8684** | **0.8960** | **+0.0276 ± 0.0092** | **0.9165** | **0.0660** | **90.91%** | **35.09%** | **0.6782 ± 0.3911 (IQR 0.3333)** |
| **Random Forest** | 0.8304 | 0.8541 | +0.0237 ± 0.0094 | 0.8022 | 0.0951 | 72.73% | 22.38% | 0.4896 ± 0.4592 (IQR 0.4252) |
| **Gradient Boosting** | 0.8279 | 0.8700 | +0.0421 ± 0.0082 | 0.8663 | 0.0889 | 84.09% | 33.04% | 0.6216 ± 0.4212 (IQR 0.6667) |

### Clinical Safety Threshold & Operational Limitation
- Rather than an arbitrary 0.50 cutoff, the decision threshold was tuned strictly inside validation folds to prioritize patient safety, averaging **0.114** (range: **0.100 to 0.140**; Reference Run: **0.110**).
- **Operational Limitation:** At the CV-tuned threshold, the model catches an average of **88.9% of severe events** (range: **82.4% to 94.6%**) with **37.4% precision** (range: **34.2% to 45.0%**) on small synthetic test cohorts. Approximately 6 out of 10 alerts represent false alarms, underscoring that the system serves as an advisory risk flag to support clinician judgment, not a safety guarantee.

### Timing Ranking Evaluation: Definition & Empirical Spread
- **Kendall Tau Definition:** Computed strictly per patient and treatment cycle between the model's **predicted toxicity risk** and the **TRUE simulated risk** across candidate delays (+0, +3, +7, +14 days). The TRUE simulated risk is derived from the latent ANC trajectory (where lower nadir ANC corresponds to greater acute toxicity risk: $\text{true\_risk} = -\text{nadir\_anc}$). The concordance coefficients are then averaged across all test cycles.
- **Empirical Multi-Seed Distribution:**
  - **Logistic Regression (Main):** Mean Tau across seeds = **0.7251** (range: **0.6384 to 0.8281**; intra-run spread IQR = **0.3333 to 0.6667**).
  - **Random Forest:** Mean Tau across seeds = **0.5790** (range: **0.4896 to 0.6760**).
  - **Gradient Boosting:** Mean Tau across seeds = **0.7033** (range: **0.6216 to 0.7581**).
- **Honest Framing on Timing & Ordering:** We do **not** claim perfect monotonic ordering (+1.0000). Biological noise, nonlinear marrow recovery rates ($r_i$), and stochastic stress spikes produce realistic rank inversions across candidate delays, accurately captured by the spread metrics.
- **Prospective Daily Re-scoring:** All candidate delays are evaluated strictly with retrospective features frozen at decision day $D$. In clinical practice, the twin operates by **daily morning re-scoring**, updating risk as real-world biometrics reveal whether marrow recovery is actually underway.

### Explainability & Upgraded Cost-of-Delay Module
- **Standardized Log-Odds Weights:** Standardized coefficients ([`docs/figures/feature_weights.png`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/docs/figures/feature_weights.png)) highlight key protective and risk factors: lower 7-day HRV ratio and step ratio elevate risk, while greater elapsed time since previous dose and higher pre-dose ANC lower risk.
- **Patient-Specific Gompertz Estimation:** At decision day $D$, [`GompertzTumorModel`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/tumor_model.py) fits patient-specific biological kinetics (growth rate $\alpha$ and chemotherapy kill effect $\kappa$) using bounded nonlinear least squares strictly on retrospective observations ($\text{day\_index} \le D$, zero future lookahead).
- **Forecast Uncertainty (90% Band):** For candidate delays (0, 3, 7, 14 days), the engine generates prospective tumor volume trajectories with a conservative uncertainty ribbon ($\pm 1.645 \cdot \sigma \sqrt{\text{delay} + 1}$) accounting for imaging measurement noise.
- **Held-Out Forecast Accuracy (MAPE Benchmark):**
  - Evaluated on held-out post-decision measurements ($\text{day\_index} > D$) across test patients ($N = 37$):
  - **Patient-Specific Gompertz Fit:** **$4.77\% \pm 3.33\%$ MAPE**
  - **Population-Average Gompertz Curve:** **$8.19\% \pm 6.63\%$ MAPE**
  - Patient-tailored fitting reduces future trajectory forecast error by **41.8%** over static population averages while strictly preserving zero lookahead invariance.
- **Empirical Coverage of 90% Uncertainty Band:**
  - Evaluated across held-out test patients and multiple decision days ($N = 726$ forward evaluations). Coverage was checked only for delay scenarios where the real patient received no dose in the window:
  - **Delay +0 days:** **100.0% coverage** ($N = 222$)
  - **Delay +3 days:** **95.5% coverage** ($N = 202$)
  - **Delay +7 days:** **98.3% coverage** ($N = 179$)
  - **Delay +14 days:** **95.1% coverage** ($N = 123$)
  - **Overall Empirical Coverage:** **Conservative (97.5% empirical coverage vs 90% nominal)**, providing reliable protection against underestimating regrowth uncertainty.

---

## 5. Doctor Dashboard (Streamlit)

The clinical decision-support application ([`app/dashboard.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/app/dashboard.py)) provides interactive simulation:

- **Mandatory Clinical Banner:** Permanent header reminding users: *"For Clinical Decision Support and Simulation Only. Not an Autonomous Prescription Engine."*
- **Patient & Simulation Day Selector:** Select any of the 150 synthetic patients and choose current decision day $D$.
- **Real-Time Patient Summary Cards:** Displays baseline profile, latest ANC, current resting heart rate, and current tumor volume.
- **Tab 1: Candidate Dosing Window & Trade-Off:**
  - Dual-axis interactive Plotly chart comparing toxicity risk (%) against fitted Gompertz tumor regrowth penalty (%) across candidate windows (+0, +3, +7, +14 days).
  - 90% confidence ribbon for prospective tumor volume.
  - **Clinical Action Matrix:** Translates probabilities into actionable tiers (🟢 Lower Risk $< 0.11$, 🟡 Elevated $0.11–0.22$, 🔴 High Risk $> 0.22$) with volume 90% CI and operational trade-off metrics.
- **Tab 2: Physiological Trajectories:**
  - Longitudinal Absolute Neutrophil Count (ANC) trajectory with Grade 3/4 severe neutropenia threshold ($1.0 \times 10^9/\text{L}$) callout.
  - Longitudinal Heart Rate Variability (HRV RMSSD) showing personal baseline vs 7-day rolling window.
- **Tab 3: Model Performance & Ablation:**
  - Live ablation summary table across Logistic Regression, Random Forest, and Gradient Boosting.
  - Embedded ROC curve ([`roc_curve_ablation.png`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/docs/figures/roc_curve_ablation.png)) and Calibration Curve ([`calibration_curve.png`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/docs/figures/calibration_curve.png)).

---

## 6. Demo Walkthrough

1. **Launch Dashboard:** Run `streamlit run app/dashboard.py`.
2. **Select Patient & Decision Point:** Load patient `PT_001` at Day 21 (nominal Cycle 2 dosing day).
3. **Inspect Autonomic State:** Wearables show suppressed HRV RMSSD and elevated resting heart rate relative to the 30-day personal baseline, indicating unresolved biological strain.
4. **Evaluate Trade-Off Curve:**
   - Day +0 (today) carries high severe toxicity risk (> 40%).
   - Postponing by +7 days lowers predicted toxicity risk below the safety threshold ($0.11$) as marrow recovery proceeds, at the cost of modest tumor regrowth (+8.4% excess volume).
5. **Review Gompertz Uncertainty:** Examine the 90% tumor volume forecast interval to ensure tumor regrowth remains within safe clinical bounds.
6. **Clinician Judgment:** The oncologist reviews drivers, explains rationale to the patient, and schedules lab verification.

---

## 7. Development Roadmap & Completed Build Order

| Step | Task | Status | Details |
|:---:|:---|:---:|:---|
| **1** | Synthetic Data Generator | **Completed** | Generates 150 patients, 90 days, 600 doses, Friberg-style myelosuppression, Gompertz tumor kinetics, noisy lagged wearables ([`src/generate_data.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/generate_data.py)). |
| **2** | Cost-of-Delay Gompertz Module | **Completed** | Patient-specific parameter fitting ($\alpha, \kappa$), 90% uncertainty band, held-out MAPE validation (4.77% vs 8.19%), empirical coverage validation (97.5%) ([`src/tumor_model.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/tumor_model.py)). |
| **3** | Feature Engineering Pipeline | **Completed** | Strictly retrospective features ($\le D$), 7d rolling trends vs 30d baseline, timing candidate features, zero future lookahead ([`src/features.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/features.py)). |
| **4** | Toxicity Model Training & Ablations | **Completed** | Repeated 5x5 Group CV, paired diffs (+0.0354 AUROC gain), validation-tuned thresholding (0.11), Kendall Tau timing ranking (0.7251), multi-seed evaluation across 5 seeds ([`src/train_toxicity.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/train_toxicity.py)). |
| **5** | Evaluation & Artifact Generation | **Completed** | ROC curves, calibration diagrams, PR operating curves, feature weight bar plots generated in `docs/figures/` ([`src/evaluate.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/evaluate.py)). |
| **6** | Streamlit Clinician Dashboard | **Completed** | Interactive 3-tab decision support app, dual-axis trade-off chart, action matrix, trajectory monitoring ([`app/dashboard.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/app/dashboard.py)). |
| **7** | Automated Test Suite | **Completed** | 12 tests passing (`python -m pytest -q`): tests data generation, Gompertz kinetics, future perturbation invariance, and label window alignment. |
| **8** | Code Quality & Documentation | **Completed** | Ruff linting clean (`python -m ruff check .`), comprehensive scientific references ([`docs/references.md`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/docs/references.md)), architecture and presentation decks in `docs/`. |

---

## 8. Repository Structure

```
Aurora_TheApolloUniversity/
├── README.md                      # Comprehensive project documentation & evaluation reports
├── LICENSE                        # MIT License
├── requirements.txt               # Pinned Python package dependencies
├── ruff.toml                      # Ruff linting configuration
├── task.md                        # Project tracking and audit checklist
├── repomix-output.xml             # Context pack artifact
├── app/
│   └── dashboard.py               # Interactive Streamlit clinician decision-support dashboard
├── data/                          # Longitudinal CSV datasets (synthetic)
│   ├── patient_profiles.csv       # 150 patient demographic & baseline profiles
│   ├── clinical_logs.csv          # 13,500 daily clinical lab & treatment records
│   ├── wearable_streams.csv       # 13,500 daily biometric time series records
│   ├── timing_eval.csv            # 1,200 candidate dosing day counterfactual evaluations
│   ├── processed_dataset.csv      # 1,500 engineered training samples
│   └── timing_features.csv        # 1,200 engineered prospective timing evaluation features
├── docs/                          # Presentations, architecture, and scientific references
│   ├── architecture.pdf           # End-to-end system architecture schematic
│   ├── presentation.pdf           # Challenge slide deck
│   ├── references.md              # Scientific and mathematical citations
│   └── figures/                   # Generated evaluation plots
│       ├── roc_curve_ablation.png
│       ├── calibration_curve.png
│       ├── precision_recall_operating_point.png
│       └── feature_weights.png
├── models/                        # Serialized ML artifacts and benchmarks
│   ├── main_toxicity_model.joblib # Standardized Logistic Regression main model
│   ├── rf_toxicity_model.joblib   # Random Forest benchmark
│   ├── gb_toxicity_model.joblib   # Gradient Boosting benchmark
│   ├── scaler.joblib              # StandardScaler fitted on training folds
│   ├── feature_names.joblib       # 16-feature schema list
│   ├── model_meta.joblib          # Model metadata (optimal threshold, test metrics)
│   └── ablation_summary.csv       # Complete ablation benchmark metrics table
├── src/                           # Core algorithms and pipelines
│   ├── generate_data.py           # Synthetic patient & wearable stream generator
│   ├── tumor_model.py             # Gompertz tumor regrowth engine & uncertainty bands
│   ├── features.py                # Feature extraction & zero-leakage pipeline
│   ├── train_toxicity.py          # Model training, Group CV, and threshold tuning
│   └── evaluate.py                # Artifact generation & explainability plots
└── tests/                         # Pytest automated test suite (12 tests)
    ├── test_generate_data.py       # Data generator shape and column sanity tests
    ├── test_tumor_model.py         # Gompertz step, delay tradeoff, MAPE, and coverage tests
    └── test_features_and_models.py # Future perturbation invariance & label window tests
```

---

## 9. Technical Stack & Verification Instructions

### Tech Stack
- **Language & Runtime:** Python 3.10+
- **Data & Computation:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-Learn, Joblib
- **Visualization:** Plotly, Matplotlib, Seaborn
- **User Interface:** Streamlit
- **Code Quality & Testing:** Pytest, Ruff

### Setup & Local Execution

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/Aurora_TheApolloUniversity.git
   cd Aurora_TheApolloUniversity
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run automated test suite (12 tests):**
   ```bash
   python -m pytest -q
   ```

4. **Verify linting compliance:**
   ```bash
   python -m ruff check .
   ```

5. **Regenerate data and retrain models (optional):**
   ```bash
   python src/generate_data.py --patients 150 --days 90 --seed 42
   python src/train_toxicity.py
   python src/evaluate.py
   ```

6. **Launch doctor decision-support dashboard:**
   ```bash
   streamlit run app/dashboard.py
   ```

---

## 10. Limitations & Safety

> [!WARNING]
> **Essential Methodological Limitations & Non-Clinical Status:**
> - **Synthetic Proxy Assumption:** The observed paired wearable gain (mean $+0.0354$ CV AUROC, range $+0.0266$ to $+0.0478$ across 5 seeds) directly depends on the synthetic generator's assumption that autonomic wearable trends (RHR, HRV, steps) serve as noisy proxies for an unobserved biological recovery rate ($r_i$) and latent marrow vulnerability. If real-world wearable signals are decoupled from bone marrow nadir dynamics, this incremental utility may not generalize.
> - **Optimistic Tumor Forecast Validation:** Tumor volume forecasts are validated only on synthetic data generated from the same Gompertz mathematical family; real-world tumors often deviate from pure Gompertzian dynamics, making this error comparison (4.77% vs. 8.19% MAPE) optimistic.
> - **Approximate Uncertainty Band:** The 90% tumor forecast uncertainty band relies on a normal measurement error approximation and accumulated variance propagation; it represents an approximate uncertainty guide rather than a rigorous Bayesian credible interval.
> - **Cohort Size & Seed Variability:** All toxicity and calibration metrics come from a small synthetic cohort (150 patients) and exhibit meaningful empirical variation across random generator seeds (e.g., recall varies from 82.4% to 94.6%, precision from 34.2% to 45.0%, and Kendall tau from 0.6384 to 0.8281).
> - **Risk Flag, Not Safety Guarantee:** At the CV-tuned threshold (mean 0.114), the model catches an average of 88.9% of severe events with 37.4% precision across seeds, meaning it serves as an advisory risk flag to support clinician judgment rather than a definitive safety guarantee.
> - **Synthetic Data Only:** All findings and performance figures reflect simulated data, not clinically validated patient outcomes.
> - **Not Monotonically Perfect:** Dosing delay ranking exhibits realistic variance (Tau mean 0.7251, range 0.6384 to 0.8281 across seeds) due to individual recovery kinetics and biological noise; perfect ordering should never be claimed.
> - **Decision Support Only:** The system generates informational risk trajectories for simulation and pair-programming research. It is **not** an autonomous prescribing device.
> - **Regulatory Prerequisites:** Real-world translational application mandates clinical trial validation, formal prospective auditing, and full regulatory clearance (DPDP Act, HIPAA, FDA SaMD).

---

## 11. License

MIT License. See [`LICENSE`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/LICENSE).
