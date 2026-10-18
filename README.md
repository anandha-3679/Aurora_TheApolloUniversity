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
- [x] Comprehensive test suite passing with zero-leakage assertions (30 passing tests in [`tests/`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/tests/))
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
| **Paired Wearable Gain $(B - A)$** | **+0.0096 AUROC** | **-0.0087 to +0.0212** | **+0.0113 ± 0.0081** |
| **CV AUROC Ablation B (+ Wearables)** | **0.8624** | **0.8392 to 0.8815** | **0.8767** |
| **CV AUROC Ablation A (Clinical only)**| **0.8528** | **0.8358 to 0.8654** | **0.8654** |
| **Holdout Test AUROC** | **0.8914** | **0.7738 to 0.9531** | **0.7738** |
| **Holdout Test Brier Score** | **0.0964** | **0.0544 to 0.1280** | **0.1280** |
| **CV-Tuned Threshold** | **0.108** | **0.100 to 0.120** | **0.100** |
| **Untouched Test Recall** | **90.89%** | **84.75% to 96.08%** | **84.75%** |
| **Untouched Test Precision** | **42.73%** | **26.18% to 52.23%** | **26.18%** |
| **Timing Ranking Kendall Tau** | **0.3203** | **-0.1337 to 0.4714** | **0.4326** |

> **Methodological Honesty Note on Multi-Seed Metrics:**
> - Across 5 distinct random generation seeds under physiological upstream marrow coupling ($\kappa = 1.0$), continuous passive wearable biometrics provide an average cross-validation discrimination gain of **+0.0096 AUROC** (ranging from **-0.0087 to +0.0212**; reference seed 42 paired difference: **+0.0113 ± 0.0081**).
> - Test set generalization across held-out patient groups averages **0.8914 AUROC** (range: **0.7738 to 0.9531**), demonstrating stable out-of-sample risk classification without data leakage.

### Model Comparison Table (`models/ablation_summary.csv`)

| Model | CV AUROC A | CV AUROC B | Paired Gain (B - A) | Test AUROC | Test Brier | Untouched Recall | Untouched Precision | Kendall Tau (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Main)** | **0.8654** | **0.8767** | **+0.0113 ± 0.0081** | **0.7738** | **0.1280** | **84.75%** | **26.18%** | **0.4326 ± 0.5586 (IQR 0.8174)** |
| **Random Forest** | 0.8453 | 0.8227 | -0.0226 ± 0.0084 | 0.6976 | 0.1284 | 69.49% | 24.85% | 0.4494 ± 0.5239 (IQR 0.7303) |
| **Gradient Boosting** | 0.8373 | 0.8307 | -0.0065 ± 0.0089 | 0.7013 | 0.1372 | 66.10% | 23.08% | 0.4356 ± 0.5667 (IQR 0.7303) |

> **Tree Model Wearable Degradation Note:**
> Appending continuous wearable biometric ratios slightly degraded tree ensembles (Random Forest by **-0.0226 AUROC**, Gradient Boosting by **-0.0065 AUROC**) due to noise overfitting on non-stationary biometric fluctuations. Conversely, L2-regularized Logistic Regression successfully filtered redundant dimensions and captured the underlying signal (+0.0113 in reference run, +0.0096 multi-seed average), confirming it as our empirically superior main model.

### Clinical Safety Threshold & Operational Limitation
- Rather than an arbitrary 0.50 cutoff, the decision threshold was tuned strictly inside validation folds to prioritize patient safety, averaging **0.108** (range: **0.100 to 0.120**; Reference Run: **0.100**).
- **Operational Limitation:** At the CV-tuned threshold, the model catches an average of **90.9% of severe events** (range: **84.8% to 96.1%**) with **42.7% precision** (range: **26.2% to 52.2%**) on held-out test cohorts. In the conservative Seed 42 reference run, precision is 26.2% (~74% false alarm rate), underscoring that the system serves as an advisory early-warning risk flag to support oncologist review, not an autonomous safety guarantee.

### Timing Ranking Evaluation: Dropping "Predicts the Best Day"
- **Kendall Tau Definition:** Computed strictly per patient and treatment cycle between the model's **predicted toxicity risk** and the **TRUE simulated risk** across candidate delays (+0, +3, +7, +14 days). The TRUE simulated risk is derived from the latent ANC trajectory ($\text{true\_risk} = -\text{nadir\_anc}$).
- **Empirical Multi-Seed Distribution (Classification Model):**
  - **Logistic Regression (Main):** Mean Tau across seeds = **0.3203** (range: **-0.1337 to 0.4714**; Reference Run: **0.4326 ± 0.5586, IQR 0.8174**).
  - **Random Forest:** Mean Tau across seeds = **0.3703** (range: **0.2604 to 0.4494**).
  - **Gradient Boosting:** Mean Tau across seeds = **0.3885** (range: **0.1936 to 0.4764**).
- **Mechanistic Friberg ODE Twin Ranking:**
  - Evaluated directly on forward-simulated nadir trajectories against true simulated nadirs:
  - **Mean Kendall Tau:** **0.4003 ± 0.5614** (Median Tau: **0.5477**, $N = 300$ cycles).
- **Explicit Methodological Stance:** Because a Kendall Tau of 0.32–0.40 indicates moderate, imperfect monotonic ordering over candidate dosing days, we explicitly **drop any claim that ONCO-TWIN "predicts the best day"**. The system provides advisory risk stratification across candidate windows rather than claiming deterministic day optimization.
- **Prospective Daily Re-scoring:** All candidate delays are evaluated strictly with retrospective features frozen at decision day $D$. In clinical practice, the twin operates by **daily morning re-scoring**, updating risk as real-world biometrics reveal whether marrow recovery is actually underway.

### Explainability & Upgraded Cost-of-Delay Module
- **Standardized Log-Odds Weights:** Standardized coefficients ([`docs/figures/feature_weights.png`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/docs/figures/feature_weights.png)) highlight key protective and risk factors: lower 7-day HRV ratio and step ratio elevate risk, while greater elapsed time since previous dose and higher pre-dose ANC lower risk.
- **Patient-Specific Gompertz Estimation:** At decision day $D$, [`GompertzTumorModel`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/tumor_model.py) fits patient-specific biological kinetics (growth rate $\alpha$ and chemotherapy kill effect $\kappa$) using bounded nonlinear least squares strictly on retrospective observations ($\text{day\_index} \le D$, zero future lookahead).
- **Forecast Uncertainty (90% Band):** For candidate delays (0, 3, 7, 14 days), the engine generates prospective tumor volume trajectories with a conservative uncertainty ribbon ($\pm 1.645 \cdot \sigma \sqrt{\text{delay} + 1}$) accounting for imaging measurement noise.
- **Held-Out Forecast Accuracy (Verified Re-run Benchmark):**
  - Evaluated on held-out post-decision measurements ($\text{day\_index} > D$) across test patients ($N = 37$):
  - **Patient-Specific Gompertz Fit:** **$4.74\% \pm 3.72\%$ MAPE**
  - **Population-Average Gompertz Curve:** **$7.73\% \pm 6.44\%$ MAPE**
  - Patient-tailored fitting reduces future trajectory forecast error by **38.7%** over static population averages while strictly preserving zero lookahead invariance.
- **Empirical Coverage of 90% Uncertainty Band:**
  - Evaluated across held-out test patients and multiple decision days ($N = 736$ forward evaluations):
  - **Delay +0 days:** **100.0% coverage** ($N = 222$)
  - **Delay +3 days:** **97.1% coverage** ($N = 204$)
  - **Delay +7 days:** **98.9% coverage** ($N = 182$)
  - **Delay +14 days:** **95.3% coverage** ($N = 128$)
  - **Overall Empirical Coverage:** **Conservative (98.1% empirical coverage vs 90% nominal)**, providing reliable protection against underestimating regrowth uncertainty.

### Mechanistic Bone Marrow Module Fit Checks & "Experimental" Status
To determine whether the bone marrow mechanistic twin ([`src/friberg_model.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/friberg_model.py)) can transition from experimental to validated status, we executed explicit numerical boundary and monotonicity audits across all 150 patients (300 cycles, 900 candidate delay transitions):
1. **Parameter Boundary Hit Rate:**
   - Maturation Transit Time ($MTT$): **0/150 hits (0.0%)** on bounds ($[2.0, 15.0]\text{ days}$).
   - Hematopoietic Recovery Rate ($\gamma$): **0/150 hits (0.0%)** on bounds ($[0.05, 0.8]$).
   - Drug Sensitivity Slope: **5/150 hits (3.3%)** on upper bound ($10.0$).
   - Overall, **96.7% of parameter fits reside strictly in the interior** without boundary pinning.
2. **Nadir Delay Monotonicity Rate:**
   - Proportion of candidate delay steps ($0 \to 3\text{d}, 3 \to 7\text{d}, 7 \to 14\text{d}$) where predicted post-dose nadir strictly rises or recovers: **49.8% (448/900 transitions)**.
   - **Critical Finding:** This falls well short of the target $\ge 95\%$ monotonicity criterion. In non-linear myelosuppressive kinetics, delaying an infusion allows intermediate marrow rebound, but once cytotoxic drug is re-administered, the acute cell-kill depth remains primarily dictated by drug dose and clearance kinetics rather than disappearing.
3. **Status Verdict:** Because nadir monotonicity across candidate delays does not achieve $\ge 95\%$, the Bone Marrow Mechanistic Twin **strictly retains its "Experimental / Research Prototype" designation** across all documentation and interfaces.

### Coupling Sensitivity Stress Test (Biological Plausibility Deliverable)

A central critique of multimodal wearable integration in oncology is: *Why should wearable biosensors add predictive value if clinical laboratory tests (pre-dose ANC) are already collected at decision day $D$?*

The biological rationale lies in the **transit-compartment delay** of human granulopoiesis (Friberg et al., 2002):
1. **Mitotic Precursor Shock:** Cytotoxic chemotherapy primarily damages proliferating progenitor stem cells ($Prol$) in the bone marrow within days 1–3 post-infusion.
2. **Transit Compartment Lag:** Because mature circulating neutrophils ($Circ$) take an average maturation transit time ($MTT \approx 5.4 \text{ days}$) across compartments $T_1, T_2, T_3$, circulating blood tests at day 21 often appear rebounded even when the upstream stem pool and hematopoietic reserve remain heavily depleted by cumulative toxicity.
3. **Autonomic Leading Indicator Hypothesis:** Systemic physiological stress, subclinical inflammatory cascades, and autonomic dysregulation manifest in continuous biometrics (dampened HRV, elevated resting heart rate, activity decline) days before a secondary circulating crash occurs.

To rigorously test this hypothesis and confirm our model does not hallucinate wearable value, we performed a **Coupling Strength Stress Test** sweeping parameter $\kappa \in [0.0, 1.0]$, where $\kappa = 0.0$ represents completely uncoupled wearables (noise features) and $\kappa = 1.0$ represents full physiological coupling to upstream progenitor depletion ($Prol, T_1$).

![Coupling Sensitivity Stress Test](docs/figures/coupling_sensitivity_stress_test.png)

#### Multi-Seed Coupling Sensitivity Results (5-Seed Repeated Group CV)

| Coupling Strength ($\kappa$) | Clinical Baseline (Ablation A) | Multimodal Model (Ablation B) | Paired Lift $\Delta$ AUROC ($B - A$) [95% CI] | Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| **$\kappa = 0.00$** | 0.8528 | 0.8424 | **-0.0104 ± 0.0072** (std=0.0082) | **Uncoupled Noise Penalty:** Adding uninformative noise features degrades out-of-fold generalization. |
| **$\kappa = 0.25$** | 0.8528 | 0.8592 | **+0.0064 ± 0.0115** (std=0.0131) | **Emergent Signal:** Mild autonomic coupling overcomes feature dimensionality and lifts discrimination. |
| **$\kappa = 0.50$** | 0.8528 | 0.8617 | **+0.0089 ± 0.0116** (std=0.0132) | **Robust Signal:** Wearable trend metrics capture early progenitor reserve strain. |
| **$\kappa = 0.75$** | 0.8528 | 0.8623 | **+0.0095 ± 0.0111** (std=0.0126) | **Plateauing Value:** Autonomic leading indicators saturate early-warning utility. |
| **$\kappa = 1.00$** | 0.8528 | 0.8624 | **+0.0096 ± 0.0101** (std=0.0115) | **Full Upstream Coupling:** Matches headline benchmark (+0.0096 AUROC; positive in 4 of 5 seeds, range -0.0087 to +0.0212). |

> **Takeaway:** At $\kappa = 0.0$, the model suffers a $-0.0104$ noise penalty, demonstrating that our evaluation setup does not generate spurious lift. As biological coupling increases, the lift rises monotonically and plateaus at $+0.0096$, fully matching our headline 5-seed benchmark (+0.0096 AUROC; positive in 4 of 5 seeds, range -0.0087 to +0.0212).

### In-Silico Simulation Limitations & Digital Twin Scope

To ensure complete scientific and methodological integrity, we explicitly declare the biological and clinical boundaries of this system:

1. **In-Silico Self-Consistency vs. Clinical Efficacy:**
   - The mechanistic bone marrow model ([`src/friberg_model.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/friberg_model.py)) fits longitudinal trajectories with a low mean residual error ($\text{std} = 0.3559 \times 10^9/\text{L}$).
   - **Important Distinction:** This low fitting error confirms **mathematical identifiability, structural parameter recovery, and algorithmic stability** in an in-silico environment where the twin's ODE equations correspond to the underlying myelosuppressive physics. It does **not** constitute proof of clinical efficacy on real human oncology cohorts.
2. **Coupling Hypothesis Status:**
   - The link between bone marrow stem cell depletion and continuous autonomic biometrics (HRV, RHR, steps) is a scientifically motivated, biologically plausible hypothesis supported by autonomic-hematopoietic literature, but remains an in-silico construct.
   - Prospective clinical trials with medical-grade wearable sensors and serial laboratory complete blood counts (CBC) are required before any real-world bedside deployment.
3. **Advisory Decision Support:**
   - ONCO-TWIN is strictly an advisory risk stratification and simulation instrument. All dosing and delay recommendations must undergo qualified multidisciplinary oncologist evaluation.

---

## 5. Doctor Dashboard (Streamlit)

The clinical decision-support application ([`app/dashboard.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/app/dashboard.py)) provides interactive simulation:

- **Mandatory Clinical Banner:** Permanent header reminding users: *"For Clinical Decision Support and Simulation Only. Not an Autonomous Prescription Engine."*
- **Patient & Simulation Day Selector:** Select any of the 150 synthetic patients and choose current decision day $D$.
- **Real-Time Patient Summary Cards:** Displays baseline profile, latest ANC, current resting heart rate, and current tumor volume.
- **Tab 1: Candidate Dosing Window & Trade-Off:**
  - Dual-axis interactive Plotly chart comparing toxicity risk (%) against fitted Gompertz tumor regrowth penalty (%) across candidate windows (+0, +3, +7, +14 days).
  - 90% confidence ribbon for prospective tumor volume.
  - **Clinical Action Matrix:** Translates probabilities into actionable tiers (🟢 Lower Risk $< \theta$, 🟡 Elevated $\theta–2\theta$, 🔴 High Risk $> 2\theta$, where $\theta \approx 0.10$ is the CV-tuned operating threshold) with volume 90% CI and operational trade-off metrics.
  - **Experimental Marrow Model Toggle:** A dedicated sidebar toggle (*"Show experimental marrow (Friberg) model"*, default **OFF**) isolates the exploratory Friberg PK/PD module. When switched ON, a prominent clinical warning banner is displayed alongside mechanistic nadir curves and hybrid risk predictions.
- **Tab 2: Physiological Trajectories:**
  - Longitudinal Absolute Neutrophil Count (ANC) trajectory with Grade 3/4 severe neutropenia threshold ($1.0 \times 10^9/\text{L}$) callout.
  - Longitudinal Heart Rate Variability (HRV RMSSD) showing personal baseline vs 7-day rolling window.
- **Tab 3: Model Performance & Ablation:**
  - **Headline Multi-Seed Benchmark Table:** Prominently displays aggregate performance across 5 generator seeds (`multi_seed_aggregate_table.csv`) to prevent single-seed bias.
  - **Reference Run Benchmark:** Live ablation summary across Logistic Regression, Random Forest, and Gradient Boosting for reference Seed 42 (`models/ablation_summary.csv`).
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
| **2** | Cost-of-Delay Gompertz Module | **Completed** | Patient-specific parameter fitting ($\alpha, \kappa$), 90% uncertainty band, held-out MAPE validation (4.74% vs 7.73%), empirical coverage validation (98.1%) ([`src/tumor_model.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/tumor_model.py)). |
| **3** | Feature Engineering Pipeline | **Completed** | Strictly retrospective features ($\le D$), 7d rolling trends vs 30d baseline, timing candidate features, zero future lookahead ([`src/features.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/features.py)). |
| **4** | Toxicity Model Training & Ablations | **Completed** | Repeated 5x5 Group CV, paired diffs (+0.0096 AUROC mean gain across 5 seeds; reference run +0.0113), validation-tuned thresholding (0.108 mean, 0.100 ref), Kendall Tau timing ranking (0.3203 mean, 0.4326 ref; mechanistic ODE twin tau 0.4003) ([`src/train_toxicity.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/train_toxicity.py)). |
| **5** | Evaluation & Artifact Generation | **Completed** | ROC curves, calibration diagrams, PR operating curves, coupling stress tests, feature weight bar plots generated in `docs/figures/` ([`src/evaluate.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/src/evaluate.py)). |
| **6** | Streamlit Clinician Dashboard | **Completed** | Interactive 3-tab decision support app with experimental Friberg sidebar toggle, dual-axis trade-off chart, action matrix, trajectory monitoring ([`app/dashboard.py`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/app/dashboard.py)). |
| **7** | Automated Test Suite | **Completed** | 30 tests passing (`python -m pytest -q`): tests data generation, Gompertz kinetics, future perturbation invariance, label window alignment, and Friberg ODE kinetics. |
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
│       ├── feature_weights.png
│       ├── coupling_sensitivity_stress_test.png
│       └── multi_seed_aggregate_table.csv
├── models/                        # Serialized ML artifacts and benchmarks
│   ├── main_toxicity_model.joblib # Standardized Logistic Regression main model
│   ├── rf_toxicity_model.joblib   # Random Forest benchmark
│   ├── gb_toxicity_model.joblib   # Gradient Boosting benchmark
│   ├── scaler.joblib              # StandardScaler fitted on training folds
│   ├── feature_names.joblib       # 16-feature schema list
│   ├── model_meta.joblib          # Model metadata (optimal threshold, test metrics)
│   └── ablation_summary.csv       # Complete ablation benchmark metrics table
├── scripts/                       # Reproducibility and benchmark evaluation scripts
│   ├── run_multi_seed_benchmark.py # Multi-seed 5-seed benchmark runner & table builder
│   ├── generate_coupling_curve.py  # Autonomic-marrow coupling stress test sweep
│   └── check_friberg_and_tumor.py # Boundary hit rates, nadir monotonicity & tumor checks
├── src/                           # Core algorithms and pipelines
│   ├── generate_data.py           # Synthetic patient & wearable stream generator
│   ├── friberg_model.py           # Mechanistic 5-compartment Friberg bone marrow PK/PD
│   ├── tumor_model.py             # Gompertz tumor regrowth engine & uncertainty bands
│   ├── features.py                # Feature extraction & zero-leakage pipeline
│   ├── train_toxicity.py          # Model training, Group CV, and threshold tuning
│   └── evaluate.py                # Artifact generation & explainability plots
└── tests/                         # Pytest automated test suite (30 passing tests)
    ├── test_generate_data.py       # Data generator shape and column sanity tests
    ├── test_tumor_model.py         # Gompertz step, delay tradeoff, MAPE, and coverage tests
    ├── test_features_and_models.py # Future perturbation invariance & label window tests
    └── test_friberg_model.py       # ODE kinetics, EKF observer, and state recovery tests
```

---

## 9. Technical Stack & Reproduce Results

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
   git clone https://github.com/anandha-3679/Aurora_TheApolloUniversity.git
   cd Aurora_TheApolloUniversity
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Reproduce Results (Exact Execution Order)

To verify and reproduce all headline metrics, multi-seed benchmarks, and figures reported in this README, run the following commands from the repository root:

1. **Run full automated test suite (30 passing tests):**
   ```bash
   python -m pytest -q
   ```

2. **Verify linting compliance (errors-only, clean):**
   ```bash
   python -m ruff check .
   ```

3. **Train baseline models & generate core diagnostic figures:**
   ```bash
   python src/train_toxicity.py
   python src/evaluate.py
   ```

4. **Run the 5-seed cross-validation & holdout benchmark:**
   ```bash
   python scripts/run_multi_seed_benchmark.py
   ```
   *Generates `docs/figures/multi_seed_aggregate_table.csv` and `models/ablation_summary.csv`.*

5. **Run the autonomic-marrow coupling sensitivity stress test:**
   ```bash
   python scripts/generate_coupling_curve.py
   ```
   *Generates `docs/figures/coupling_sensitivity_stress_test.png` and `docs/figures/coupling_sensitivity_data.csv`.*

6. **Audit Friberg parameter bound hits, nadir monotonicity, and tumor model re-run:**
   ```bash
   python scripts/check_friberg_and_tumor.py
   ```
   *Verifies boundary rates, nadir monotonicity (49.8%), and Gompertz MAPE (4.74%).*

7. **Launch interactive clinician decision-support dashboard:**
   ```bash
   streamlit run app/dashboard.py
   ```

---

## 10. Limitations & Safety

> [!WARNING]
> **Essential Methodological Limitations & Non-Clinical Status:**
> - **Synthetic Proxy Assumption:** The observed paired wearable gain (mean $+0.0096$ CV AUROC, range $-0.0087$ to $+0.0212$ across 5 seeds; reference run $+0.0113 \pm 0.0081$) directly depends on the synthetic generator's physiological assumption that autonomic wearable trends (RHR, HRV, steps) serve as noisy, lagged proxies for upstream marrow strain. If real-world wearable signals are decoupled from bone marrow nadir dynamics ($\kappa = 0$), this incremental utility collapses to a noise penalty ($-0.0104$ AUROC).
> - **Optimistic Tumor Forecast Validation:** Tumor volume forecasts are validated on synthetic data generated within the Gompertz mathematical family; real-world tumors may deviate from pure Gompertzian dynamics, making this error comparison (4.74% vs. 7.73% MAPE) optimistic.
> - **Approximate Uncertainty Band:** The 90% tumor forecast uncertainty band relies on a normal measurement error approximation and accumulated variance propagation; it represents an approximate uncertainty guide rather than a rigorous Bayesian credible interval, though it demonstrated 98.1% empirical coverage.
> - **Cohort Size & Seed Variability:** All toxicity and calibration metrics come from a synthetic cohort of 150 patients and exhibit meaningful empirical variation across random generator seeds (e.g., test recall varies from 84.8% to 96.1%, precision from 26.2% to 52.2%, and Kendall tau from -0.1337 to 0.4714).
> - **Advisory Risk Flag, Not Safety Guarantee:** At the CV-tuned threshold (mean 0.108, reference run 0.100), the model catches an average of 90.9% of severe events with 42.7% precision across seeds (26.2% precision in Seed 42 reference run, ~74% false alarm rate), meaning it serves as an advisory early-warning risk flag to support oncologist review, not an autonomous safety guarantee.
> - **Synthetic Data Only:** All findings and performance figures reflect simulated data, not clinically validated patient outcomes.
> - **Timing Ranking Ordering:** Dosing delay ranking exhibits moderate correlation (Kendall Tau mean 0.3203 across seeds for ML; 0.4003 for Friberg ODE twin). We explicitly drop any claim that ONCO-TWIN "predicts the best day" and treat candidate delay scoring as advisory risk stratification across candidate windows.
> - **Decision Support Only:** The system generates informational risk trajectories for simulation and clinical decision support. It is **not** an autonomous prescribing device.
> - **Regulatory Prerequisites:** Real-world translational application mandates clinical trial validation, formal prospective auditing, and full regulatory clearance (DPDP Act, HIPAA, FDA SaMD).

---

## 11. License

MIT License. See [`LICENSE`](file:///c:/Users/Anandha%20Lakshmi/Digital_twin/LICENSE).
