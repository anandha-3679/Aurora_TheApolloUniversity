# Task Management

## Architecture Overview
- **Project**: ONCO-TWIN (Digital Twin for predicting chemotherapy toxicity & choosing safest dosing window)
- **Stack**: Python (Pandas, NumPy, Scikit-Learn, SHAP, Plotly, Streamlit)
- **Testing & Quality**: `pytest -q`, `ruff` (errors-only mode)
- **Key Modules**:
  - `src/generate_data.py`: Multi-tier synthetic clinical & wearable streams generator
  - `src/tumor_model.py`: Gompertz tumor regrowth cost-of-delay engine
  - `src/features.py`: Feature engineering pipeline (trends vs rolling baseline)
  - `src/train_toxicity.py`: Toxicity prediction model (patient-level split)
  - `src/evaluate.py`: Metrics, calibration, ablation studies, and SHAP explainability
  - `app/dashboard.py`: Interactive Streamlit clinician decision-support dashboard

## Immediate Next Steps
- [x] Task 1 (Audit): Trace candidate dosing day feature builder (`src/features.py`), timing dataset (`data/timing_eval.csv`), dashboard (`app/dashboard.py`), and training pipeline (`src/train_toxicity.py`) to document all data leakage from days > decision day D.
- [x] Task 2 (Fix): Refactor feature builder to freeze wearable and lab metrics at decision day D (<= D), represent delay via known scheduling attributes (`days_until_dose` / `delay_days`, `days_since_last_dose`), ensuring zero future wearable lookahead.
- [x] Task 3 (Training & Timing Realignment): Rebuild training dataset where decisions at day D predict outcomes under candidate delays using strictly <= D features.
- [ ] Task 4 (Tests): Implement pytest suite asserting future perturbation invariance (perturbing data > D leaves predictions for all delays identical) and verifying label windows start post-dose.
- [ ] Task 5 (Re-evaluation): Rerun repeated 5x5 Group CV, paired diffs (A vs B), validation-only threshold tuning, and intra-patient Kendall tau; report honest before vs after metrics.
- [ ] Task 6 (Documentation): Update `README.md` and `app/dashboard.py` methodology notes to state strictly zero future lookahead.


## Completed Tasks
- [x] Initial project bootstrap (.antigravityrules, ruff.toml, .gitignore, task.md, repomix mapping)
- [x] Step 1: Synthetic data generator (`src/generate_data.py`), CSV outputs generated (600 doses, 37% severe toxicity rate), tests passing (`pytest -q`)
- [x] Step 2: Cost-of-delay Gompertz module (`src/tumor_model.py`) with delay penalty trade-off curves, tests passing (`pytest -q`)
- [x] Step 3: Feature engineering (`src/features.py`), Toxicity model training (`src/train_toxicity.py`), patient-level split, and ablation comparisons (Logistic Regression, Random Forest, Gradient Boosting)
- [x] Phase 1 Refactor: Dynamic cycle-varying latent recovery factor in `src/generate_data.py`, giving wearables non-redundant predictive signal, + intra-patient timing ground-truth dataset (`data/timing_eval.csv`)
- [x] Phase 2 Refactor: Timing evaluation dataset pipeline added to `src/features.py` (`build_timing_evaluation_dataset`), computing 7d/30d wearable deviations per candidate dosing day; verified with tests passing (`pytest -q`)
- [x] Phase 3 Refactor: Repeated patient-level CV with paired differences (+0.0518 ± 0.0093 AUROC), intra-patient timing ranking (Tau -0.776), validation-tuned thresholding (0.17 threshold -> 94.1% untouched test recall), and empirical selection of Logistic Regression as Main Model
- [x] Phase 4: Full evaluation artifact generation (`src/evaluate.py`), interactive Streamlit decision-support dashboard (`app/dashboard.py`), transparent precision/recall trade-off callouts, and honest documentation in `README.md`

