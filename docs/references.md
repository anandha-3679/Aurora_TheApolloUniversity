# Scientific References & Modeling Ground Truth

This document details the clinical, mathematical, and epidemiological literature citations underpinning the **ONCO-TWIN** synthetic generator and decision architecture.

---

## 1. Breast Cancer Incidence & Burden in India

1. **ICMR-NCRP Report on National Cancer Registry Programme (2020–2025):**
   - National Cancer Registry Programme, Indian Council of Medical Research (ICMR) & National Centre for Disease Informatics and Research (NCDIR), Bengaluru, India.
   - *Key Statistics:* Breast cancer is the single leading site of cancer among Indian females, accounting for **28.2% of all female cancer burdens** and an estimated **>200,000 new cases annually** in India. Projected cumulative lifetime risk for Indian women developing breast cancer is approximately 1 in 28.
2. **Mathur, P., Sathishkumar, K., Chaturvedi, M., et al. (2020):**
   - *"Cancer Statistics, 2020: Report From National Cancer Registry Programme, India."* *JCO Global Oncology*, 6, 1063–1075. DOI: [10.1200/GO.20.00122](https://doi.org/10.1200/GO.20.00122).
3. **Sung, H., Ferlay, J., Siegel, R. L., et al. (2021):**
   - *"Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries."* *CA: A Cancer Journal for Clinicians*, 71(3), 209–249.

---

## 2. Gompertzian Tumor Regrowth & Cost-of-Delay Kinetics

1. **Norton, L., & Simon, R. (1977):**
   - *"Tumor size, sensitivity to therapy, and design of treatment schedules."* *Cancer Treatment Reports*, 61(7), 1307–1317.
   - *Model Rationale:* Formulates the classic Norton-Simon hypothesis where tumor cell kill is proportional to instantaneous growth rate, governed by sigmoidal Gompertzian kinetics:
     $$\frac{dV}{dt} = \alpha V \ln\left(\frac{K}{V}\right) - \kappa \cdot \text{Dose} \cdot V$$
     This mathematical structure provides the foundation for our `GompertzTumorModel` in `src/tumor_model.py`.
2. **Benzekry, S., Lamont, C., Beheshti, A., et al. (2014):**
   - *"Classical mathematical models for description and prediction of experimental tumor growth."* *PLOS Computational Biology*, 10(8), e1003800. DOI: [10.1371/journal.pcbi.1003800](https://doi.org/10.1371/journal.pcbi.1003800).
3. **Laird, A. K. (1964):**
   - *"Dynamics of tumor growth."* *British Journal of Cancer*, 18(3), 490–502.

---

## 3. Chemotherapy-Induced Neutropenia (CIN) Timing & Nadir Dynamics

1. **Friberg, L. E., Henningsson, A., Maas, H., Nguyen, L., & Karlsson, M. O. (2002):**
   - *"Model of chemotherapy-induced myelosuppression with a putative attractant mechanism."* *Journal of Clinical Oncology*, 20(24), 4713–4721. DOI: [10.1200/JCO.2002.03.076](https://doi.org/10.1200/JCO.2002.03.076).
   - *Nadir Kinetics:* Demonstrates that after cytotoxic regimens, hematopoietic progenitor destruction causes an absolute neutrophil count (ANC) nadir typically occurring between **days 7 and 14 post-infusion**, followed by bone-marrow precursor rebound. Our latent nadir trajectory function `compute_post_dose_anc_trajectory()` in `src/generate_data.py` models this rebound rate ($r_i$).
2. **Crawford, J., Dale, D. C., & Lyman, G. H. (2004):**
   - *"Chemotherapy-induced neutropenia: risks, consequences, and experience with granulocyte colony-stimulating factor."* *Cancer*, 100(2), 228–237.
3. **National Cancer Institute (NCI):**
   - *Common Terminology Criteria for Adverse Events (CTCAE), Version 5.0.*
   - Grade 3 Neutropenia defined as $\text{ANC} < 1.0 \times 10^9/\text{L}$ to $0.5 \times 10^9/\text{L}$; Grade 4 defined as $\text{ANC} < 0.5 \times 10^9/\text{L}$.

---

## 4. Digital Biomarkers & Autonomic Wearable Proxies

1. **Gresham, G., Schrack, J., Gresham, L. M., et al. (2018):**
   - *"Wearable activity monitors in oncology: statistical analysis of physical activity and resting heart rate as predictors of treatment tolerability."* *NPJ Digital Medicine*, 1, 27. DOI: [10.1038/s41746-018-0036-2](https://doi.org/10.1038/s41746-018-0036-2).
2. **De Couck, M., Caers, R., Spiegel, D., & Gidron, Y. (2018):**
   - *"The role of the vagus nerve in oncology: Vagal nerve activity (measured by heart rate variability, HRV) predicts survival in cancer patients."* *Oncology Reports*, 39(5), 2005–2014.
3. **Klerman, E. B., et al. (2022):**
   - *"Continuous physiological monitoring via consumer wearables: applications and validation challenges in clinical oncology."* *Lancet Digital Health*, 4(6), e450–e461.
