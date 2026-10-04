# Multimodal Deep Learning for Emergency Department Triage Risk Prediction

**H9MLAI CA2** — *National College of Ireland*

**Author:** Prasanth Oddepally (x25205595)

---

## Overview

This project investigates whether adding unstructured clinical text (chief complaints and disease history) to structured vital signs improves the accuracy of admission risk prediction in Emergency Department (ED) triage. It benchmarks four model architectures — Logistic Regression, XGBoost, Deep Neural Networks (DNN), and TabNet — across three progressively richer feature sets.

---

## Research Questions

| RQ | Question | Approach |
|:--|:---------|:---------|
| **RQ1** | Does adding chief complaint and disease history to vital signs improve predictive performance? | Compare AUROC across three feature views (vitals → +complaint → +history) for all four models. Bootstrap significance test for best model. |
| **RQ2** | How well does the model's admission prediction agree with the nurse-assigned ESI (Emergency Severity Index)? | Cohen's Kappa between model prediction (admit vs. discharge) and nurse ESI ≤ 2 (high urgency). Benchmark: human inter-rater κ ≈ 0.786. |
| **RQ3** | Is model performance fair across age and gender subgroups? | Subgroup AUROC analysis for age (<65 / 65+) and gender groups. |

---

## Dataset

- **Source:** Yale New Haven ED data, released by Hong, Haimovich & Taylor (PLoS ONE, 2018). [Publicly available on Kaggle](https://kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data).
- **Size:** ~109,000 patient encounters.
- **Target:** `admitted` (binary) — derived from disposition containing "Admit".
- **Feature groups:**

| Group | Description | # Features |
|:------|:------------|:-----------|
| **Vitals** | Heart rate, SBP, DBP, respiratory rate, O₂ saturation, temperature | 6 |
| **Demographics** | Age, gender | 2 |
| **Chief Complaint (CC)** | One-hot encoded presenting complaint flags | ~200 |
| **Disease History** | One-hot encoded comorbidity/history flags | ~754 |

### Three Feature Views for RQ1

1. **vitals** — vital signs + age + gender (8 features)
2. **+complaint** — vitals + chief complaint flags (208 features)
3. **+history** — vitals + chief complaint + full disease history (962 features — richest)

---

## Models

| Model | Type | Key Hyperparameters |
|:------|:-----|:--------------------|
| **Logistic Regression** | Linear baseline | class_weight='balanced', max_iter=1000 |
| **XGBoost** | Gradient-boosted trees | 200 trees, max_depth=5, lr=0.1, scale_pos_weight=imbalance |
| **DNN** | Feed-forward neural network | 3 layers (128→64→1), ReLU, BatchNorm, Dropout(0.3), Adam |
| **TabNet** | Attention-based tabular NN | 40 epochs, patience=8, batch_size=1024 |

All models use stratified 70/15/15 train/val/test split with class-weighting to handle admission imbalance.

---

## Results

### RQ1 — AUROC by Model and Feature Set

Adding chief complaint and disease history to vital signs dramatically improves predictive performance — AUROC jumps from ~0.76 to ~0.90 for all models. All four architectures performed near-identically at the richest feature view (0.898–0.903 AUROC), indicating that the choice of **information mattered far more than the choice of algorithm**.

| Model | vitals | +complaint | +history |
|:------|:------:|:----------:|:--------:|
| **LogReg** | 0.750 | 0.831 | 0.900 |
| **XGBoost** | 0.768 | 0.832 | 0.901 |
| **DNN** | 0.767 | 0.835 | 0.903 |
| **TabNet** | 0.763 | 0.833 | 0.898 |

> *Detailed results in [`outputs/results_summary.csv`](outputs/results_summary.csv) and [`outputs/rq1_auroc_by_model_inputs.csv`](outputs/rq1_auroc_by_model_inputs.csv).*

### RQ2 — Agreement with Nurse ESI

- **Model vs. Nurse ESI (≤2) Kappa:** κ = 0.368
- **Human inter-rater benchmark:** κ ≈ 0.786
- **Under-triage rate** (model predicts discharge but patient admitted): 6.2%
- **Over-triage rate** (model predicts admit but patient discharged): 10.9%

> *Full results in [`outputs/rq2_esi_agreement.csv`](outputs/rq2_esi_agreement.csv).*

### RQ3 — Fairness Across Subgroups

Subgroup AUROC analysis showed consistent performance across demographic groups, with no evidence of systematic bias:

| Subgroup | AUROC |
|:---------|:-----:|
| Age < 65 | 0.887 |
| Age ≥ 65 | 0.882 |
| Male | 0.902 |
| Female | 0.903 |

> *Full results in [`outputs/rq3_subgroup_auroc.csv`](outputs/rq3_subgroup_auroc.csv).*

### Latency Benchmark

Timings measured on the **+history** (962-feature) view.

| Model | Train Time (s) | Predict (ms per 1k patients) |
|:------|:--------------:|:----------------------------:|
| **LogReg**  | 7.71 | 3.58 |
| **XGBoost** | 27.78 | 4.62 |
| **DNN**     | 55.53 | 3.19 |
| **TabNet**  | 848.38 | 229.65|

> *See [`outputs/latency.csv`](outputs/latency.csv) for exact timings.*

---

## Project Structure

```
ML CA 2/
├── ED_Triage_REAL_final.ipynb   # Main analysis notebook
├── 5v_cleandf.csv               # Input dataset (not tracked in git)
├── README.md                    # This file
├── ML CA 2 REPORT.docx          # Full written report (Word)
├── ML_CA2_REPORT.pdf            # Full written report (PDF)
└── outputs/                     # Generated results and figures
    ├── results_summary.csv
    ├── rq1_auroc_by_model_inputs.csv
    ├── rq1.png
    ├── rq2_esi_agreement.csv
    ├── rq3_subgroup_auroc.csv
    ├── admission_counts.csv
    ├── admission_rate_by_esi.csv
    ├── esi_distribution.csv
    ├── vitals_summary.csv
    ├── vitals_missingness.csv
    ├── bootstrap_significance.csv
    ├── latency.csv
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    ├── classification_report.csv
    ├── missed_vs_caught_vitals.csv
    ├── Admission rate by ESI level.png
    ├── Admission rate.png
    ├── Nurse ESI level.png
    ├── RQ1 AUROC by model and feature set.png
    └── missed_vs_caught_vitals.csv
```

---

## Dependencies

| Package | Version |
|:--------|:--------|
| Python | ≥ 3.8 |
| pandas | — |
| numpy | — |
| scikit-learn | — |
| xgboost | — |
| torch | — |
| pytorch-tabnet | — |
| matplotlib | — |

Install with:

```bash
pip install pandas numpy scikit-learn xgboost torch pytorch-tabnet matplotlib
```

---

## How to Run

1. Ensure the dataset `5v_cleandf.csv` is in the project root directory.
2. Adjust `DATA_PATH` in the notebook if the dataset is located elsewhere.
3. Run all cells in `ED_Triage_REAL_final.ipynb` sequentially.
4. Results and figures are saved automatically to the `outputs/` directory.

> **Note:** The notebook loads the full 109k-row dataset. If you encounter RAM limitations, reduce `SAMPLE_FRAC` in the "Data Loading" cell (e.g., to `0.5` for a 50% sample).

## Key Findings

- **Adding chief complaint and disease history to vital signs dramatically improves admission prediction** — AUROC jumps from ~0.76 (vitals-only) to ~0.90 (+history) across all architectures.
- **All four architectures performed near-identically at the richest feature view (0.898–0.903 AUROC)**, indicating that the choice of information mattered far more than the choice of algorithm.
- **Model–nurse agreement was moderate (κ = 0.368)**, well below the human inter-rater benchmark (κ ≈ 0.786), reflecting that admission and urgency are related but distinct targets. Errors skewed toward over-triage (10.9%) rather than under-triage (6.2%), the safer failure direction.
- **Fairness** holds well across age and gender subgroups (AUROC 0.882–0.903), though further investigation on larger cohorts is warranted.
- **False negatives (missed admissions)** were younger (mean age 51.6 vs. 66.4) and of lower acuity (mean ESI 2.63 vs. 2.38) than true positives, with broadly similar vital signs — suggesting that incorporating clinical text is especially valuable for identifying at-risk younger patients with less extreme physiology.


