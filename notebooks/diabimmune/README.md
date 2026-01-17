# DIABIMMUNE Food Allergy Baseline Notebook

## Overview

This folder contains a **self-contained, transplantable** baseline analysis for predicting food allergy development from infant gut microbiome embeddings using the DIABIMMUNE three-country cohort.

**Primary notebook:** `01_food_allergy_baseline.ipynb`

---

## Quick Start

```bash
# Required inputs (relative to repo root):
data_preprocessing/datasets_preprocessing_scripts/diabimmune/preprocessing_scripts/preprocessed_diabimmune_longitudinal/Month_*.csv   # Metadata with labels
data_preprocessing/microbiome_embeddings/microbiome_embeddings_100d.h5  # 100-dim embeddings (see dataset_manifest.json for provenance)

# Run the notebook:
cd notebooks/diabimmune
uv run jupyter nbconvert --execute 01_food_allergy_baseline.ipynb --to html
```

**Outputs** (committed for reproducibility; written to `notebooks/diabimmune/results/`):
- `notebooks/diabimmune/results/cv_metrics.csv` — per-fold AUROC, AUPRC, F1 by horizon
- `notebooks/diabimmune/results/cv_summary.csv` — mean ± std across folds
- `notebooks/diabimmune/results/loco_metrics.csv` — Leave-One-Country-Out results

---

## Scientific Background

### The DIABIMMUNE Cohort

The [DIABIMMUNE study](https://diabimmune.broadinstitute.org/) investigates the hygiene hypothesis across three countries with different autoimmune disease rates:

| Country | Setting | T1D/Allergy Rate |
|---------|---------|------------------|
| Finland | Western, developed | Highest globally |
| Estonia | Rapidly modernizing | Increasing |
| Russia (Karelia) | Less developed | Relatively rare |

**Cohort details:**
- 222 infants enrolled (74 per country)
- Monthly stool samples from birth to age 3
- 785 samples with WGS sequencing (used for embeddings)
- Selection criteria: Similar HLA risk, matched gender

### The Prediction Task

**Goal:** Predict eventual food allergy (milk, egg, peanut) from early-life gut microbiome composition.

**Why this matters:** The gut microbiome in early life is implicated in immune system development. If microbiome patterns can predict allergy before clinical onset, this could enable early intervention.

---

## Methodology

### 1. Cumulative Horizons (Not Disjoint Bins)

We analyze **cumulative time windows**, not month-by-month or disjoint bins:

| Horizon | Data Used | Claim Strength |
|---------|-----------|----------------|
| ≤3mo | Samples with `month <= 3` (Month_1–Month_3) | Exploratory (high variance; strongest predictive framing) |
| ≤6mo | Samples with `month <= 6` (Month_1–Month_6) | Moderate prediction |
| ≤12mo | Samples with `month <= 12` (Month_1–Month_12) | Mixed prediction/association |
| All | All 785 samples | Association only |

**Why cumulative horizons?**
- The question is "how early can we predict?" not "what happens in month 7-12 specifically"
- Disjoint bins (e.g., months 7-12 only) answer a different question and have severe class imbalance
- Month-by-month analysis is impossible: most months have <20 samples with extreme class/country imbalance

**Why cap at ≤12 months?**
- Food allergies often manifest in the first year of life
- After onset, the microbiome may reflect the disease state (e.g., dietary restrictions, inflammation)
- Later samples risk **reverse causation**: we'd be detecting the effect of allergy, not predicting it

### 2. Subject-Level Aggregation

Each infant has multiple stool samples over time. We aggregate to **one embedding per infant per horizon**:

```python
# For each infant in the horizon, average their sample embeddings
subject_embedding = mean(sample_embeddings for that infant up to month m)
```

**Why aggregate?**
- The label (allergic vs not) is at the infant level, not sample level
- Training on raw samples would let subjects with more samples dominate
- CV must be at the subject level anyway (see below)

### 3. Leakage-Safe Cross-Validation

**Critical requirement:** The model must be evaluated at the **infant (patient) level** with no subject leakage.

This notebook first aggregates to **one row per patient per horizon**, then runs CV with `groups=patient_id` for safety and future-proofing.

Note: after aggregation, `groups=patient_id` is redundant (one row per patient), but keeping it is a defensive guardrail against future refactors that might accidentally re-introduce leakage.

```python
from sklearn.model_selection import StratifiedGroupKFold

subj = build_subject_table(df_h)  # one row per patient
X = np.stack(subj["embedding"].to_list())
y = subj["label"].to_numpy()
groups = subj["patient_id"].to_numpy()

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in cv.split(X, y, groups=groups):
    # Each patient is in train OR test, never both
```

**What would go wrong without this?**
- If infant A has samples in both train and test, the model "sees" infant A during training
- Test performance would be inflated (memorization, not generalization)
- This is **subject leakage** — a fatal flaw in longitudinal study design

### 4. Fixed Hyperparameters (No Tuning)

The baseline model is a scikit-learn `Pipeline`:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(...)),
])
```

The baseline uses fixed hyperparameters (no tuning):

```python
LogisticRegression(
    C=1.0,                    # Default regularization
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs',
    max_iter=2000,
    random_state=42
)
```

**Why no hyperparameter tuning?**
- Fixed hyperparameters = no selection bias = outer CV alone gives unbiased estimates
- This is a **baseline** — we want to know if ANY signal exists before optimizing
- Tuning adds complexity and risks overfitting on small datasets

**Why NOT the "two-phase" approach?**

An alternative was proposed:
> Phase 1: GridSearchCV on entire dataset → find best hyperparameters
> Phase 2: Evaluate with those hyperparameters using different CV splits

**This is methodologically incorrect.** Phase 1 uses ALL data (including eventual test folds) to select hyperparameters. Phase 2's "unseen" test data was already seen during hyperparameter selection. This produces **optimistically biased** estimates.

**Correct approach for tuning (if needed later):** Nested CV
- Outer loop: 5 folds for final evaluation
- Inner loop: GridSearchCV on train fold only
- Test fold is truly unseen during all model selection

### 5. Leave-One-Country-Out (LOCO) Analysis

Allergy prevalence differs dramatically by country:

| Country | Patients | Allergic | Rate |
|---------|----------|----------|------|
| Finland | 71 | 35 | 49% |
| Estonia | 71 | 27 | 38% |
| Russia | 70 | 8 | 11% |

A model could learn "is this from Russia?" instead of "does this microbiome pattern predict allergy?"

**LOCO tests generalization:**
- Train on 2 countries, test on the held-out country
- If LOCO AUROC ≈ CV AUROC: model learns transferable patterns
- If LOCO AUROC << CV AUROC: model may exploit country-specific batch effects

**Edge cases (handled in code):**
- AUROC is undefined if the held-out test set has only one class → those rows are skipped.
- Russia at ≤3mo is explicitly skipped (too few patients and no positives).

### 6. Metrics

- **Primary**: AUROC (subject-level)
- **Secondary**: AUPRC and F1 (F1 uses a fixed 0.5 threshold on predicted probability)
- **LOCO outputs**: AUROC and AUPRC (no F1)

---

## Dataset Details

### Track A: HF Embeddings (This Notebook)

| Metric | Value |
|--------|-------|
| Total samples | 785 |
| Unique patients | 212 |
| Embedding dimension | 100 |
| Allergic patients | 68 (32%) |
| Healthy patients | 144 (68%) |

### Horizon Breakdown (Patient-Level)

| Horizon | Samples | Patients | Healthy | Allergic | Notes |
|---------|---------|----------|---------|----------|-------|
| ≤3 months | 45 | 44 | 23 | 21 | Only 3 RUS patients |
| ≤6 months | 110 | 92 | 51 | 41 | Viable balance |
| ≤12 months | 307 | 160 | 101 | 59 | Viable balance |
| All | 785 | 212 | 144 | 68 | Association baseline |

### Country Distribution by Horizon (Patients)

| Horizon | FIN | EST | RUS |
|---------|-----|-----|-----|
| ≤3 months | 24 | 17 | 3 |
| ≤6 months | 39 | 31 | 22 |
| ≤12 months | 56 | 48 | 56 |
| All | 71 | 71 | 70 |

---

## Results Interpretation

### AUROC Guide

| AUROC | Interpretation |
|-------|----------------|
| 0.50 | Random chance (no signal) |
| 0.55–0.65 | Weak signal |
| 0.65–0.75 | Moderate signal |
| 0.75+ | Strong signal |

### LOCO Interpretation

| Pattern | Meaning |
|---------|---------|
| LOCO ≈ CV | Model learns transferable microbiome patterns |
| LOCO << CV | Model may exploit country-specific effects |
| LOCO < 0.50 | Model fails to generalize to held-out country |

### What to Look For

1. **Does AUROC decrease as horizon decreases?** Expected — less data = noisier estimates
2. **Is ≤3mo AUROC above chance?** Key question for "early prediction" claim
3. **Do LOCO results hold up?** Critical for ruling out country confounding

---

## Known Limitations

### 1. Onset Timing is Unknown

The label is an **endpoint outcome** (eventual food allergy status), not a diagnosis at sample collection time. We don't know when each infant first developed symptoms.

**Implication:** Any horizon may include post-onset samples for some infants. Even ≤3 months is not guaranteed "pre-disease."

### 2. Milk Allergy Can Manifest Very Early

Infants can be exposed to cow's milk protein via formula or breast milk in the first weeks of life. Some develop milk allergy before age 3 months.

**Implication:** The ≤3 month horizon doesn't guarantee "prediction before exposure." Claims should be framed carefully.

### 3. Country Confounding

Russia has dramatically lower allergy rates (11%) than Finland (49%). This could be:
- True biological signal (hygiene hypothesis)
- Batch effects in sample processing
- Differences in diagnosis criteria

**Mitigation:** LOCO analysis helps assess generalization.

### 4. Russia Underrepresented at Early Horizons

At ≤3 months, only 3 Russian patients have samples. LOCO results for Russia at this horizon are not meaningful.

### 5. Claim Strength is a Gradient

| Horizon | Framing |
|---------|---------|
| ≤3 months | Exploratory (high variance; strongest "prediction" framing with caveats) |
| ≤6 months | Moderate prediction |
| ≤12 months | Mixed prediction/association |
| All | Association only |

There is no clean boundary between "prediction" and "association" without onset timing data.

### 6. Statistical Uncertainty at ≤3mo (Exploratory)

At `≤3mo` there are only **44 patients**. With 5-fold CV, each test fold has roughly **~8–9 patients**, so fold-wise AUROC can vary substantially. Treat `≤3mo` results as exploratory unless you add statistical inference (e.g., bootstrap confidence intervals or permutation testing) to show performance is distinguishable from chance.

Multiple horizons/metrics are reported in this baseline; interpret horizon comparisons as exploratory (no multiple-comparisons correction is applied here).

---

## Reproducibility

### Requirements

```
numpy
pandas
h5py
scikit-learn
jupyter
```

### Seed Policy

All randomness uses `RANDOM_SEED = 42`:
- `np.random.seed(42)`
- `StratifiedGroupKFold(..., random_state=42)`
- `LogisticRegression(..., random_state=42)`

### Verification

The notebook includes assertions that fail fast if invariants break:
- Expected sample count (785)
- Expected patient count (212)
- No duplicate sample IDs
- Labels consistent within each patient
- Expected horizon counts

---

## File Structure

```
gut_microbiome_project/
├── notebooks/diabimmune/
│   ├── 01_food_allergy_baseline.ipynb   # Main notebook
│   ├── 01_food_allergy_baseline.html    # Rendered output
│   ├── README.md                         # This file
│   └── results/                          # Committed outputs
│       ├── cv_metrics.csv
│       ├── cv_summary.csv
│       └── loco_metrics.csv
│
└── data_preprocessing/
    ├── microbiome_embeddings/
    │   ├── microbiome_embeddings_100d.h5  # Consolidated embeddings (785 x 100)
    │   └── dataset_manifest.json          # Provenance documentation
    │
    └── datasets_preprocessing_scripts/diabimmune/preprocessing_scripts/
        └── preprocessed_diabimmune_longitudinal/
            └── Month_*.csv                # Metadata with labels
```

---

## References

### Primary Data
- [DIABIMMUNE Project](https://diabimmune.broadinstitute.org/)
- [BioProject PRJNA290380](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA290380)

### Key Paper
- Vatanen et al. (2016) "Variation in Microbiome LPS Immunogenicity Contributes to Autoimmunity in Humans" — *Cell*

### Embedding Model
- Track A uses **precomputed** 100‑dim embeddings from HuggingFace: `hugging-science/AI4FA-Diabimmune` (keys are SRS IDs). Provenance and hashes are documented in `data_preprocessing/microbiome_embeddings/dataset_manifest.json`.
