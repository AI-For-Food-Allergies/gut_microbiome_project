## Diabimmune Microbiome Modelling Playground

This repo explores linking DIABIMMUNE cohort metadata to MicrobeAtlas samples, extracting ProkBERT OTU embeddings, and scoring simple downstream models. The end goal is to rapidly prototype prediction tasks (e.g. milk-feeding type, HLA risk) using a shared embedding layer inspired by *[Micro-Modelling the Microbiome](https://the-puzzler.github.io/post.html?p=posts%2Fmicro-modelling%2Fmicro-modelling.html)*.

### Contents

- `config.yaml` – centralized YAML configuration for all run parameters (data paths, model architecture, cross-validation settings). See `CONFIG_GUIDE.md` for details.
- `utils.py` – lightweight helpers to
  - load configuration from YAML
  - read SRA run tables, MicrobeAtlas, and DIABIMMUNE metadata  
  - align SRS ↔ sample ↔ subject IDs  
  - load the transformer checkpoint (`checkpoint_epoch_0_final_epoch3_conf00.pt`)  
  - pull OTU embeddings from `prokbert_embeddings.h5` and generate per-sample vectors.
- `prepping_code.py` – manual scratchpad for data inspection.
- `predict_milk.py` – builds balanced cohorts, bins ages, trains 5× stratified CV logistic models, and visualises ROC AUC and confusion matrices for early milk type prediction. Output plots:
  - `milk_cm.png`
- `predict_hla.py` – similar pipeline for HLA risk (classes 2 vs 3) with downsampling and per-bin confusion matrices. Output plot:
  - `fla_cm.png`
- `data/` – raw tables (samples, HLA, milk, run tables, DIABIMMUNE covariates, pretrained embeddings).
- Public data package: [Figshare dataset](https://figshare.com/articles/dataset/Model_and_Data_for_diabimmune_example/30429055?file=58993825) (includes the `data/` directory contents referenced above).

### Setup

1. Python ≥ 3.10 with `uv` (project uses `pyproject.toml` + `uv.lock`) or vanilla `pip`.
   ```bash
   uv sync
   # or
   pip install -e .
   ```
2. Ensure the following files are present under `data/`:
   - `SraRunTable_wgs.csv`, `SraRunTable_extra.csv`
   - `samples.csv`, `pregnancy_birth.csv`, `milk.csv`, and other cohort sheets
   - `microbeatlas_samples.tsv`, `samples-otus...biom`
   - `prokbert_embeddings.h5`
   - `checkpoint_epoch_0_final_epoch3_conf00.pt`
3. (Optional) Customize `config.yaml` to modify:
   - Data file paths
   - Cross-validation settings (k-folds, random seed, logistic regression hyperparameters)
   
   Note: Model architecture parameters are hardcoded to match the pre-trained checkpoint.
   See `CONFIG_GUIDE.md` for detailed configuration options.

### Workflow

1. **Embedding prep** (`predict_milk.py`/`predict_hla.py`, first cells):
   - call `load_run_data()` → SRA ↔ MicrobeAtlas ↔ DIABIMMUNE mappings
   - call `collect_micro_to_otus()` → SRS → OTU IDs
   - call `load_microbiome_model()` → instantiate MicrobiomeTransformer with article hyperparameters
   - call `build_sample_embeddings()` → runs model over ProkBERT vectors and averages final hidden states for each SRS.
2. **Task-specific scripts**:
   - link sample embeddings back to subject-level metadata (milk/HLA tables)
   - compute age bins (tertiles) for reporting
   - balance cohorts (downsample majority class + `class_weight='balanced'`)
   - run 5-fold Stratified CV logistic models
   - produce evaluation plots (ROC AUC distributions, confusion matrices).

### Results

![Milk confusion matrices](milk_cm.png)

![HLA confusion matrices](fla_cm.png)

### References

- Transformer backbone: [Micro-Modelling the Microbiome](https://the-puzzler.github.io/post.html?p=posts%2Fmicro-modelling%2Fmicro-modelling.html)
- DIABIMMUNE cohort metadata (public release)
- MicrobeAtlas sample catalogue

### Notes

- The scripts are intentionally notebook-like: every cell is linear and verbose to keep context obvious.
- Quantile-based age bins are reused across tasks; change them in-script if you need clinically defined ranges.
- To add a new task, mirror the structure in `predict_milk.py`: load embeddings, merge metadata, balance classes, evaluate.
