# Input Data Format Specification

This document describes the required input data formats for the DIABIMMUNE Microbiome Modelling project. All data files should be placed in the `data/` directory.

## Overview

The project requires several interconnected data files to link sequencing data with subject metadata:

1. **Sequencing Run Tables** - Link SRA runs to subjects
2. **Sample Metadata** - Subject and sample information
3. **MicrobeAtlas Mappings** - Link SRA runs to sample IDs (SRS)
4. **BIOM File** - OTU abundance data
5. **ProkBERT Embeddings** - Pre-computed OTU embeddings
6. **Metadata Tables** - Phenotype/outcome data (milk, HLA, etc.)
7. **Model Checkpoint** - Pre-trained transformer weights

---

## 1. Sequencing Run Tables

### Files
- `SraRunTable_wgs.csv` - Whole genome sequencing runs
- `SraRunTable_extra.csv` - Additional sequencing runs

### Format
CSV file with standard SRA metadata fields.

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `Run` | SRA run accession (required) | `SRR4305031` |
| `Library Name` | Unique library identifier used to link to samples | `G69146` |
| `host_subject_id` | Subject identifier (optional, fallback when Library Name not found) | `P018832` |

### Example
```csv
Run,Assay Type,Library Name,host_subject_id,Instrument,Platform
SRR4305031,WGS,G69146,P018832,Illumina HiSeq 2500,ILLUMINA
SRR4305032,WGS,G69147,P017743,Illumina HiSeq 2500,ILLUMINA
```

### Notes
- The `Library Name` field is used as the primary key to link runs to samples via `gid_wgs` or `gid_16s` fields in the samples table
- Other columns from standard SRA format are optional but preserved

---

## 2. Sample Metadata Table

### File
- `samples.csv`

### Format
CSV file linking samples to subjects with age information.

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `subjectID` | Unique subject identifier | `E000823` |
| `country` | Country code | `FIN`, `EST`, `RUS` |
| `sampleID` | Unique sample identifier | `3000150` |
| `age_at_collection` | Age in days at sample collection | `56`, `259`, `760` |
| `cohort` | Cohort identifier | `abx`, `T1D` |
| `gid_wgs` | WGS library identifier (links to Library Name in run tables) | `G74397` |
| `gid_16s` | 16S library identifier (links to Library Name in run tables) | `G74397` |

### Example
```csv
subjectID,country,sampleID,age_at_collection,cohort,gid_wgs,gid_16s
E000823,FIN,3000150,56,abx,,G74397
E000823,FIN,3102720,259,abx,,G69867
E000823,FIN,3114328,760,abx,,G74423
```

### Notes
- Each row represents one sample from one subject
- Multiple samples can belong to the same subject (longitudinal data)
- At least one of `gid_wgs` or `gid_16s` should be present
- `age_at_collection` must be numeric (days)
- The file may contain UTF-8 BOM which is automatically handled

---

## 3. MicrobeAtlas Sample Mappings

### File
- `microbeatlas_samples.tsv`

### Format
Tab-separated values (TSV) file.

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `#sid` | Sample identifier (SRS accession) | `SRS1719087` |
| `rids` | Comma or semicolon-separated list of SRA run IDs | `SRR4305031` |

### Example
```tsv
#sid	name	rids	projects
SRS1719087	SRP090628...	SRR4305031	[{"pid": "SRP090628"}]
SRS1719088	SRP090628...	SRR4305032	[{"pid": "SRP090628"}]
SRS1719089	SRP090628...	SRR4305033	[{"pid": "SRP090628"}]
```

### Notes
- The `#sid` column provides the canonical sample ID (SRS) used throughout the pipeline
- Multiple SRA runs can be linked to one sample via the `rids` field
- Other columns (name, keywords, taxa_stats, etc.) are informational and not required
- This file establishes the connection: `SRA Run → SRS → Subject`

---

## 4. BIOM File (OTU Abundance Data)

### File
- `samples-otus.97.metag.minfilter.minCov90.noMulticell.rod2025companion.biom`

### Format
HDF5-based BIOM format (version 2.1)

### Structure
The BIOM file must be in HDF5 format with the following structure:

```
/observation/ids       - Array of OTU identifiers (string)
/sample/ids            - Array of sample identifiers (string, format: "prefix.SRS123456")
/sample/matrix/data    - Sparse matrix data (abundance counts)
/sample/matrix/indices - Row indices for sparse matrix
/sample/matrix/indptr  - Index pointers for sparse matrix
```

### Sample ID Format
Sample IDs in the BIOM file should follow this pattern:
```
<prefix>.<SRS_ID>
```

Example: `ABC123.SRS1719087`

The code extracts the SRS ID using: `sample_entry.split('.')[-1]`

### OTU Identifiers
OTU IDs must match the keys in the ProkBERT embeddings file (see below).

Example OTU IDs:
- `90_217;96_789;97_918`
- `90_21;96_89;97_94`
- `90_61;96_130;97_140`

### Notes
- The BIOM file uses column-sparse format (sample-major)
- OTUs (observations) are features, samples are columns
- Only presence/absence is used (abundance values are not required)
- The file is read using `h5py` Python library

---

## 5. ProkBERT Embeddings

### File
- `prokbert_embeddings.h5`

### Format
HDF5 file containing pre-computed embeddings for each OTU.

### Structure
```
/embeddings/<OTU_ID>   - NumPy array of shape (384,) for each OTU
```

### Example Structure
```python
prokbert_embeddings.h5
├── embeddings/
│   ├── 90_217;96_789;97_918  → array of shape (384,)
│   ├── 90_21;96_89;97_94     → array of shape (384,)
│   ├── 90_61;96_130;97_140   → array of shape (384,)
│   └── ...
```

### Requirements
- Each OTU embedding must be a 1D array of length 384 (float32 or float64)
- OTU IDs must match those in the BIOM file
- Missing OTUs are counted but don't cause failures

### Access Pattern
```python
import h5py
with h5py.File('prokbert_embeddings.h5') as f:
    otu_embedding = f['embeddings']['90_217;96_789;97_918'][()]
    # Returns numpy array of shape (384,)
```

---

## 6. Metadata Tables (Phenotypes/Outcomes)

These tables contain subject-level metadata for prediction tasks.

### 6.1 Milk Feeding Data

#### File
- `milk.csv`

#### Format
CSV file with milk feeding information per subject.

#### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `subjectID` | Subject identifier (matches samples.csv) | `E000823` |
| `milk_first_three_days` | Type of milk in first 3 days | `mothers_breast_milk`, `multiple_types_or_not_reported` |
| `bf_length` (optional) | Breastfeeding length in days | `397` |
| `bf_length_exclusive` (optional) | Exclusive breastfeeding length | `150` |

#### Example
```csv
subjectID,bf_length,bf_length_exclusive,milk_first_three_days
E000823,397,150,mothers_breast_milk
E001463,,120,mothers_breast_milk
E002338,398,,multiple_types_or_not_reported
```

### 6.2 HLA Risk and Birth Data

#### File
- `pregnancy_birth.csv`

#### Format
CSV file with pregnancy, birth, and HLA risk information.

#### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `subjectID` | Subject identifier | `E003188` |
| `HLA_risk_class` | HLA-conferred disease risk class | `2`, `3`, `4` |
| `gender` (optional) | Subject gender | `Male`, `Female` |
| `csection` (optional) | C-section delivery | `True`, `False` |
| `gestational_diabetes` (optional) | Gestational diabetes status | `True`, `False` |
| `location` (optional) | Location type | `urban`, `rural` |

#### Example
```csv
subjectID,mom_age_at_birth,gestational_diabetes,csection,gender,HLA_risk_class,location
E003188,30.85,True,False,Female,3,urban
E004898,30.12,False,False,Male,2,urban
E005786,26.79,False,False,Female,3,urban
```

### Notes
- Empty cells are permitted for optional fields
- Subject IDs must match those in `samples.csv`
- Boolean values can be `True`/`False` or left empty
- The file may contain UTF-8 BOM which is automatically handled

---

## 7. Model Checkpoint

### File
- `checkpoint_epoch_0_final_epoch3_conf00.pt`

### Format
PyTorch checkpoint file (`.pt`)

### Required Contents
The checkpoint must be a dictionary containing at minimum:

```python
{
    'model_state_dict': OrderedDict(...)  # Model weights
}
```

### Model Architecture
The checkpoint should match the `MicrobiomeTransformer` architecture with these hyperparameters:

```python
input_dim_type1 = 384      # OTU embeddings (ProkBERT)
input_dim_type2 = 1536     # Text embeddings (not used in current pipeline)
d_model = 100              # Transformer hidden size
nhead = 5                  # Number of attention heads
num_layers = 5             # Number of transformer layers
dim_feedforward = 400      # FFN dimension
dropout = 0.1              # Dropout rate
```

### Loading
```python
checkpoint = torch.load('checkpoint_epoch_0_final_epoch3_conf00.pt', 
                       map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

---

## Data Flow Diagram

```
SRA Run Tables (CSV)
    ├── Run ID → Library Name
    └── Run ID → Subject ID (fallback)
            ↓
MicrobeAtlas (TSV)
    └── Run ID → SRS ID
            ↓
BIOM File (HDF5)
    └── SRS ID → [OTU IDs]
            ↓
ProkBERT Embeddings (HDF5)
    └── OTU ID → Embedding Vector (384-dim)
            ↓
Transformer Model
    └── Sample → Embedding Vector (d_model-dim)
            ↓
Samples Table (CSV)
    ├── Library Name → Sample ID
    ├── Sample ID → Subject ID
    └── Sample ID → Age
            ↓
Metadata Tables (CSV)
    └── Subject ID → Labels (milk type, HLA risk, etc.)
```

---

## Validation Checklist

Before running the pipeline, ensure:

- [ ] All required CSV files have proper headers
- [ ] UTF-8 encoding is used (BOM is okay)
- [ ] Sample IDs are consistent across files
- [ ] OTU IDs in BIOM match those in ProkBERT embeddings
- [ ] Age values are numeric (days)
- [ ] BIOM file is in HDF5 format (BIOM 2.1)
- [ ] ProkBERT embeddings are 384-dimensional
- [ ] Model checkpoint matches expected architecture
- [ ] Subject IDs in metadata tables match those in samples.csv

---

## Common Issues and Troubleshooting

### Issue: Missing subject/sample mappings
**Cause:** `Library Name` field doesn't match `gid_wgs` or `gid_16s` in samples table.

**Solution:** Verify Library Name values are correctly specified and match between run tables and samples.csv.

### Issue: Missing OTU embeddings
**Cause:** OTU IDs in BIOM file don't match keys in ProkBERT embeddings.

**Solution:** Regenerate ProkBERT embeddings for all OTUs in the BIOM file, or update BIOM file to only include OTUs with embeddings.

### Issue: Sample ID extraction fails
**Cause:** Sample IDs in BIOM file don't follow expected format.

**Solution:** Ensure BIOM sample IDs end with SRS identifier (e.g., `prefix.SRS1719087`).

### Issue: Age data missing
**Cause:** `age_at_collection` field is empty or non-numeric.

**Solution:** Populate age field with numeric values (days). Records without age will be skipped in age-binned analyses.

### Issue: Checkpoint loading fails
**Cause:** Model architecture mismatch or corrupted checkpoint.

**Solution:** Verify checkpoint was saved from compatible model architecture. Check hyperparameters in `utils.py` match those used during training.

---

## Creating New Metadata Tables

To add new prediction tasks, create additional metadata CSV files following this template:

```csv
subjectID,outcome_variable,covariate1,covariate2
E000001,positive,value1,value2
E000002,negative,value3,value4
```

**Requirements:**
1. First column must be `subjectID` matching samples.csv
2. At least one outcome/label column for prediction
3. UTF-8 encoding
4. Empty cells allowed for missing data
5. Follow existing naming conventions

Then update prediction scripts following the pattern in `predict_milk.py` or `predict_hla.py`.

---

## File Size Expectations

Typical file sizes for reference:

| File | Typical Size |
|------|--------------|
| SRA run tables (CSV) | 1-10 MB |
| samples.csv | 100 KB - 1 MB |
| microbeatlas_samples.tsv | 1-50 MB |
| BIOM file | 10-500 MB |
| ProkBERT embeddings (HDF5) | 100 MB - 5 GB |
| Model checkpoint | 1-50 MB |
| Metadata tables | 10-100 KB |

---

## Further Resources

- [BIOM Format Documentation](http://biom-format.org/)
- [HDF5 Python Documentation](https://docs.h5py.org/)
- [MicrobeAtlas](https://www.microbeatlas.org/)
- [DIABIMMUNE Study](http://www.diabimmune.org/)
- [ProkBERT Paper](https://www.nature.com/articles/s41587-023-01953-y)

---

## Contact & Support

For questions about data format or issues with the pipeline, refer to the main README.md or open an issue on the repository.

