# Configuration Guide

This project now uses a YAML-based configuration system for improved parameter management and user experience.

## Configuration File

Run parameters are defined in `config.yaml`. This file contains:

### Data Paths
- `data.wgs_run_table`: Path to WGS SRA run table
- `data.extra_run_table`: Path to extra SRA run table
- `data.microbeatlas_samples`: Path to MicrobeAtlas samples TSV
- `data.samples_table`: Path to samples CSV
- `data.biom_path`: Path to BIOM file with OTU data
- `data.checkpoint_path`: Path to pre-trained model checkpoint
- `data.prokbert_path`: Path to ProkBERT embeddings HDF5 file

### Cross-Validation Parameters
- `cross_validation.k_folds`: Number of CV folds (default: 5)
- `cross_validation.random_state`: Random seed for reproducibility (default: 42)
- `cross_validation.max_iter`: Maximum iterations for logistic regression (default: 1000)
- `cross_validation.class_weight`: Class weighting strategy (default: "balanced")
- `cross_validation.solver`: Solver for logistic regression (default: "lbfgs")

### Model Architecture (Not Configurable)

Model architecture parameters are **hardcoded** in `utils.py` and match the pre-trained checkpoint:
- `D_MODEL = 100`: Transformer model dimension
- `NHEAD = 5`: Number of attention heads
- `NUM_LAYERS = 5`: Number of transformer layers
- `DROPOUT = 0.1`: Dropout rate
- `DIM_FF = 400`: Feedforward dimension
- `OTU_EMB = 384`: OTU embedding dimension
- `TXT_EMB = 1536`: Text embedding dimension

These parameters cannot be changed via the config file as they must match the pre-trained model checkpoint.

## Usage

### Basic Usage

All utility functions now automatically load the configuration from `config.yaml`:

```python
from utils import (
    load_run_data,
    collect_micro_to_otus,
    load_microbiome_model,
    get_config
)

# Load configuration
config = get_config()

# Functions use config automatically
run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
model, device = load_microbiome_model()
```

### Custom Configuration

You can also pass custom configuration or override specific parameters:

```python
from utils import load_config, load_microbiome_model

# Load custom config file
custom_config = load_config("path/to/custom_config.yaml")

# Use custom config
model, device = load_microbiome_model(config=custom_config)
```

### Override Individual Parameters

```python
from utils import load_run_data

# Override specific data path
run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data(
    wgs_path="path/to/custom/wgs_table.csv"
)
```

## Prediction Scripts

The prediction scripts (`predict_milk.py` and `predict_hla.py`) now use the configuration for:
- Cross-validation parameters (k-folds, random_state)
- Logistic regression hyperparameters (max_iter, solver, class_weight)
- Data balancing (random_state for reproducibility)

Example from `predict_milk.py`:

```python
from utils import get_config

config = get_config()

# Cross-validation with config parameters
skf = StratifiedKFold(
    n_splits=config['cross_validation']['k_folds'], 
    shuffle=True, 
    random_state=config['cross_validation']['random_state']
)

# Logistic regression with config parameters
clf = LogisticRegression(
    max_iter=config['cross_validation']['max_iter'],
    solver=config['cross_validation']['solver'],
    class_weight=config['cross_validation']['class_weight']
)
```


## Example: Creating Custom Configurations

Create a new config file for experimentation:

```yaml
# experiment_config.yaml
data:
  # Use same data paths
  wgs_run_table: "data/SraRunTable_wgs.csv"
  extra_run_table: "data/SraRunTable_extra.csv"
  microbeatlas_samples: "data/microbeatlas_samples.tsv"
  samples_table: "data/samples.csv"
  biom_path: "data/samples-otus.97.metag.minfilter.minCov90.noMulticell.rod2025companion.biom"
  checkpoint_path: "data/checkpoint_epoch_0_final_epoch3_conf00.pt"
  prokbert_path: "data/prokbert_embeddings.h5"

cross_validation:
  # Try different CV strategy
  k_folds: 10
  random_state: 123
  max_iter: 2000
  class_weight: "balanced"
  solver: "saga"
```

Then use it:

```python
from utils import load_config

# Load experimental config
exp_config = load_config("experiment_config.yaml")

# Run your analysis with the new parameters...
```

