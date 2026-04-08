"""
Unified Data Loading Pipeline

This module provides a complete pipeline for loading microbiome data:
1. Sample CSV (SID, label) → Artifacts (DNA CSVs, Embeddings H5) → PyTorch DataLoader
2. Handles artifact generation, caching, and efficient data loading
3. Generates per sample embeddings from prokbert_embeddings
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import h5py
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import os
import subprocess
from utils.data_utils import load_config, split_dataset_df
from modules.model import MicrobiomeTransformer
from omegaconf import DictConfig, OmegaConf
from rich.console import Console


# Try to import transformers (needed for embedding generation)
try:
    from transformers import AutoTokenizer, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Embedding generation will fail.")


# ==================== Configuration ====================


def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """Load configuration from YAML file."""
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_default_paths(config: dict = None) -> dict:
    """Get default paths for data files."""
    if config is None:
        config = load_config()

    data_config = config.get("data", {})

    return {
        "srs_to_otu_parquet": Path(
            data_config.get(
                "srs_to_otu_parquet",
                "data_preprocessing/mapref_data/samples-otus-97.parquet",
            )
        ),
        "otu_to_dna_parquet": Path(
            data_config.get(
                "otu_to_dna_parquet",
                "data_preprocessing/mapref_data/otus_97_to_dna.parquet",
            )
        ),
        "dna_csv_dir": Path(
            data_config.get("dna_csv_dir", "data_preprocessing/dna_sequences")
        ),
        "embeddings_h5": Path(
            data_config.get(
                "embeddings_h5",
                "data_preprocessing/dna_embeddings/prokbert_embeddings.h5",
            )
        ),
        "microbiome_embeddings_h5": Path(
            data_config.get(
                "microbiome_embeddings_h5",
                "data_preprocessing/microbiome_embeddings/microbiome_embeddings.h5",
            )
        ),
        "checkpoint_path": Path(
            data_config.get(
                "checkpoint_path", "data/checkpoint_epoch_0_final_epoch3_conf00.pt"
            )
        ),
        "model_name": data_config.get(
            "embedding_model", "neuralbioinfo/prokbert-mini-long"
        ),
        "batch_size_embedding": data_config.get("batch_size_embedding", 32),
        "device": data_config.get("device", "cpu"),
    }


# ==================== Artifact Generation Functions ====================


def get_otus_from_srs(srs_id: str, srs_to_otu_parquet: Path) -> List[str]:
    """
    Get OTU IDs for a given SRS ID from parquet file.

    Args:
        srs_id: Sample ID (SRS/SID)
        srs_to_otu_parquet: Path to parquet file mapping srs_id -> otu_id

    Returns:
        List of OTU IDs
    """
    filters = [("srs_id", "=", srs_id)]
    table = pq.read_table(srs_to_otu_parquet, filters=filters)
    if len(table) == 0:
        return []
    return table.to_pandas()["otu_id"].tolist()


def get_dna_from_otu(otu_id: str, otu_to_dna_parquet: Path) -> Optional[str]:
    """
    Get DNA sequence for a given OTU ID from parquet file.

    Args:
        otu_id: OTU ID
        otu_to_dna_parquet: Path to parquet file mapping otu_97_id -> dna_sequence

    Returns:
        DNA sequence string, or None if not found
    """
    filters = [("otu_97_id", "=", otu_id)]
    table = pq.read_table(otu_to_dna_parquet, filters=filters)
    if len(table) == 0:
        return None
    return table.to_pandas().iloc[0]["dna_sequence"]


def generate_dna_csv_for_sample(
    srs_id: str, srs_to_otu_parquet: Path, otu_to_dna_parquet: Path, output_dir: Path
) -> Path:
    """
    Generate DNA CSV file for a single sample.

    Args:
        srs_id: Sample ID
        srs_to_otu_parquet: Path to SRS->OTU parquet
        otu_to_dna_parquet: Path to OTU->DNA parquet
        output_dir: Directory to save CSV file

    Returns:
        Path to generated CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{srs_id}.csv"

    # Check if already exists
    if csv_path.exists():
        return csv_path

    # Get OTUs for this sample
    otus = get_otus_from_srs(srs_id, srs_to_otu_parquet)

    if not otus:
        print(f"Warning: No OTUs found for {srs_id}")
        # Create empty CSV
        pd.DataFrame(columns=["otu_id", "dna_sequence"]).to_csv(csv_path, index=False)
        return csv_path

    # Get DNA sequences for each OTU
    srs_dna_map = {}
    for otu in tqdm(otus, desc=f"  Processing {srs_id}", leave=False):
        dna_seq = get_dna_from_otu(otu, otu_to_dna_parquet)
        if dna_seq is not None:
            srs_dna_map[otu] = dna_seq

    # Save to CSV
    df = pd.DataFrame(srs_dna_map.items(), columns=["otu_id", "dna_sequence"])
    df.to_csv(csv_path, index=False)

    return csv_path


def generate_dna_csvs_for_samples(
    sids: List[str],
    srs_to_otu_parquet: Path,
    otu_to_dna_parquet: Path,
    output_dir: Path,
) -> List[Path]:
    """
    Generate DNA CSV files for multiple samples.

    Args:
        sids: List of sample IDs
        srs_to_otu_parquet: Path to SRS->OTU parquet
        otu_to_dna_parquet: Path to OTU->DNA parquet
        output_dir: Directory to save CSV files

    Returns:
        List of paths to generated CSV files
    """
    generated_paths = []

    for srs_id in tqdm(sids, desc="Generating DNA CSVs"):
        csv_path = generate_dna_csv_for_sample(
            srs_id, srs_to_otu_parquet, otu_to_dna_parquet, output_dir
        )
        generated_paths.append(csv_path)

    return generated_paths


def generate_dna_embeddings_h5(
    dna_csv_dir: Path,
    output_h5_path: Path,
    model_name: str = "neuralbioinfo/prokbert-mini-long",
    batch_size: int = 32,
    device: str = "cpu",
) -> Path:
    """
    Generate embeddings H5 file from DNA CSV files.

    Args:
        dna_csv_dir: Directory containing DNA CSV files (one per SRS ID)
        output_h5_path: Path to output HDF5 file
        model_name: ProkBERT model name
        batch_size: Batch size for embedding generation
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        Path to generated H5 file
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for embedding generation")

    # Check if already exists
    if output_h5_path.exists():
        print(f"Embeddings H5 already exists at {output_h5_path}")
        return output_h5_path

    # Get all CSV files
    csv_files = [f for f in os.listdir(dna_csv_dir) if f.endswith(".csv")]

    if not csv_files:
        raise ValueError(f"No CSV files found in {dna_csv_dir}")

    print(f"Found {len(csv_files)} CSV files to process")

    # Load model
    device_obj = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device_obj)
    model.eval()

    # Create output directory
    output_h5_path.mkdir(parents=True, exist_ok=True)
    output_h5_path_file = output_h5_path / "dna_embeddings.h5"

    # Create/open HDF5 file
    with h5py.File(output_h5_path_file, "w") as hdf5_file:
        # Process each CSV file (one per SRS ID)
        for csv_file in tqdm(csv_files, desc="Processing SRS samples"):
            # Extract SRS ID from filename
            srs_id = csv_file.replace(".csv", "")

            # Read the CSV file
            csv_path = dna_csv_dir / csv_file
            table = pd.read_csv(csv_path)

            if len(table) == 0:
                print(f"  Warning: Empty CSV for {srs_id}, skipping")
                continue

            # Create group for this SRS ID
            srs_group = hdf5_file.create_group(srs_id)

            # Process sequences in batches
            num_sequences = len(table)

            for batch_start in range(0, num_sequences, batch_size):
                batch_end = min(batch_start + batch_size, num_sequences)
                batch_df = table.iloc[batch_start:batch_end]

                # Get batch of sequences
                batch_sequences = batch_df["dna_sequence"].tolist()
                batch_otu_ids = batch_df["otu_id"].tolist()

                # Tokenize batch
                with torch.no_grad():
                    inputs = tokenizer(
                        batch_sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device_obj)

                    # Get embeddings
                    outputs = model(**inputs)
                    # Masked mean over real tokens (exclude padding)
                    hidden = outputs.last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    denom = mask.sum(dim=1).clamp(min=1e-9)
                    embeddings = (hidden * mask).sum(dim=1) / denom
                    embeddings = embeddings.cpu().numpy()

                # Save each embedding with its OTU ID
                for i, otu_id in enumerate(batch_otu_ids):
                    # Replace forward slashes in OTU IDs if any (HDF5 key compatibility)
                    otu_key = str(otu_id).replace("/", "_")
                    srs_group.create_dataset(otu_key, data=embeddings[i])

            print(f"  ✓ Processed {num_sequences} OTUs for {srs_id}")

    print(f"\nComplete! Saved embeddings to {output_h5_path}")
    return output_h5_path


def ensure_embeddings_exist(
    sample_csv_path: Path,
    paths: dict = None,
    force_regenerate: bool = False,
    check_samples_match: bool = True,
) -> Path:
    """
    Ensure embeddings H5 file exists, generating it if necessary.

    Args:
        sample_csv_path: Path to sample CSV file (SID, label)
        paths: Dictionary of paths (from get_default_paths())
        force_regenerate: If True, regenerate even if H5 exists
        check_samples_match: If True, verify H5 contains samples from CSV

    Returns:
        Path to embeddings H5 file

    Raises:
        FileNotFoundError: If the sample CSV file doesn't exist
    """
    if paths is None:
        paths = get_default_paths()

    embeddings_h5 = paths["embeddings_h5"]

    # Convert to Path object if string
    sample_csv_path = Path(sample_csv_path)

    # Validate CSV file exists before proceeding
    if not sample_csv_path.exists():
        # Try to resolve the path for better error message
        try:
            abs_path = sample_csv_path.resolve()
        except (OSError, RuntimeError):
            abs_path = Path.cwd() / sample_csv_path

        raise FileNotFoundError(
            f"Sample CSV file not found: {sample_csv_path}\n"
            f"  Resolved path: {abs_path}\n"
            f"  Current directory: {Path.cwd()}\n"
            f"  Please check the file path and ensure the file exists."
        )

    # Load sample IDs from CSV
    labels_dict = load_labels(sample_csv_path)
    sids = list(labels_dict.keys())

    # Check if embeddings H5 exists
    if embeddings_h5.exists() and not force_regenerate:
        print(f"Embeddings H5 found at {embeddings_h5}")

        # Check if samples match
        if check_samples_match:
            h5_samples = inspect_embeddings_h5(embeddings_h5, max_samples=None)
            h5_samples_set = set(h5_samples)
            sids_set = set(sids)

            # Check how many samples match
            matching = sids_set & h5_samples_set
            missing = sids_set - h5_samples_set

            if len(matching) == 0 and len(sids) > 0:
                print("\nWarning: H5 file exists but contains different sample IDs!")
                print(
                    f"  Expected {len(sids)} samples from CSV, found {len(h5_samples_set)} in H5"
                )
                print("  No matching samples found.")
                print(
                    f"  Solution: Delete {embeddings_h5} and regenerate, or use force_regenerate=True"
                )
                # Don't raise error here - let load_dataset handle it with better diagnostics
            elif len(missing) > 0:
                print(f"  Found {len(matching)}/{len(sids)} matching samples in H5")
                print(
                    f"  Missing {len(missing)} samples - will generate missing embeddings"
                )

        return embeddings_h5

    print("Embeddings H5 not found or regeneration requested. Generating...")

    # Check which DNA CSVs exist
    dna_csv_dir = paths["dna_csv_dir"]
    existing_csvs = (
        set(
            f.replace(".csv", "") for f in os.listdir(dna_csv_dir) if f.endswith(".csv")
        )
        if dna_csv_dir.exists()
        else set()
    )

    # Generate missing DNA CSVs
    missing_sids = [sid for sid in sids if sid not in existing_csvs]
    if missing_sids:
        print(f"Generating {len(missing_sids)} missing DNA CSV files...")
        generate_dna_csvs_for_samples(
            missing_sids,
            paths["srs_to_otu_parquet"],
            paths["otu_to_dna_parquet"],
            paths["dna_csv_dir"],
        )
    else:
        print("All DNA CSV files already exist")

    # Generate embeddings H5
    print("Generating embeddings H5 file...")
    generate_dna_embeddings_h5(
        paths["dna_csv_dir"],
        embeddings_h5,
        paths["model_name"],
        paths["batch_size_embedding"],
        paths["device"],
    )

    return embeddings_h5


# ==================== Data Loading Functions ====================


def load_labels(
    labels_csv: Path, id_col: str = "sid", label_col: str = "label"
) -> Dict[str, int]:
    """
    Load labels from CSV file.

    Args:
        labels_csv: Path to CSV with sample IDs and labels
        id_col: Column name for sample IDs
        label_col: Column name for labels

    Returns:
        Dictionary mapping SID -> label

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file is empty or missing required columns
    """
    # Convert to Path object if string
    labels_csv = Path(labels_csv)

    # Check if file exists
    if not labels_csv.exists():
        # Try to resolve the path for better error message
        try:
            abs_path = labels_csv.resolve()
        except (OSError, RuntimeError):
            abs_path = Path.cwd() / labels_csv

        raise FileNotFoundError(
            f"CSV file not found: {labels_csv}\n"
            f"  Resolved path: {abs_path}\n"
            f"  Current directory: {Path.cwd()}\n"
            f"  Please check the file path and ensure the file exists."
        )

    # Check if file is readable
    if not labels_csv.is_file():
        raise ValueError(f"Path exists but is not a file: {labels_csv}")

    try:
        df = pd.read_csv(labels_csv)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {labels_csv}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {labels_csv}: {e}")

    # Handle different column name variations
    if id_col not in df.columns:
        # Try common variations
        for col in ["SID", "sid", "srs_id", "SRS_ID", "ERS_ID"]:
            if col in df.columns:
                id_col = col
                break
        else:
            # If still not found, use first column
            id_col = df.columns[0]
            print(f"Warning: ID column not found, using '{id_col}'")

    if label_col not in df.columns:
        # Try common variations
        for col in ["label", "Label", "LABEL", "y"]:
            if col in df.columns:
                label_col = col
                break
        else:
            # If still not found, use second column
            if len(df.columns) > 1:
                label_col = df.columns[1]
                print(f"Warning: Label column not found, using '{label_col}'")
            else:
                raise ValueError(f"Could not find label column in {labels_csv}")

    return dict(zip(df[id_col], df[label_col]))


def inspect_embeddings_h5(embeddings_h5: Path, max_samples: int = 10) -> List[str]:
    """
    Inspect what sample IDs are in the embeddings H5 file.

    Args:
        embeddings_h5: Path to embeddings H5 file
        max_samples: Maximum number of sample IDs to return (if None, returns all)

    Returns:
        List of sample IDs found in H5 file
    """
    try:
        with h5py.File(embeddings_h5, "r") as f:
            all_samples = list(f.keys())
            if max_samples is None or max_samples >= len(all_samples):
                return all_samples
            return all_samples[:max_samples]
    except Exception as e:
        print(f"Error inspecting H5 file: {e}")
        return []


def load_embeddings_for_sample(
    embeddings_h5: Path, srs_id: str
) -> Optional[np.ndarray]:
    """
    Load embeddings for a single sample from H5 file.

    Args:
        embeddings_h5: Path to embeddings H5 file
        srs_id: Sample ID

    Returns:
        Array of shape (num_otus, 384) or None if not found
    """
    try:
        with h5py.File(embeddings_h5, "r") as f:
            if srs_id not in f:
                return None

            srs_group = f[srs_id]
            otu_ids = list(srs_group.keys())

            if not otu_ids:
                return None

            # Stack all OTU embeddings
            embeddings = []
            for otu_id in otu_ids:
                embeddings.append(srs_group[otu_id][:])

            return np.stack(embeddings, axis=0)  # (num_otus, 384)
    except Exception as e:
        print(f"Error loading embeddings for {srs_id}: {e}")
        return None


def load_microbiome_embeddings_dataset(
    sample_csv_path: Path, microbiome_h5_path: Optional[Path] = None, paths: dict = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load fixed-size microbiome sample embeddings + labels.
    """
    sample_csv_path = Path(sample_csv_path)
    if paths is None:
        paths = get_default_paths()

    if microbiome_h5_path is None:
        microbiome_h5_path = paths["microbiome_embeddings_h5"]
    microbiome_h5_path = Path(microbiome_h5_path)

    if not sample_csv_path.exists():
        raise FileNotFoundError(f"Sample CSV file not found: {sample_csv_path}")
    if not microbiome_h5_path.exists():
        raise FileNotFoundError(
            f"Microbiome embeddings H5 file not found: {microbiome_h5_path}"
        )

    labels_dict = load_labels(sample_csv_path)

    X_list = []
    y_list = []
    sids_list = []
    missing_sids = []

    with h5py.File(microbiome_h5_path, "r") as h5f:
        available_sids = set(h5f.keys())

        for sid, label in labels_dict.items():
            if sid in available_sids:
                vec = h5f[sid][:]  # (D_MODEL,)
                X_list.append(vec)
                y_list.append(int(label))
                sids_list.append(sid)
            else:
                missing_sids.append(sid)

    if missing_sids:
        print(
            f"\nWarning: {len(missing_sids)} samples from CSV "
            f"are missing in microbiome embeddings H5."
        )
        print(f"  Example missing SIDs: {missing_sids[:5]}")

    if not X_list:
        raise ValueError(
            "No samples were loaded from microbiome_embeddings.h5. "
            "Check that the SIDs in your CSV match the H5 file."
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print(f"Loaded {X.shape[0]} microbiome sample embeddings from {microbiome_h5_path}")

    return X, y, sids_list


def load_dataset(
    sample_csv_path: Path, embeddings_h5_path: Optional[Path] = None, paths: dict = None
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load dataset: embeddings and labels from CSV + H5.

    Args:
        sample_csv_path: Path to sample CSV (SID, label)
        embeddings_h5_path: Path to embeddings H5 (if None, uses default)
        paths: Dictionary of paths (from get_default_paths())

    Returns:
        Tuple of (X_list, y_list, sids_list) where:
        - X_list[i]: (num_otus, 384) numpy array
        - y_list[i]: label (int)
        - sids_list[i]: sample ID (str)
    """
    if paths is None:
        paths = get_default_paths()

    if embeddings_h5_path is None:
        embeddings_h5_path = ensure_embeddings_exist(sample_csv_path, paths)

    # Load labels
    labels_dict = load_labels(sample_csv_path)

    # Load embeddings for each sample
    X_list = []
    y_list = []
    sids_list = []

    missing_samples = []

    for sid, label in tqdm(labels_dict.items(), desc="Loading samples"):
        embeddings = load_embeddings_for_sample(embeddings_h5_path, sid)

        if embeddings is not None and len(embeddings) > 0:
            X_list.append(embeddings)
            y_list.append(int(label))
            sids_list.append(sid)
        else:
            missing_samples.append(sid)

    if missing_samples:
        print(f"\nWarning: {len(missing_samples)} samples not found in embeddings H5")
        print(f"  Missing samples (first 5): {missing_samples[:5]}")

        # Inspect what's actually in the H5 file
        h5_samples = inspect_embeddings_h5(embeddings_h5_path, max_samples=10)
        if h5_samples:
            print(f"  Found samples in H5 (first 10): {h5_samples}")
        else:
            print("  H5 file appears to be empty or corrupted")

        # Check if all samples are missing
        if len(X_list) == 0:
            print(
                f"\nError: No samples loaded! All {len(missing_samples)} samples are missing."
            )
            print(
                "  This usually means the embeddings H5 was generated for different sample IDs."
            )
            print(f"  Solution: Delete {embeddings_h5_path} and regenerate embeddings.")
            raise ValueError(
                f"No samples found in embeddings H5. Expected samples like {missing_samples[:3]}, "
                f"but H5 contains different IDs. Please regenerate embeddings."
            )

    print(f"Loaded {len(X_list)} samples")
    return X_list, y_list, sids_list


# ==================== PyTorch Dataset & DataLoader ====================


class MicrobiomeDataset(Dataset):
    """
    PyTorch Dataset for microbiome data.

    Each sample returns:
    - embeddings: (num_otus, 384) numpy array
    - label: int
    - sid: str
    """

    def __init__(
        self,
        sample_csv_path: Path,
        embeddings_h5_path: Optional[Path] = None,
        paths: dict = None,
    ):
        """
        Initialize dataset.

        Args:
            sample_csv_path: Path to sample CSV (SID, label)
            embeddings_h5_path: Path to embeddings H5 (if None, auto-generates)
            paths: Dictionary of paths (from get_default_paths())

        Raises:
            FileNotFoundError: If the sample CSV file doesn't exist
        """
        # Convert to Path object if string
        sample_csv_path = Path(sample_csv_path)

        # Validate CSV file exists
        if not sample_csv_path.exists():
            # Try to resolve the path for better error message
            try:
                abs_path = sample_csv_path.resolve()
            except (OSError, RuntimeError):
                abs_path = Path.cwd() / sample_csv_path

            raise FileNotFoundError(
                f"Sample CSV file not found: {sample_csv_path}\n"
                f"  Resolved path: {abs_path}\n"
                f"  Current directory: {Path.cwd()}\n"
                f"  Please check the file path and ensure the file exists."
            )

        self.sample_csv_path = sample_csv_path
        self.paths = paths or get_default_paths()

        # Ensure embeddings exist
        if embeddings_h5_path is None:
            self.embeddings_h5_path = ensure_embeddings_exist(
                sample_csv_path, self.paths
            )
        else:
            self.embeddings_h5_path = embeddings_h5_path

        # Load data
        self.X_list, self.y_list, self.sids_list = load_dataset(
            sample_csv_path, self.embeddings_h5_path, self.paths
        )

        # Validate dataset is not empty
        if len(self.X_list) == 0:
            raise ValueError(
                f"Dataset is empty! No samples were loaded from {sample_csv_path}. "
                f"Check that embeddings H5 contains matching sample IDs."
            )

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single sample.

        Returns:
            Tuple of (embeddings_tensor, label, sid)
            embeddings_tensor: (num_otus, 384)
        """
        embeddings = torch.tensor(self.X_list[idx], dtype=torch.float32)
        label = self.y_list[idx]
        sid = self.sids_list[idx]

        return embeddings, label, sid


def collate_fn(batch: List[Tuple[torch.Tensor, int, str]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length sequences.

    Args:
        batch: List of (embeddings, label, sid) tuples

    Returns:
        Dictionary with:
        - 'embeddings': (batch_size, max_otus, 384) padded tensor
        - 'labels': (batch_size,) tensor
        - 'mask': (batch_size, max_otus) boolean mask (True for valid, False for padding)
        - 'sids': List of sample IDs
    """
    embeddings_list, labels_list, sids_list = zip(*batch)

    # Find max sequence length in batch
    max_otus = max(emb.shape[0] for emb in embeddings_list)
    emb_dim = embeddings_list[0].shape[1]  # Should be 384

    batch_size = len(batch)

    # Initialize padded tensors
    padded_embeddings = torch.zeros(batch_size, max_otus, emb_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_otus, dtype=torch.bool)
    labels = torch.tensor(labels_list, dtype=torch.long)

    # Fill in actual data
    for i, emb in enumerate(embeddings_list):
        seq_len = emb.shape[0]
        padded_embeddings[i, :seq_len] = emb
        mask[i, :seq_len] = True

    return {
        "embeddings": padded_embeddings,
        "labels": labels,
        "mask": mask,
        "sids": list(sids_list),
    }


def get_dataloader(
    sample_csv_path: Path,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    embeddings_h5_path: Optional[Path] = None,
    paths: dict = None,
) -> DataLoader:
    """
    Get PyTorch DataLoader for microbiome data.

    Args:
        sample_csv_path: Path to sample CSV (SID, label)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        embeddings_h5_path: Path to embeddings H5 (if None, auto-generates)
        paths: Dictionary of paths (from get_default_paths())

    Returns:
        PyTorch DataLoader

    Raises:
        FileNotFoundError: If the sample CSV file doesn't exist
        ValueError: If dataset is empty or batch_size is larger than dataset size
    """
    # Convert to Path object if string
    sample_csv_path = Path(sample_csv_path)

    # Validate CSV file exists before creating dataset
    if not sample_csv_path.exists():
        # Try to resolve the path for better error message
        try:
            abs_path = sample_csv_path.resolve()
        except (OSError, RuntimeError):
            abs_path = Path.cwd() / sample_csv_path

        raise FileNotFoundError(
            f"Sample CSV file not found: {sample_csv_path}\n"
            f"  Resolved path: {abs_path}\n"
            f"  Current directory: {Path.cwd()}\n"
            f"  Please check the file path and ensure the file exists."
        )

    dataset = MicrobiomeDataset(sample_csv_path, embeddings_h5_path, paths)

    # Validate dataset size
    if len(dataset) == 0:
        raise ValueError(
            "Cannot create DataLoader: dataset is empty. "
            "Check that embeddings H5 contains matching sample IDs."
        )

    # Adjust batch size if needed
    if batch_size > len(dataset):
        print(
            f"Warning: batch_size ({batch_size}) > dataset size ({len(dataset)}). "
            f"Setting batch_size to {len(dataset)}"
        )
        batch_size = len(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# =================== MicrobiomeTransformer Embeddings Generate ==========


def generate_microbiome_embeddings_h5(
    prokbert_h5_path: Path,
    output_h5_path: Path,
    checkpoint_path: Path,
    device: str = "cpu",
) -> Path:
    """
    Generate microbiome sample embeddings H5 from ProkBERT OTU embeddings.

    For each sample ID (group) in prokbert_h5_path:
      - load all OTU embeddings (num_otus, 384)
      - run them through the MicrobiomeTransformer encoder
      - mean-pool over sequence length
      - save a single vector per sample to output_h5_path

    Output layout:
      /<sid> -> (d_model,) float32
    """
    device_obj = torch.device(device)

    # Checkpoint values from example_scripts/utils
    D_MODEL = 100
    NHEAD = 5
    NUM_LAYERS = 5
    DIM_FF = 400
    OTU_EMB = 384
    TXT_EMB = 1536
    DROPOUT = 0.1

    # Load checkpoint + model
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    state_dict = checkpoint["model_state_dict"]

    model = MicrobiomeTransformer(
        input_dim_type1=OTU_EMB,
        input_dim_type2=TXT_EMB,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        use_output_activation=False,  # just want features
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device_obj)
    model.eval()

    output_h5_path.mkdir(parents=True, exist_ok=True)
    output_h5_path_file = output_h5_path / "microbiome_embeddings.h5"

    with h5py.File(prokbert_h5_path, "r") as h_in:
        sample_ids = list(h_in.keys())
        if not sample_ids:
            raise ValueError(
                f"ProkBERT H5 file {prokbert_h5_path} contains no sample groups."
            )

        with h5py.File(output_h5_path_file, "w") as h_out:
            for srs_id in tqdm(
                sample_ids, desc="Building microbiome sample embeddings"
            ):
                srs_group = h_in[srs_id]
                otu_ids = list(srs_group.keys())
                if not otu_ids:
                    continue

                # Stack OTU embeddings -> (num_otus, 384)
                vectors = [srs_group[otu_id][:] for otu_id in otu_ids]
                otu_tensor = torch.tensor(
                    vectors, dtype=torch.float32, device=device_obj
                ).unsqueeze(0)  # (1, num_otus, 384)

                # No text embeddings in this step
                type2_tensor = torch.zeros(
                    (1, 0, TXT_EMB), dtype=torch.float32, device=device_obj
                )
                mask = torch.ones(
                    (1, otu_tensor.shape[1]), dtype=torch.bool, device=device_obj
                )

                with torch.no_grad():
                    hidden_type1 = model.input_projection_type1(otu_tensor)
                    hidden_type2 = model.input_projection_type2(type2_tensor)
                    combined_hidden = torch.cat([hidden_type1, hidden_type2], dim=1)
                    hidden = model.transformer(
                        combined_hidden, src_key_padding_mask=~mask
                    )
                    sample_vec = (
                        hidden.mean(dim=1).squeeze(0).cpu().numpy()
                    )  # (D_MODEL,)

                h_out.create_dataset(srs_id, data=sample_vec)

    print(f"\nSaved microbiome sample embeddings to {output_h5_path}")
    return output_h5_path


def build_paths(config: dict, dataset_path: Path):
    # extract dataset name and csv file name from dataset path (example: tanaka, month_2.csv)
    dataset_name = dataset_path.parent.name
    dataset_csv_name = dataset_path.name.split(".")[0]

    sequences_dir = Path(
        config["data"]["dna_csv_dir"] + "/" + dataset_name + "/" + dataset_csv_name
    )
    dna_embeddings_dir = Path(
        config["data"]["dna_embeddings_dir"]
        + "/"
        + dataset_name
        + "/"
        + dataset_csv_name
    )
    microbiome_embeddings_dir = Path(
        config["data"]["microbiome_embeddings_dir"]
        + "/"
        + dataset_name
        + "/"
        + dataset_csv_name
    )
    return sequences_dir, dna_embeddings_dir, microbiome_embeddings_dir


def extract_csv_sequences(sequences_dir: Path, config: dict):
    # Load sample IDs from CSV
    labels_dict = load_labels(config["data"]["dataset_path"])
    sids = list(labels_dict.keys())

    print(f"Extracting CSV sequences for {len(sids)} samples")
    generate_dna_csvs_for_samples(
        sids,
        config["data"]["srs_to_otu_parquet"],
        config["data"]["otu_to_dna_parquet"],
        sequences_dir,
    )


def sanity_check_dna_and_microbiome_embeddings(
    dna_embeddings_dir: Path, microbiome_embeddings_dir: Path
):
    """
    Sanity check for consistency on dna embeddings and microbiome embeddings
    Args:
        dna_embeddings_dir: Path to dna embeddings directory
        microbiome_embeddings_dir: Path to microbiome embeddings directory
    Returns:
        True if sanity check is passed
    Raises:
        FileNotFoundError: If DNA embeddings H5 file not found
        FileNotFoundError: If Microbiome embeddings H5 file not found
        ValueError: If DNA embeddings and microbiome embeddings have different number of samples
        ValueError: If DNA embeddings and microbiome embeddings have different sample IDs
    """

    dna_embeddings_h5_path = dna_embeddings_dir / "dna_embeddings.h5"
    microbiome_embeddings_h5_path = (
        microbiome_embeddings_dir / "microbiome_embeddings.h5"
    )
    if not dna_embeddings_h5_path.exists():
        raise FileNotFoundError(
            f"DNA embeddings H5 file not found: {dna_embeddings_h5_path}"
        )
    if not microbiome_embeddings_h5_path.exists():
        raise FileNotFoundError(
            f"Microbiome embeddings H5 file not found: {microbiome_embeddings_h5_path}"
        )
    if len(inspect_embeddings_h5(dna_embeddings_h5_path)) != len(
        inspect_embeddings_h5(microbiome_embeddings_h5_path)
    ):
        raise ValueError(
            "DNA embeddings and microbiome embeddings have different number of samples"
        )
    if set(inspect_embeddings_h5(dna_embeddings_h5_path)) != set(
        inspect_embeddings_h5(microbiome_embeddings_h5_path)
    ):
        raise ValueError(
            "DNA embeddings and microbiome embeddings have different sample IDs"
        )
    print("DNA embeddings and microbiome embeddings are consistent")
    return True


def load_abundance_for_sid(                                                                 #<-
    sid: str,
    abundance_h5: h5py.File,
    ab_ids: List[str],
    ab_matrix: np.ndarray,
    abundance_dim: int,
) -> np.ndarray:
    """
    Return the abundance vector for a given sample ID.
    Falls back to a zero vector if the sample is not found in the abundance file.

    Args:
        sid: Sample ID
        abundance_h5: Open h5py File handle for abundance.h5
        ab_ids: List of sample IDs from abundance_matrix (pre-loaded)
        ab_matrix: Full abundance matrix (n_samples, abundance_dim)
        abundance_dim: Dimensionality of abundance vector
    Returns:
        numpy array of shape (abundance_dim,)
    """
    if sid in ab_ids:
        idx = ab_ids.index(sid)
        return ab_matrix[idx]
    return np.zeros(abundance_dim, dtype=np.float32)                                         #<-


def create_dataset_df(                                                                       #<-
    dataset_path: Path,
    microbiome_embeddings_dir: Path,
    abundance_h5_path: Optional[Path] = None,
    abundance_zero_pad_missing: bool = False,
):
    """
    Iterate through the csv ids, take the label and use the id to look in the h5
    file to get the microbiome embedding, then create a dataframe with the id, label, and embedding.

    If abundance_h5_path is provided, the abundance vector is concatenated to the
    microbiome embedding: final embedding = [microbiome (100) | abundance (2123)].

    The 82 samples that have abundance but NO microbiome embedding are handled by
    abundance_zero_pad_missing:
      - False (default): drop them — only samples with a real microbiome embedding
        are included. Safer for training; avoids injecting 100 zeros as a spurious signal.
      - True: include them with a zero-padded microbiome embedding (100 zeros).
        Gives more training samples; useful when the dataset is small.

    Args:
        dataset_path: Path to dataset csv
        microbiome_embeddings_dir: Path to microbiome embeddings directory
        abundance_h5_path: Optional path to abundance.h5 for concatenation
        abundance_zero_pad_missing: Whether to include abundance-only samples
            by zero-padding their missing microbiome embedding (default: False)
    Returns:
        DataFrame with id, label, and embedding
    """
    dataset_df = pd.read_csv(dataset_path)
    microbiome_embeddings_h5_path = (
        microbiome_embeddings_dir / "microbiome_embeddings.h5"
    )

    if not microbiome_embeddings_h5_path.exists():
        raise FileNotFoundError(
            f"Microbiome embeddings H5 file not found: {microbiome_embeddings_h5_path}"
        )

    # Pre-load abundance data if provided
    ab_ids, ab_matrix, abundance_dim = [], None, 0
    if abundance_h5_path is not None:
        if not abundance_h5_path.exists():
            raise FileNotFoundError(
                f"Abundance H5 file not found: {abundance_h5_path}"
            )
        with h5py.File(abundance_h5_path, "r") as ab_h5:
            ab_ids = list(ab_h5["sample_ids"][:].astype(str))
            ab_matrix = ab_h5["abundance_matrix"][:]  # (n_samples, abundance_dim)
            abundance_dim = ab_matrix.shape[1]
        print(f"Loaded abundance matrix {ab_matrix.shape} from {abundance_h5_path}")

    # Read embeddings from H5 file using h5py
    embeddings_data = []
    with h5py.File(microbiome_embeddings_h5_path, "r") as h5f:
        mb_sids = set(h5f.keys())
        for sid in h5f.keys():
            emb = h5f[sid][:]  # (D_MODEL,) numpy array
            if abundance_h5_path is not None:
                ab_vec = load_abundance_for_sid(sid, None, ab_ids, ab_matrix, abundance_dim)
                emb = np.concatenate([emb, ab_vec])
            embeddings_data.append({"sid": sid, "embedding": emb})

        # Optionally include abundance-only samples (zero-pad microbiome side)
        if abundance_h5_path is not None and abundance_zero_pad_missing:
            microbiome_dim = list(h5f.values())[0].shape[0]
            abundance_only = [s for s in ab_ids if s not in mb_sids]
            if abundance_only:
                print(
                    f"Zero-padding microbiome embedding for {len(abundance_only)} "
                    f"abundance-only samples (abundance_zero_pad_missing=True)"
                )
                for sid in abundance_only:
                    mb_zeros = np.zeros(microbiome_dim, dtype=np.float32)
                    ab_vec = load_abundance_for_sid(sid, None, ab_ids, ab_matrix, abundance_dim)
                    emb = np.concatenate([mb_zeros, ab_vec])
                    embeddings_data.append({"sid": sid, "embedding": emb})

    microbiome_embeddings_df = pd.DataFrame(embeddings_data)
    dataset_df = dataset_df.merge(
        microbiome_embeddings_df, left_on="sid", right_on="sid", how="left"
    )

    if abundance_h5_path is not None:
        emb_dim = embeddings_data[0]["embedding"].shape[0]
        print(
            f"Final embedding dim: {emb_dim} "
            f"(microbiome={emb_dim - abundance_dim} + abundance={abundance_dim})"
        )

    return dataset_df                                                                       #<-


def create_dataset_df_from_unified(                                                         #<-
    dataset_path: Path,
    unified_embeddings_h5_path: Path,
    abundance_h5_path: Optional[Path] = None,
    abundance_zero_pad_missing: bool = False,
) -> pd.DataFrame:
    """
    Create dataset DataFrame using unified embeddings file.

    This function loads embeddings from a unified H5 file that contains
    all sample IDs across all months/groups for a dataset. This allows
    for flexible loading of custom sample subsets without recomputing embeddings.

    If abundance_h5_path is provided, the abundance vector is concatenated to the
    microbiome embedding: final embedding = [microbiome (100) | abundance (2123)].

    The 82 samples that have abundance but NO microbiome embedding are handled by
    abundance_zero_pad_missing:
      - False (default): drop them — only samples with a real microbiome embedding
        are included. Safer for training; avoids injecting 100 zeros as a spurious signal.
      - True: include them with a zero-padded microbiome embedding (100 zeros).
        Gives more training samples; useful when the dataset is small.

    Args:
        dataset_path: Path to dataset csv with 'sid' and 'label' columns
        unified_embeddings_h5_path: Path to unified microbiome embeddings H5 file
        abundance_h5_path: Optional path to abundance.h5 for concatenation
        abundance_zero_pad_missing: Whether to include abundance-only samples
            by zero-padding their missing microbiome embedding (default: False)

    Returns:
        DataFrame with id, label, and embedding
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset CSV file not found: {dataset_path}")

    if not unified_embeddings_h5_path.exists():
        raise FileNotFoundError(
            f"Unified embeddings H5 file not found: {unified_embeddings_h5_path}\n"
            f"Run 'python create_unified_embeddings.py' to generate unified embeddings."
        )

    # Load dataset CSV
    dataset_df = pd.read_csv(dataset_path)

    if "sid" not in dataset_df.columns:
        raise ValueError(
            f"Dataset CSV must contain 'sid' column. Found: {dataset_df.columns.tolist()}"
        )

    # Check for duplicates in CSV
    if dataset_df["sid"].duplicated().any():
        n_duplicates = dataset_df["sid"].duplicated().sum()
        duplicate_sids = dataset_df[dataset_df["sid"].duplicated(keep=False)][
            "sid"
        ].unique()
        print(f"\n⚠️  Warning: Found {n_duplicates} duplicate sample IDs in CSV:")
        for dup_sid in duplicate_sids[:5]:
            count = (dataset_df["sid"] == dup_sid).sum()
            print(f"  - {dup_sid}: appears {count} times")
        if len(duplicate_sids) > 5:
            print(f"  ... and {len(duplicate_sids) - 5} more")
        print("\n  Keeping first occurrence of each duplicate.\n")

        # Remove duplicates, keeping first occurrence
        dataset_df = dataset_df.drop_duplicates(subset=["sid"], keep="first")

    # Pre-load abundance data if provided
    ab_ids, ab_matrix, abundance_dim = [], None, 0
    if abundance_h5_path is not None:
        if not abundance_h5_path.exists():
            raise FileNotFoundError(
                f"Abundance H5 file not found: {abundance_h5_path}"
            )
        with h5py.File(abundance_h5_path, "r") as ab_h5:
            ab_ids = list(ab_h5["sample_ids"][:].astype(str))
            ab_matrix = ab_h5["abundance_matrix"][:]  # (n_samples, abundance_dim)
            abundance_dim = ab_matrix.shape[1]
        print(f"Loaded abundance matrix {ab_matrix.shape} from {abundance_h5_path}")

    # Read embeddings from unified H5 file (now only for unique SIDs)
    embeddings_data = []
    missing_sids = []

    with h5py.File(unified_embeddings_h5_path, "r") as h5f:
        available_sids = set(h5f.keys())

        for sid in dataset_df["sid"]:
            if sid in available_sids:
                emb = h5f[sid][:]  # (D_MODEL,) numpy array
                if abundance_h5_path is not None:
                    ab_vec = load_abundance_for_sid(sid, None, ab_ids, ab_matrix, abundance_dim)
                    emb = np.concatenate([emb, ab_vec])
                embeddings_data.append({"sid": sid, "embedding": emb})
            else:
                missing_sids.append(sid)

    if missing_sids:
        print(
            f"\nWarning: {len(missing_sids)} sample IDs from CSV not found in unified embeddings:"
        )
        print(f"  Example missing SIDs: {missing_sids[:5]}")
        if len(missing_sids) == len(dataset_df):
            raise ValueError(
                "No sample IDs from CSV were found in unified embeddings. "
                "Check that you're using the correct unified embeddings file for this dataset."
            )

    # Optionally include abundance-only samples (zero-pad microbiome side)
    if abundance_h5_path is not None and abundance_zero_pad_missing and embeddings_data:
        microbiome_dim = embeddings_data[0]["embedding"].shape[0] - abundance_dim
        abundance_only = [s for s in ab_ids if s not in available_sids]
        if abundance_only:
            print(
                f"Zero-padding microbiome embedding for {len(abundance_only)} "
                f"abundance-only samples (abundance_zero_pad_missing=True)"
            )
            for sid in abundance_only:
                mb_zeros = np.zeros(microbiome_dim, dtype=np.float32)
                ab_vec = load_abundance_for_sid(sid, None, ab_ids, ab_matrix, abundance_dim)
                emb = np.concatenate([mb_zeros, ab_vec])
                embeddings_data.append({"sid": sid, "embedding": emb})

    if not embeddings_data:
        raise ValueError(
            "No embeddings loaded from unified file. "
            "Check that the SIDs in your CSV match those in the unified embeddings."
        )

    # Create DataFrame with embeddings
    microbiome_embeddings_df = pd.DataFrame(embeddings_data)
    dataset_df = dataset_df.merge(
        microbiome_embeddings_df, left_on="sid", right_on="sid", how="inner"
    )                                                                                      #<- 

    print(f"Loaded {len(dataset_df)} samples from unified embeddings")

    return dataset_df


def download_dataset_from_hf(
    config: DictConfig,
    console: Console,
) -> Tuple[Path, Path, Path]:
    """
    Clone or update dataset from Git LFS repository and return paths.
    Args:
        config: Dictionary with keys:
            - dataset_repo_url: URL of the Git repo (with LFS)
            - download_cache_dir: Local path to store dataset
    Returns:
        Tuple of Paths: (sequences_dir, dna_embeddings_dir, microbiome_embeddings_dir)
    """
    base_repo_url = config.data.hugging_face.base_repo_url

    dataset_name = config.data.hugging_face.dataset_name
    if dataset_name not in config.valid_dataset_names:
        raise ValueError(
            f"Dataset name '{dataset_name}' not in valid filenames: "
            f"{list(config.valid_dataset_names)}"
        )

    download_cache_dir = Path(config.data.hugging_face.download_path) / dataset_name
    if not download_cache_dir.exists():
        download_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_repo_url = f"{base_repo_url}/AI4FA-{str(dataset_name)}"

    csv_filename = config.data.hugging_face.csv_filename
    csv_folder = config.data.hugging_face.csv_filename.split(".")[0]

    # Ensure Git LFS is installed
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "Git LFS installation failed. Make sure Git LFS is installed."
        )

    if download_cache_dir.exists() and (download_cache_dir / ".git").exists():
        # Repo already cloned; pull latest changes
        console.print(
            f"Updating existing dataset repo at {download_cache_dir}...", style="info"
        )
        subprocess.run(["git", "-C", str(download_cache_dir), "pull"], check=True)
        subprocess.run(
            ["git", "-C", str(download_cache_dir), "lfs", "pull"], check=True
        )
    else:
        # Clone repo
        console.print(
            f"Cloning dataset repo from {dataset_repo_url} to {download_cache_dir}...",
            style="info",
        )
        subprocess.run(
            ["git", "clone", dataset_repo_url, str(download_cache_dir)], check=True
        )
        subprocess.run(
            ["git", "-C", str(download_cache_dir), "lfs", "pull"], check=True
        )

    console.print(f"Dataset ready at {download_cache_dir}", style="success")

    dataset_path = download_cache_dir / "metadata" / csv_filename
    if not dataset_path.exists():
        console.print("Available files in metadata directory:", style="warning")
        for f in (download_cache_dir / "metadata").iterdir():
            console.print(f"  {f}", style="warning")
        raise FileNotFoundError(
            f"Dataset CSV file not found at expected path: {dataset_path}"
        )
    sequences_dir = download_cache_dir / "processed" / "dna_sequences" / csv_folder
    dna_embeddings_dir = (
        download_cache_dir / "processed" / "dna_embeddings" / csv_folder
    )
    microbiome_embeddings_dir = (
        download_cache_dir / "processed" / "microbiome_embeddings" / csv_folder
    )

    return dataset_path, sequences_dir, dna_embeddings_dir, microbiome_embeddings_dir


def load_dataset_df(config: DictConfig, console: Console) -> pd.DataFrame:                        #<-
    """
    Process dataset and return dataframe with id, label, and embedding.

    Supports two modes:
    1. Unified mode (use_unified_embeddings=True): Load from unified embeddings file
    2. Standard mode (use_unified_embeddings=False): Traditional pipeline

    If config.data.abundance_h5_path is set, the abundance vector is concatenated
    to every microbiome embedding (option 2 fusion):
      final embedding = [microbiome (100) | abundance (2123)]
    Missing abundance samples are zero-padded.

    Args:
        config: Configuration dictionary
    Returns:
        DataFrame with id, label, and embedding
    """
    # Resolve optional abundance path
    abundance_h5_path = None
    raw_abundance_path = getattr(config.data, "abundance_h5_path", None)
    if raw_abundance_path:
        abundance_h5_path = Path(raw_abundance_path)
        console.print(f"Abundance concatenation enabled: {abundance_h5_path}", style="info")

    abundance_zero_pad_missing = bool(
        getattr(config.data, "abundance_zero_pad_missing", False)
    )
    if abundance_h5_path is not None:
        console.print(
            f"abundance_zero_pad_missing={abundance_zero_pad_missing} "
            f"({'including' if abundance_zero_pad_missing else 'dropping'} "
            f"the ~82 samples with abundance but no microbiome embedding)",
            style="info",
        )

    if config.data.hugging_face.pull_from_huggingface:
        dataset_path, sequences_dir, dna_embeddings_dir, microbiome_embeddings_dir = (
            download_dataset_from_hf(config=config, console=console)
        )
    else:
        dataset_path = Path(config.data.dataset_path)
        sequences_dir, dna_embeddings_dir, microbiome_embeddings_dir = build_paths(
            config, dataset_path
        )

    # Check if unified embeddings should be used
    if config.data.use_unified_embeddings:
        # Try to use unified embeddings
        if config.data.hugging_face.pull_from_huggingface:
            dataset_name = config.data.hugging_face.dataset_name
            unified_path = (
                Path(config.data.hugging_face.download_path)
                / dataset_name
                / "processed"
                / "microbiome_embeddings"
                / "unified_all_samples.h5"
            )
        else:
            # For local datasets, infer the dataset name from the path
            # Assuming structure: .../huggingface_datasets/{dataset_name}/...
            try:
                dataset_root = dataset_path
                while (
                    dataset_root.parent.name != "huggingface_datasets"
                    and dataset_root != dataset_root.parent
                ):
                    dataset_root = dataset_root.parent

                if dataset_root.parent.name == "huggingface_datasets":
                    unified_path = (
                        dataset_root
                        / "processed"
                        / "microbiome_embeddings"
                        / "unified_all_samples.h5"
                    )
                else:
                    raise ValueError(
                        "Could not determine dataset root for unified embeddings"
                    )
            except Exception as e:
                console.print(
                    f"Warning: Could not locate unified embeddings: {e}", style="danger"
                )
                console.print("Falling back to standard pipeline...", style="warning")
                config.data.use_unified_embeddings = False

        if config.data.use_unified_embeddings:
            if unified_path.exists():
                console.print(
                    f"Using unified embeddings from: {unified_path}", style="success"
                )
                dataset_df = create_dataset_df_from_unified(
                    dataset_path, unified_path,
                    abundance_h5_path=abundance_h5_path,
                    abundance_zero_pad_missing=abundance_zero_pad_missing,
                )
                print(
                    f"Created dataset dataframe with {len(dataset_df)} rows and {len(dataset_df.columns)} columns"
                )
                return dataset_df
            else:
                console.print(
                    f"Warning: Unified embeddings not found at {unified_path}",
                    style="warning",
                )
                console.print(
                    "Run 'python create_unified_embeddings.py' to generate unified embeddings.",
                    style="info",
                )
                console.print("Falling back to standard pipeline...", style="warning")
                config.data.use_unified_embeddings = False

    # Standard pipeline (if unified not requested or not available)

    # 1) extract csv sequences from dataset
    if not sequences_dir.exists():
        # if not already extracted
        extract_csv_sequences(sequences_dir, config)
    else:
        console.print(
            f"CSV sequences already exist at {sequences_dir}, skipping extraction",
            style="info",
        )

    # 2) generate dna embeddings from csv sequences
    if not dna_embeddings_dir.exists():
        generate_dna_embeddings_h5(
            sequences_dir,
            dna_embeddings_dir,
            config["data"]["embedding_model"],
            batch_size=config["data"]["batch_size_embedding"],
            device=config["data"]["device"],
        )
    else:
        console.print(
            f"DNA embeddings already exist at {dna_embeddings_dir}, skipping generation",
            style="info",
        )

    # 3) generate microbiome embeddings from dna embeddings
    if not microbiome_embeddings_dir.exists():
        dna_embeddings_h5_path = dna_embeddings_dir / "dna_embeddings.h5"
        generate_microbiome_embeddings_h5(
            dna_embeddings_h5_path,
            microbiome_embeddings_dir,
            config["data"]["microbiome_transformer_checkpoint"],
            device=config["data"]["device"],
        )
    else:
        console.print(
            f"Microbiome embeddings already exist at {microbiome_embeddings_dir}, skipping generation",
            style="info",
        )

    # sanity check for consistency on dna embeddings and microbiome embeddings
    if not sanity_check_dna_and_microbiome_embeddings(
        dna_embeddings_dir, microbiome_embeddings_dir
    ):
        raise ValueError(
            "DNA embeddings and microbiome embeddings are not consistent",
            style="danger",
        )
    else:
        console.print("Data sanity check passed")

    # create dataframe relating labels from dataset csv and corresponding microbiome embeddings
    dataset_df = create_dataset_df(
        dataset_path, microbiome_embeddings_dir,
        abundance_h5_path=abundance_h5_path,
        abundance_zero_pad_missing=abundance_zero_pad_missing,
    )
    console.print(
        f"Created dataset dataframe with {len(dataset_df)} rows and {len(dataset_df.columns)} columns",
        style="success",
    )
    return dataset_df


def load_train_test_datasets(
    config: DictConfig,
    console: Optional[Console] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test datasets based on the configured split mode.

    Split modes:
      - "none": Raises error (use load_dataset_df directly for full dataset)
      - "auto": Stratified split of single dataset
      - "manual": Load separate pre-split train/test files

    Returns:
        (train_df, test_df) with validated schema alignment and leakage checks
    """
    if console is None:
        console = Console()

    split_cfg = config.evaluation.split
    mode = split_cfg.mode.lower()

    valid_modes = ("none", "auto", "manual")
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid split mode: '{mode}'. "
            f"Must be one of: {valid_modes}"
        )

    if mode == "none":
        raise ValueError(
            "Split mode is 'none' but load_train_test_datasets was called. "
            "Use load_dataset_df directly for full dataset evaluation, "
            "or set evaluation.split.mode to 'auto' or 'manual'."
        )

    elif mode == "manual":
        # --- Manual pre-split mode ---
        train_path_str = split_cfg.manual.train_dataset_path
        test_path_str = split_cfg.manual.test_dataset_path

        # Validate paths are configured
        if train_path_str is None or test_path_str is None:
            raise ValueError(
                "Split mode is 'manual' but paths are not configured.\n"
                "Set both:\n"
                "  - evaluation.split.manual.train_dataset_path\n"
                "  - evaluation.split.manual.test_dataset_path"
            )

        console.print(
            f"[bold]Loading manual train/test split...[/bold]", style="bold"
        )

        # Validate paths exist
        train_path = Path(train_path_str)
        test_path = Path(test_path_str)

        if not train_path.exists():
            raise FileNotFoundError(
                f"Train dataset path does not exist: {train_path}\n"
                f"Check evaluation.split.manual.train_dataset_path"
            )
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test dataset path does not exist: {test_path}\n"
                f"Check evaluation.split.manual.test_dataset_path"
            )

        # Catch same-file error
        if train_path.resolve() == test_path.resolve():
            raise ValueError(
                "CRITICAL: Train and test paths point to the same file!\n"
                f"  Train: {train_path}\n"
                f"  Test: {test_path}"
            )

        cfg_dict = OmegaConf.to_container(config, resolve=True)

        # Load train set
        console.print(f"  Loading train set from: {train_path_str}", style="dim")
        cfg_dict["data"]["dataset_path"] = train_path_str
        train_config = OmegaConf.create(cfg_dict)
        train_df = load_dataset_df(train_config, console=console)

        # Load test set
        console.print(f"  Loading test set from: {test_path_str}", style="dim")
        cfg_dict["data"]["dataset_path"] = test_path_str
        # Infer test dataset name and csv filename from the test path
        # Expected path structure: huggingface_datasets/{DatasetName}/metadata/{file.csv}
        test_path_obj = Path(test_path_str)
        test_dataset_name = test_path_obj.parts[-3]  # e.g. "Gadir"
        test_csv_filename = test_path_obj.name        # e.g. "gadir_all_months.csv"
        cfg_dict["data"]["hugging_face"]["dataset_name"] = test_dataset_name
        cfg_dict["data"]["hugging_face"]["csv_filename"] = test_csv_filename
        test_config = OmegaConf.create(cfg_dict)
        test_df = load_dataset_df(test_config, console=console)
        
        # --- Schema alignment check ---
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)

        if train_cols != test_cols:
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            error_msg = "Schema mismatch between train and test datasets:\n"
            if missing_in_test:
                error_msg += f"  Columns in train but not test: {missing_in_test}\n"
            if missing_in_train:
                error_msg += f"  Columns in test but not train: {missing_in_train}\n"
            raise ValueError(error_msg)

        # --- Sample ID overlap check (data leakage) ---
        if "sid" in train_df.columns and "sid" in test_df.columns:
            train_sids = set(train_df["sid"])
            test_sids = set(test_df["sid"])
            overlap = train_sids & test_sids

            if overlap:
                raise ValueError(
                    f"CRITICAL: Data leakage detected! {len(overlap)} sample IDs "
                    f"appear in BOTH train and test sets.\n"
                    f"  Overlapping IDs (first 5): {list(overlap)[:5]}\n"
                    f"  Did you split BEFORE creating the CSV files?"
                )

        # --- Label distribution check ---
        if "label" in train_df.columns and "label" in test_df.columns:
            train_labels = train_df["label"].value_counts(normalize=True)
            test_labels = test_df["label"].value_counts(normalize=True)

            console.print("\n  Label distributions:", style="dim")
            console.print(f"    Train: {dict(train_labels.round(3))}", style="dim")
            console.print(f"    Test:  {dict(test_labels.round(3))}", style="dim")

            # Warn if distributions are very different
            for label in train_labels.index:
                if label in test_labels.index:
                    diff = abs(train_labels[label] - test_labels[label])
                    if diff > 0.2:  # More than 20% difference
                        console.print(
                            f"  ⚠️  Warning: Label '{label}' has very different "
                            f"proportions in train ({train_labels[label]:.1%}) vs "
                            f"test ({test_labels[label]:.1%})",
                            style="warning",
                        )

        console.print(
            f"\n✓ Manual split datasets loaded: "
            f"{len(train_df)} train, {len(test_df)} test samples",
            style="success",
        )
        return train_df, test_df

    else:  # mode == "auto"
        # --- Auto split mode ---
        test_size = split_cfg.auto.test_size
        random_state = split_cfg.auto.random_state

        if test_size is None:
            raise ValueError(
                "Split mode is 'auto' but test_size is not configured.\n"
                "Set evaluation.split.auto.test_size (e.g., 0.2 or 0.3)"
            )

        console.print(
            f"[bold]Loading dataset for auto split "
            f"(test_size={test_size}, random_state={random_state})...[/bold]",
            style="bold",
        )

        full_df = load_dataset_df(config, console=console)

        # Validate we have enough samples
        min_samples = max(10, int(1 / test_size) + 1)
        if len(full_df) < min_samples:
            console.print(
                f"⚠️  Warning: Dataset has only {len(full_df)} samples. "
                f"Consider using more data for reliable train/test evaluation.",
                style="warning",
            )

        train_df, test_df = split_dataset_df(
            full_df,
            test_size=test_size,
            random_state=random_state,
        )

        console.print(
            f"✓ Auto-split complete: {len(train_df)} train, {len(test_df)} test samples",
            style="success",
        )
        return train_df, test_df


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example usage
    config_path = Path("config.yaml")
    config = load_config(config_path)
    dataset_path = Path(
        config["data"]["dataset_path"]
    )  # example: data_preprocessing/datasets/tanaka/month_2.csv

    sequences_dir, dna_embeddings_dir, microbiome_embeddings_dir = build_paths(
        config, dataset_path
    )

    # 1) extract csv sequences from dataset
    if not sequences_dir.exists():
        # if not already extracted
        extract_csv_sequences(sequences_dir, config)
    else:
        print(f"CSV sequences already exist at {sequences_dir}, skipping extraction")

    # 2) generate dna embeddings from csv sequences
    if not dna_embeddings_dir.exists():
        generate_dna_embeddings_h5(
            sequences_dir,
            dna_embeddings_dir,
            config["data"]["embedding_model"],
            batch_size=config["data"]["batch_size_embedding"],
            device=config["data"]["device"],
        )
    else:
        print(
            f"DNA embeddings already exist at {dna_embeddings_dir}, skipping generation"
        )

    # 3) generate microbiome embeddings from dna embeddings
    if not microbiome_embeddings_dir.exists():
        dna_embeddings_h5_path = dna_embeddings_dir / "dna_embeddings.h5"
        generate_microbiome_embeddings_h5(
            dna_embeddings_h5_path,
            microbiome_embeddings_dir,
            config["data"]["microbiome_transformer_checkpoint"],
            device=config["data"]["device"],
        )
    else:
        print(
            f"Microbiome embeddings already exist at {microbiome_embeddings_dir}, skipping generation"
        )

    # sanity check for consistency on dna embeddings and microbiome embeddings
    if not sanity_check_dna_and_microbiome_embeddings(
        dna_embeddings_dir, microbiome_embeddings_dir
    ):
        raise ValueError("DNA embeddings and microbiome embeddings are not consistent")
    else:
        print("Data sanity check passed")

    # create dataframe relating labels from dataset csv and corresponding microbiome embeddings
    dataset_df = create_dataset_df(dataset_path, microbiome_embeddings_dir)
    print(
        f"Created dataset dataframe with {len(dataset_df)} rows and {len(dataset_df.columns)} columns"
    )
