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

from modules.model import MicrobiomeTransformer

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
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_default_paths(config: dict = None) -> dict:
    """Get default paths for data files."""
    if config is None:
        config = load_config()
    
    data_config = config.get('data', {})
    
    return {
        'srs_to_otu_parquet': Path(data_config.get(
            'srs_to_otu_parquet', 
            'data_preprocessing/mapref_data/samples-otus-97.parquet'
        )),
        'otu_to_dna_parquet': Path(data_config.get(
            'otu_to_dna_parquet',
            'data_preprocessing/mapref_data/otus_97_to_dna.parquet'
        )),
        'dna_csv_dir': Path(data_config.get(
            'dna_csv_dir',
            'data_preprocessing/dna_sequences'
        )),
        'embeddings_h5': Path(data_config.get(
            'embeddings_h5',
            'data_preprocessing/dna_embeddings/prokbert_embeddings.h5'
        )),
        'microbiome_embeddings_h5':Path(data_config.get(
            'microbiome_embeddings_h5',
            'data_preprocessing/microbiome_embeddings/microbiome_embeddings.h5'
        )),
        'checkpoint_path': Path(data_config.get(
            'checkpoint_path',
            'data/checkpoint_epoch_0_final_epoch3_conf00.pt'
        )),
        'model_name': data_config.get(
            'embedding_model',
            'neuralbioinfo/prokbert-mini-long'
        ),
        'batch_size_embedding': data_config.get('batch_size_embedding', 32),
        'device': data_config.get('device', 'cpu'),
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
    filters = [('srs_id', '=', srs_id)]
    table = pq.read_table(srs_to_otu_parquet, filters=filters)
    if len(table) == 0:
        return []
    return table.to_pandas()['otu_id'].tolist()


def get_dna_from_otu(otu_id: str, otu_to_dna_parquet: Path) -> Optional[str]:
    """
    Get DNA sequence for a given OTU ID from parquet file.
    
    Args:
        otu_id: OTU ID
        otu_to_dna_parquet: Path to parquet file mapping otu_97_id -> dna_sequence
        
    Returns:
        DNA sequence string, or None if not found
    """
    filters = [('otu_97_id', '=', otu_id)]
    table = pq.read_table(otu_to_dna_parquet, filters=filters)
    if len(table) == 0:
        return None
    return table.to_pandas().iloc[0]['dna_sequence']


def generate_dna_csv_for_sample(
    srs_id: str,
    srs_to_otu_parquet: Path,
    otu_to_dna_parquet: Path,
    output_dir: Path
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
        pd.DataFrame(columns=['otu_id', 'dna_sequence']).to_csv(csv_path, index=False)
        return csv_path
    
    # Get DNA sequences for each OTU
    srs_dna_map = {}
    for otu in tqdm(otus, desc=f"  Processing {srs_id}", leave=False):
        dna_seq = get_dna_from_otu(otu, otu_to_dna_parquet)
        if dna_seq is not None:
            srs_dna_map[otu] = dna_seq
    
    # Save to CSV
    df = pd.DataFrame(srs_dna_map.items(), columns=['otu_id', 'dna_sequence'])
    df.to_csv(csv_path, index=False)
    
    return csv_path


def generate_dna_csvs_for_samples(
    sids: List[str],
    srs_to_otu_parquet: Path,
    otu_to_dna_parquet: Path,
    output_dir: Path
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


def generate_embeddings_h5(
    dna_csv_dir: Path,
    output_h5_path: Path,
    model_name: str = 'neuralbioinfo/prokbert-mini-long',
    batch_size: int = 32,
    device: str = 'cpu'
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
    csv_files = [f for f in os.listdir(dna_csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {dna_csv_dir}")
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Load model
    device_obj = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device_obj)
    model.eval()
    
    # Create output directory if needed
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create/open HDF5 file
    with h5py.File(output_h5_path, 'w') as hdf5_file:
        # Process each CSV file (one per SRS ID)
        for csv_file in tqdm(csv_files, desc="Processing SRS samples"):
            # Extract SRS ID from filename
            srs_id = csv_file.replace('.csv', '')
            
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
                batch_sequences = batch_df['dna_sequence'].tolist()
                batch_otu_ids = batch_df['otu_id'].tolist()
                
                # Tokenize batch
                with torch.no_grad():
                    inputs = tokenizer(
                        batch_sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device_obj)
                    
                    # Get embeddings
                    outputs = model(**inputs)
                    # Mean pooling over sequence length
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                # Save each embedding with its OTU ID
                for i, otu_id in enumerate(batch_otu_ids):
                    # Replace forward slashes in OTU IDs if any (HDF5 key compatibility)
                    otu_key = str(otu_id).replace('/', '_')
                    srs_group.create_dataset(otu_key, data=embeddings[i])
            
            print(f"  ✓ Processed {num_sequences} OTUs for {srs_id}")
    
    print(f"\nComplete! Saved embeddings to {output_h5_path}")
    return output_h5_path


def ensure_embeddings_exist(
    sample_csv_path: Path,
    paths: dict = None,
    force_regenerate: bool = False,
    check_samples_match: bool = True
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
    
    embeddings_h5 = paths['embeddings_h5']
    
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
                print(f"\nWarning: H5 file exists but contains different sample IDs!")
                print(f"  Expected {len(sids)} samples from CSV, found {len(h5_samples_set)} in H5")
                print(f"  No matching samples found.")
                print(f"  Solution: Delete {embeddings_h5} and regenerate, or use force_regenerate=True")
                # Don't raise error here - let load_dataset handle it with better diagnostics
            elif len(missing) > 0:
                print(f"  Found {len(matching)}/{len(sids)} matching samples in H5")
                print(f"  Missing {len(missing)} samples - will generate missing embeddings")
        
        return embeddings_h5
    
    print("Embeddings H5 not found or regeneration requested. Generating...")
    
    # Check which DNA CSVs exist
    dna_csv_dir = paths['dna_csv_dir']
    existing_csvs = set(f.replace('.csv', '') for f in os.listdir(dna_csv_dir) 
                       if f.endswith('.csv')) if dna_csv_dir.exists() else set()
    
    # Generate missing DNA CSVs
    missing_sids = [sid for sid in sids if sid not in existing_csvs]
    if missing_sids:
        print(f"Generating {len(missing_sids)} missing DNA CSV files...")
        generate_dna_csvs_for_samples(
            missing_sids,
            paths['srs_to_otu_parquet'],
            paths['otu_to_dna_parquet'],
            paths['dna_csv_dir']
        )
    else:
        print("All DNA CSV files already exist")
    
    # Generate embeddings H5
    print("Generating embeddings H5 file...")
    generate_embeddings_h5(
        paths['dna_csv_dir'],
        embeddings_h5,
        paths['model_name'],
        paths['batch_size_embedding'],
        paths['device']
    )
    
    return embeddings_h5


# ==================== Data Loading Functions ====================

def load_labels(
    labels_csv: Path,
    id_col: str = "sid",
    label_col: str = "label"
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
        for col in ['SID', 'sid', 'srs_id', 'SRS_ID']:
            if col in df.columns:
                id_col = col
                break
        else:
            # If still not found, use first column
            id_col = df.columns[0]
            print(f"Warning: ID column not found, using '{id_col}'")
    
    if label_col not in df.columns:
        # Try common variations
        for col in ['label', 'Label', 'LABEL', 'y']:
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
        with h5py.File(embeddings_h5, 'r') as f:
            all_samples = list(f.keys())
            if max_samples is None or max_samples >= len(all_samples):
                return all_samples
            return all_samples[:max_samples]
    except Exception as e:
        print(f"Error inspecting H5 file: {e}")
        return []


def load_embeddings_for_sample(
    embeddings_h5: Path,
    srs_id: str
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
        with h5py.File(embeddings_h5, 'r') as f:
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
    sample_csv_path: Path,
    microbiome_h5_path: Optional[Path] = None,
    paths: dict = None
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
        raise FileNotFoundError(f"Microbiome embeddings H5 file not found: {microbiome_h5_path}")

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
        print(f"\nWarning: {len(missing_sids)} samples from CSV "
              f"are missing in microbiome embeddings H5.")
        print(f"  Example missing SIDs: {missing_sids[:5]}")

    if not X_list:
        raise ValueError(
            "No samples were loaded from microbiome_embeddings.h5. "
            "Check that the SIDs in your CSV match the H5 file."
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print(f"Loaded {X.shape[0]} microbiome sample embeddings "
          f"from {microbiome_h5_path}")

    return X, y, sids_list



def load_dataset(
    sample_csv_path: Path,
    embeddings_h5_path: Optional[Path] = None,
    paths: dict = None
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
            print(f"  H5 file appears to be empty or corrupted")
        
        # Check if all samples are missing
        if len(X_list) == 0:
            print(f"\nError: No samples loaded! All {len(missing_samples)} samples are missing.")
            print(f"  This usually means the embeddings H5 was generated for different sample IDs.")
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
        paths: dict = None
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
        'embeddings': padded_embeddings,
        'labels': labels,
        'mask': mask,
        'sids': list(sids_list)
    }


def get_dataloader(
    sample_csv_path: Path,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    embeddings_h5_path: Optional[Path] = None,
    paths: dict = None
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
        print(f"Warning: batch_size ({batch_size}) > dataset size ({len(dataset)}). "
              f"Setting batch_size to {len(dataset)}")
        batch_size = len(dataset)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
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
    # Exit if already exists
    if output_h5_path.exists():
        print(f"Microbiome embeddings H5 already exists at {output_h5_path}")
        return output_h5_path

    # Check input H5 exists
    if not prokbert_h5_path.exists():
        raise FileNotFoundError(
            f"ProkBERT embeddings H5 not found: {prokbert_h5_path}"
        )

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

    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(prokbert_h5_path, "r") as h_in:
        sample_ids = list(h_in.keys())
        if not sample_ids:
            raise ValueError(
                f"ProkBERT H5 file {prokbert_h5_path} contains no sample groups."
            )

        with h5py.File(output_h5_path, "w") as h_out:
            for srs_id in tqdm(sample_ids, desc="Building microbiome sample embeddings"):
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
                    sample_vec = hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (D_MODEL,)

                h_out.create_dataset(srs_id, data=sample_vec)

    print(f"\nSaved microbiome sample embeddings to {output_h5_path}")
    return output_h5_path




# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example usage
    sample_csv = Path("data_preprocessing/datasets/sample_data.csv")
    
    try:
        print("Creating DataLoader...")
        dataloader = get_dataloader(
            sample_csv,
            batch_size=4,
            shuffle=True
        )
        
        print(f"\nDataLoader created with {len(dataloader.dataset)} samples")
        
        # Test a batch
        for batch in dataloader:
            print(f"\nBatch shapes:")
            print(f"  embeddings: {batch['embeddings'].shape}")
            print(f"  labels: {batch['labels'].shape}")
            print(f"  mask: {batch['mask'].shape}")
            print(f"  sids: {batch['sids']}")
            break
    except FileNotFoundError as e:
        print(f"\nFile Not Found Error:")
        print(f"{e}")
        print("\nTo fix this issue:")
        print("  1. Check that the CSV file path is correct")
        print("  2. Ensure the file exists at the specified location")
        print("  3. Use absolute paths if relative paths are causing issues")
    except ValueError as e:
        print(f"\nValue Error: {e}")
        print("\nTo fix this issue:")
        print("  1. Check that your sample CSV contains the correct sample IDs")
        print("  2. Ensure embeddings H5 contains matching sample IDs")
        print("  3. If needed, delete the H5 file and regenerate embeddings:")
        print(f"     rm data/prokbert_embeddings.h5")
        print("     Then run this script again to auto-generate embeddings")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

    # === MICROBIOME EMBEDDINGS: build microbiome embeddings AFTER PROKBERT H5 ===
    try:
        paths = get_default_paths()
        print("\nBuilding microbiome sample embeddings from ProkBERT H5...")
        generate_microbiome_embeddings_h5(
            prokbert_h5_path=paths["embeddings_h5"],
            output_h5_path=paths["microbiome_embeddings_h5"],
            checkpoint_path=paths["checkpoint_path"],
            device=paths["device"],
        )
    except Exception as e:
        print("\nError while generating microbiome embeddings:")
        print(e)
        import traceback
        traceback.print_exc()
