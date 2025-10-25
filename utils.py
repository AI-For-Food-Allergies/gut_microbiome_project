import csv
from pathlib import Path
from typing import Dict, Any

import h5py
import torch
import yaml
from tqdm import tqdm

from model import MicrobiomeTransformer


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# Load default configuration
_DEFAULT_CONFIG = load_config()


def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    return _DEFAULT_CONFIG


# Model architecture constants (hardcoded - not configurable)
D_MODEL = 100
NHEAD = 5
NUM_LAYERS = 5
DROPOUT = 0.1
DIM_FF = 400
OTU_EMB = 384
TXT_EMB = 1536


def load_run_data(
    wgs_path=None,
    extra_path=None,
    samples_path=None,
    microbeatlas_path=None,
    config=None,
):
    """
    Load and merge run data from multiple sources.
    
    Args:
        wgs_path: Path to WGS run table (default from config)
        extra_path: Path to extra run table (default from config)
        samples_path: Path to samples table (default from config)
        microbeatlas_path: Path to microbeatlas samples (default from config)
        config: Configuration dictionary (default: global config)
        
    Returns:
        Tuple of (run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample)
    """
    if config is None:
        config = get_config()
    
    wgs_path = wgs_path or config['data']['wgs_run_table']
    extra_path = extra_path or config['data']['extra_run_table']
    samples_path = samples_path or config['data']['samples_table']
    microbeatlas_path = microbeatlas_path or config['data']['microbeatlas_samples']
    run_rows = {}
    SRA_to_micro = {}
    gid_to_sample = {}
    micro_to_subject = {}
    micro_to_sample = {}

    for path in (wgs_path, extra_path):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = row['Run'].strip()
                library_name = row.get('Library Name', '').strip()
                subject_id = row.get('host_subject_id', row.get('host_subject_id (run)', '')).strip()
                run_rows[run_id] = {'library': library_name, 'subject': subject_id}

    with open(microbeatlas_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            srs = row['#sid']
            rids = row['rids']
            if not rids:
                continue
            for rid in rids.replace(';', ',').split(','):
                rid = rid.strip()
                if rid:
                    SRA_to_micro[rid] = srs

    with open(samples_path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            pieces = line.split(',')
            if header is None:
                header = pieces
                header[0] = header[0].lstrip('\ufeff')
                continue
            row = dict(zip(header, pieces))
            for key in ('gid_wgs', 'gid_16s'):
                gid = row.get(key, '').strip()
                if gid:
                    gid_to_sample[gid] = {'subject': row['subjectID'], 'sample': row['sampleID']}

    for run_id, srs in SRA_to_micro.items():
        run_info = run_rows.get(run_id, {})
        library_name = run_info.get('library', '')
        if library_name and library_name in gid_to_sample:
            micro_to_sample[srs] = gid_to_sample[library_name]
            micro_to_subject[srs] = gid_to_sample[library_name]['subject']
        elif run_info.get('subject'):
            micro_to_subject[srs] = run_info['subject']

    return run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample


def collect_micro_to_otus(
    SRA_to_micro,
    micro_to_subject,
    biom_path=None,
    config=None,
):
    """
    Collect OTU data for microbiome samples.
    
    Args:
        SRA_to_micro: Mapping from SRA run IDs to microbiome sample IDs
        micro_to_subject: Mapping from microbiome sample IDs to subject IDs
        biom_path: Path to BIOM file (default from config)
        config: Configuration dictionary (default: global config)
        
    Returns:
        Dictionary mapping microbiome sample IDs to lists of OTU IDs
    """
    if config is None:
        config = get_config()
    
    biom_path = biom_path or config['data']['biom_path']
    micro_to_otus = {}
    needed_srs = set(SRA_to_micro.values()) | set(micro_to_subject.keys())

    with h5py.File(biom_path) as biom_file:
        observation_names = [
            oid.decode('utf-8') if isinstance(oid, bytes) else str(oid)
            for oid in biom_file['observation/ids'][:]
        ]
        sample_ids = biom_file['sample/ids']
        sample_indices = biom_file['sample/matrix/indices']
        sample_indptr = biom_file['sample/matrix/indptr'][:]
        for idx in range(len(sample_ids)):
            sample_entry = sample_ids[idx]
            sample_entry = sample_entry.decode('utf-8') if isinstance(sample_entry, bytes) else str(sample_entry)
            sample_key = sample_entry.split('.')[-1]
            if sample_key not in needed_srs:
                continue
            start = sample_indptr[idx]
            end = sample_indptr[idx + 1]
            otu_list = [observation_names[i] for i in sample_indices[start:end]]
            micro_to_otus[sample_key] = otu_list
            if len(micro_to_otus) == len(needed_srs):
                break

    print('otus mapped for', len(micro_to_otus), 'samples')
    missing_srs = needed_srs - set(micro_to_otus)
    if missing_srs:
        print('missing from biom:', len(missing_srs))
    if micro_to_otus:
        example_key = next(iter(micro_to_otus))
        print('example', example_key, 'otus:', micro_to_otus[example_key][:5])

    return micro_to_otus


def load_microbiome_model(checkpoint_path=None, config=None):
    """
    Load pre-trained microbiome transformer model.
    
    Args:
        checkpoint_path: Path to model checkpoint (default from config)
        config: Configuration dictionary (default: global config)
        
    Returns:
        Tuple of (model, device)
    """
    if config is None:
        config = get_config()
    
    checkpoint_path = checkpoint_path or config['data']['checkpoint_path']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    model = MicrobiomeTransformer(
        input_dim_type1=OTU_EMB,
        input_dim_type2=TXT_EMB,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT
    )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print('model ready on', device)

    return model, device


def preview_prokbert_embeddings(prokbert_path=None, limit=10, config=None):
    """
    Preview available ProkBERT embeddings.
    
    Args:
        prokbert_path: Path to ProkBERT embeddings file (default from config)
        limit: Maximum number of example IDs to show
        config: Configuration dictionary (default: global config)
        
    Returns:
        List of example embedding IDs
    """
    if config is None:
        config = get_config()
    
    prokbert_path = prokbert_path or config['data']['prokbert_path']
    with h5py.File(prokbert_path) as emb_file:
        embedding_group = emb_file['embeddings']
        example_ids = []
        for key in embedding_group.keys():
            example_ids.append(key)
            if len(example_ids) == limit:
                break
        print('total prokbert embeddings:', len(embedding_group))
        print('example embedding ids:', example_ids)

    return example_ids


def build_sample_embeddings(
    micro_to_otus,
    model,
    device,
    prokbert_path=None,
    txt_emb=None,
    config=None,
):
    """
    Build sample embeddings from OTU embeddings.
    
    Args:
        micro_to_otus: Dictionary mapping microbiome sample IDs to OTU lists
        model: Pre-trained microbiome transformer model
        device: PyTorch device
        prokbert_path: Path to ProkBERT embeddings file (default from config)
        txt_emb: Text embedding dimension (hardcoded constant if not provided)
        config: Configuration dictionary (default: global config)
        
    Returns:
        Tuple of (sample_embeddings dict, missing_otus count)
    """
    if config is None:
        config = get_config()
    
    prokbert_path = prokbert_path or config['data']['prokbert_path']
    txt_emb = txt_emb or TXT_EMB
    sample_embeddings = {}
    missing_otus = 0

    with h5py.File(prokbert_path) as emb_file:
        embedding_group = emb_file['embeddings']
        for sample_key, otu_list in tqdm(micro_to_otus.items(), desc='Embedding samples'):
            otu_vectors = []
            for otu_id in otu_list:
                if otu_id in embedding_group:
                    vec = embedding_group[otu_id][()]
                    otu_vectors.append(torch.tensor(vec, dtype=torch.float32, device=device))
                else:
                    missing_otus += 1
            if not otu_vectors:
                continue
            otu_tensor = torch.stack(otu_vectors, dim=0).unsqueeze(0)
            type2_tensor = torch.zeros((1, 0, txt_emb), dtype=torch.float32, device=device)
            mask = torch.ones((1, otu_tensor.shape[1]), dtype=torch.bool, device=device)
            with torch.no_grad():
                hidden_type1 = model.input_projection_type1(otu_tensor)
                hidden_type2 = model.input_projection_type2(type2_tensor)
                combined_hidden = torch.cat([hidden_type1, hidden_type2], dim=1)
                hidden = model.transformer(combined_hidden, src_key_padding_mask=~mask)
                sample_vec = hidden.mean(dim=1).squeeze(0).cpu()
            sample_embeddings[sample_key] = sample_vec

    print('sample embeddings ready:', len(sample_embeddings))
    print('missing otu embeddings:', missing_otus)
    if sample_embeddings:
        first_key = next(iter(sample_embeddings))
        print('example sample embedding', first_key, sample_embeddings[first_key][:5])

    return sample_embeddings, missing_otus
