import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import h5py
import os
from tqdm import tqdm

DNA_SEQUENCES_DIR = "dna_sequences" # where SRS to OTU mapping is stored
OUTPUT_PATH = "dna_embeddings/prokbert_embeddings.h5" # where the embeddings will be stored
BATCH_SIZE = 32 # number of sequences to process at once
MODEL_NAME = 'neuralbioinfo/prokbert-mini-long' # model to use for embeddings
DEVICE = 'cpu' # device to use for embeddings [cpu, cuda]

def process_dna_sequences_to_hdf5(
    dna_sequences_dir,
    output_path,
    tokenizer,
    model,
    device,
    batch_size=32
):
    """
    Process DNA sequences from CSV files and create hierarchical HDF5 file.
    
    Structure:
    - SRS_ID (group)
      - OTU_ID (dataset with shape (384,))
        - DNA embedding vector
    
    Args:
        dna_sequences_dir: Directory containing CSV files (one per SRS ID)
        output_path: Path to output HDF5 file
        tokenizer: ProkBERT tokenizer
        model: ProkBERT model
        device: torch device
        batch_size: Number of sequences to process at once
    """
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(dna_sequences_dir) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Create/open HDF5 file
    with h5py.File(output_path, 'w') as hdf5_file:
        
        # Process each CSV file (one per SRS ID)
        for csv_file in tqdm(csv_files, desc="Processing SRS samples"):
            # Extract SRS ID from filename (e.g., "SRS7011253.csv" -> "SRS7011253")
            srs_id = csv_file.replace('.csv', '')
            
            # Read the CSV file
            csv_path = os.path.join(dna_sequences_dir, csv_file)
            table = pd.read_csv(csv_path)
            
            # Create group for this SRS ID
            srs_group = hdf5_file.create_group(srs_id)
            
            # Process sequences in batches
            num_sequences = len(table)
            
            for batch_start in tqdm(range(0, num_sequences, batch_size), 
                                   desc=f"  Batches for {srs_id}", 
                                   leave=False):
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
                    ).to(device)
                    
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
    
    print(f"\nComplete! Saved to {output_path}")
    print(f"Structure: SRS_ID/OTU_ID → embedding vector (384,)")

# Verify the hierarchical HDF5 structure
def inspect_hierarchical_hdf5(hdf5_path):
    """
    Inspect the hierarchical HDF5 file structure.
    """
    with h5py.File(hdf5_path, 'r') as f:
        print(f"📁 File: {hdf5_path}")
        print(f"📊 Number of SRS samples: {len(f.keys())}\n")
        
        # Show structure for each SRS sample
        for srs_id in list(f.keys())[:3]:  # Show first 3 samples
            srs_group = f[srs_id]
            num_otus = len(srs_group.keys())
            print(f"  📂 {srs_id}/")
            print(f"     └─ {num_otus} OTUs")
            
            # Show first few OTUs
            otu_list = list(srs_group.keys())[:3]
            for otu_id in otu_list:
                embedding = srs_group[otu_id]
                print(f"        └─ {otu_id}: shape={embedding.shape}, dtype={embedding.dtype}")
            
            if num_otus > 3:
                print(f"        └─ ... ({num_otus - 3} more OTUs)")
            print()
        
        if len(f.keys()) > 3:
            print(f"  ... ({len(f.keys()) - 3} more SRS samples)\n")
        
        # Example: How to access data
        print("\n💡 Example access:")
        first_srs = list(f.keys())[0]
        first_otu = list(f[first_srs].keys())[0]
        print(f"   embedding = f['{first_srs}']['{first_otu}'][:]")
        print(f"   Result shape: {f[first_srs][first_otu].shape}")
        print(f"   First 5 values: {f[first_srs][first_otu][:5]}")

if __name__ == "__main__": 
    device = torch.device(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    model = model.to(device)
    model.eval()
    
    # Generate DNA embeddings for all SRS sample
    process_dna_sequences_to_hdf5(
        dna_sequences_dir=DNA_SEQUENCES_DIR,
        output_path=OUTPUT_PATH,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=BATCH_SIZE  # Adjust based on your memory
    )

    # Run inspection after processing
    inspect_hierarchical_hdf5(OUTPUT_PATH)
