#%%
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

raw_data_path = 'data.tsv'
model_name = 'neuralbioinfo/prokbert-mini-long'
device = 'cpu' # cpu, cuda, mps

# Set device to CPU explicitly
device = torch.device(device)

# Load tokenizer and model for the long context version
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Move model to CPU and set to evaluation mode
model = model.to(device)
model.eval()

# Load the data
otus = pd.read_csv(raw_data_path, sep='\t')

# Filter the dataframe to keep only representative sequences
representative_seqs = otus[otus['is_repres'] == 1]
#%%
# Create H5 file to store embeddings
with h5py.File('prokbert_embeddings.h5', 'w') as h5f:
    # Pre-allocate a group for all embeddings
    emb_group = h5f.create_group('embeddings')
    
    # Process sequences in batches for better efficiency
    batch_size = 32  # Adjust based on your available memory
    
    for i in tqdm(range(0, len(representative_seqs), batch_size), desc="Processing batches"):
        batch = representative_seqs.iloc[i:i+batch_size]
        sequences = batch['seq'].tolist()
        seq_ids = batch['oid'].astype(str).tolist()
        
        try:
            # Tokenize the batch of sequences
            with torch.no_grad():
                inputs = tokenizer(sequences, padding=True, 
                                   return_tensors="pt").to(device)
                
                # Get model output
                outputs = model(**inputs)
                
                # Extract embeddings (mean of last hidden state)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                # Store each embedding in the H5 file
                for j, seq_id in enumerate(seq_ids):
                    emb_group.create_dataset(seq_id, data=embeddings[j], 
                                           compression="gzip", compression_opts=4)
        
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            # Continue with next batch

print("Processing complete. Compressed embeddings saved to 'prokbert_embeddings.h5'")