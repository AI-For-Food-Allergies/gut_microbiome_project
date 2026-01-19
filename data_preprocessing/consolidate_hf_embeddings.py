"""Consolidate per-month HuggingFace embeddings into a single H5 file.

This script reads the per-month microbiome_embeddings.h5 files from the HuggingFace
cache and consolidates them into a single file keyed by SRS ID.

The HF dataset has the same sample appearing in multiple Month_N folders with
potentially different embeddings. We use Ludo's corrected metadata to determine
which month each sample actually belongs to.

Usage:
    python data_preprocessing/consolidate_hf_embeddings.py

Requirements:
    - HuggingFace dataset must be downloaded first:
      huggingface-cli download hugging-science/AI4FA-Diabimmune --repo-type dataset
    - Ludo's corrected metadata in:
      data_preprocessing/datasets_preprocessing_scripts/diabimmune/preprocessing_scripts/preprocessed_diabimmune_longitudinal/Month_*.csv

Output:
    - data_preprocessing/microbiome_embeddings/microbiome_embeddings_100d.h5
    - data_preprocessing/microbiome_embeddings/dataset_manifest.json (updated)
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# HuggingFace dataset info
HF_REPO_ID = "hugging-science/AI4FA-Diabimmune"
HF_REVISION = "7761eea93dad5712a03452786b43031dc9b04233"

# Expected dimensions
EMBEDDING_DIM = 100


def find_hf_cache_path() -> Path:
    """Find the HuggingFace cache path for the dataset."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    dataset_dir = hf_cache / f"datasets--{HF_REPO_ID.replace('/', '--')}"
    snapshot_dir = dataset_dir / "snapshots" / HF_REVISION
    embeddings_dir = snapshot_dir / "processed" / "microbiome_embeddings"

    if not embeddings_dir.exists():
        raise FileNotFoundError(
            f"HuggingFace cache not found at {embeddings_dir}.\n"
            f"Run: huggingface-cli download {HF_REPO_ID} --repo-type dataset"
        )
    return embeddings_dir


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sample_to_month_mapping(metadata_dir: Path) -> dict[str, int]:
    """Load Ludo's corrected metadata to get sample -> month mapping."""
    sample_to_month = {}
    for csv_path in sorted(metadata_dir.glob("Month_*.csv")):
        month = int(csv_path.stem.split("_")[1])
        df = pd.read_csv(csv_path)
        for sid in df["sid"]:
            if sid in sample_to_month:
                raise ValueError(f"Sample {sid} appears in multiple month files!")
            sample_to_month[sid] = month
    return sample_to_month


def consolidate_embeddings(
    hf_embeddings_dir: Path,
    output_path: Path,
    sample_to_month: dict[str, int],
) -> dict:
    """Consolidate per-month H5 files into single file.

    For each sample, pull the embedding from its correct Month_N file
    (as determined by Ludo's corrected metadata). If not found in the
    designated month, fall back to searching all months.

    Returns dict with stats about what was consolidated.
    """
    all_embeddings = {}
    samples_from_designated = 0
    samples_from_fallback = 0
    samples_missing = []

    # Get all available month directories
    all_month_dirs = sorted(hf_embeddings_dir.glob("Month_*"))

    for srs_id, month in sample_to_month.items():
        embedding = None
        source_month = None

        # First try the designated month
        month_dir = hf_embeddings_dir / f"Month_{month}"
        h5_path = month_dir / "microbiome_embeddings.h5"

        if h5_path.exists():
            resolved_path = h5_path.resolve() if h5_path.is_symlink() else h5_path
            with h5py.File(resolved_path, "r") as f:
                if srs_id in f:
                    embedding = f[srs_id][:]
                    source_month = month
                    samples_from_designated += 1

        # Fallback: search all months if not found in designated
        if embedding is None:
            for fallback_dir in all_month_dirs:
                fb_h5_path = fallback_dir / "microbiome_embeddings.h5"
                if not fb_h5_path.exists():
                    continue
                resolved_path = fb_h5_path.resolve() if fb_h5_path.is_symlink() else fb_h5_path
                with h5py.File(resolved_path, "r") as f:
                    if srs_id in f:
                        embedding = f[srs_id][:]
                        source_month = int(fallback_dir.name.split("_")[1])
                        samples_from_fallback += 1
                        break

        if embedding is None:
            samples_missing.append((srs_id, month, "not_found_anywhere"))
            continue

        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Unexpected shape {embedding.shape} for {srs_id} in Month_{source_month}"
            )
        all_embeddings[srs_id] = embedding

    # Write consolidated file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for srs_id, embedding in sorted(all_embeddings.items()):
            f.create_dataset(srs_id, data=embedding.astype(np.float32))

    return {
        "n_samples": len(all_embeddings),
        "n_expected": len(sample_to_month),
        "n_from_designated_month": samples_from_designated,
        "n_from_fallback": samples_from_fallback,
        "n_missing": len(samples_missing),
        "missing_samples": samples_missing[:10] if samples_missing else [],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data_preprocessing" / "microbiome_embeddings"
    output_h5 = output_dir / "microbiome_embeddings_100d.h5"
    manifest_path = output_dir / "dataset_manifest.json"
    metadata_dir = (
        project_root
        / "data_preprocessing"
        / "datasets_preprocessing_scripts"
        / "diabimmune"
        / "preprocessing_scripts"
        / "preprocessed_diabimmune_longitudinal"
    )

    print(f"Loading sample-to-month mapping from Ludo's metadata...")
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {metadata_dir}\n"
            "Need Ludo's corrected Month_*.csv files."
        )
    sample_to_month = load_sample_to_month_mapping(metadata_dir)
    print(f"Found {len(sample_to_month)} samples in metadata")

    print(f"Looking for HuggingFace cache...")
    hf_embeddings_dir = find_hf_cache_path()
    print(f"Found: {hf_embeddings_dir}")

    print(f"Consolidating embeddings...")
    stats = consolidate_embeddings(hf_embeddings_dir, output_h5, sample_to_month)
    print(f"Consolidated {stats['n_samples']} samples (expected {stats['n_expected']})")
    print(f"  - {stats['n_from_designated_month']} from designated month")
    print(f"  - {stats['n_from_fallback']} from fallback search")
    if stats['n_missing'] > 0:
        print(f"Warning: {stats['n_missing']} samples missing from HF cache")
        for srs_id, month, reason in stats['missing_samples']:
            print(f"  - {srs_id} (Month_{month}): {reason}")

    # Compute hash of output
    output_hash = sha256_file(output_h5)

    # Load existing manifest or create new one
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"Updating existing manifest...")
    else:
        manifest = {}
        print(f"Creating new manifest...")

    # Update manifest
    manifest.update({
        "created_at": datetime.now(UTC).isoformat(),
        "sources": {
            "huggingface": {
                "repo_id": HF_REPO_ID,
                "revision": HF_REVISION,
                "allow_patterns": ["processed/microbiome_embeddings/Month_*/microbiome_embeddings.h5"],
            },
        },
        "embedding_export": {
            "output_h5": str(output_h5.relative_to(project_root)),
            "rule": "for each sample, use the vector from its Month_N file (all months consolidated)",
            "dims": EMBEDDING_DIM,
        },
        "dataset": {
            "n_samples": stats["n_samples"],
            "n_expected": stats["n_expected"],
            "n_missing": stats["n_missing"],
        },
        "artifacts": {
            "microbiome_embeddings_h5": {
                "path": str(output_h5.relative_to(project_root)),
                "sha256": output_hash,
            },
        },
    })

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {output_h5}")
    print(f"Wrote {manifest_path}")
    print(f"SHA256: {output_hash}")
    print(f"Done.")


if __name__ == "__main__":
    main()
