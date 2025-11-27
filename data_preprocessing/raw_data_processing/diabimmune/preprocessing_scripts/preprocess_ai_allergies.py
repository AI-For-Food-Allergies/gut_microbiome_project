#!/usr/bin/env python3
"""
Preprocessing script for AI-for-Allergies.

Pipeline:
1. Read SRS list from download_samples_tsv-2.tsv
2. For each SRS, query ENA XML to get host_subject_id
3. Join SRS ↔ host_subject_id with metadata.csv on subjectID
4. Build binary allergy label using all allergy columns
5. Aggregate to (SRS, collection_month) and write:
   - ai_allergies_srs_labels_all_months.csv
   - Month_<m>.csv (sample_id,label)
"""

import os
import re
import time
import argparse
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_SRS_TSV = "download_samples_tsv-2.tsv"
DEFAULT_META_CSV = "metadata.csv"
DEFAULT_MAP_CSV = "srs_to_host_id.csv"
MASTER_OUT_CSV = "ai_allergies_srs_labels_all_months.csv"

# Allergy columns to use for label = 1 if ANY of these is positive
ALLERGY_COLS = [
    "allergy_milk",
    "allergy_egg",
    "allergy_peanut",
    "allergy_dustmite",
    "allergy_cat",
    "allergy_dog",
    "allergy_birch",
    "allergy_timothy",
    "totalige_high",  # include high IgE in "any allergy"
]

# Column that encodes time as age in months
TIME_COL = "collection_month"


# ---------------------------------------------------------------------
# ENA helper: SRS -> host_subject_id
# ---------------------------------------------------------------------

def get_host_subject_id(srs_id: str) -> str | None:
    """
    Use ENA browser XML API to get host_subject_id for a given SRS accession.

    This matches the "Original MetaData" you see in MicrobeAtlas UI.
    """
    url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{srs_id}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] SRS={srs_id}: {e}")
        return None

    xml = r.text

    # ENA attribute structure:
    # <SAMPLE_ATTRIBUTE>
    #   <TAG>host_subject_id</TAG>
    #   <VALUE>P018832</VALUE>
    # </SAMPLE_ATTRIBUTE>
    m = re.search(
        r"<TAG>\s*host_subject_id\s*</TAG>\s*<VALUE>\s*([^<\s]+)\s*</VALUE>",
        xml,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1)

    print(f"[WARN] no host_subject_id found in XML for {srs_id}")
    return None


# ---------------------------------------------------------------------
# Build / resume SRS -> host_subject_id map
# ---------------------------------------------------------------------

def build_srs_subject_map(
    srs_tsv: str,
    out_csv: str,
    max_workers: int = 16,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Build mapping SRS -> host_subject_id using ENA XML.

    - Reads SRS list from srs_tsv
    - Uses ThreadPoolExecutor to parallelize requests
    - Supports resume if out_csv already exists
    """
    print(f"[INFO] Loading SRS from {srs_tsv}")
    df_srs = pd.read_csv(srs_tsv, sep="\t", dtype=str)
    if "#sid" in df_srs.columns:
        df_srs = df_srs.rename(columns={"#sid": "SRS"})
    df_srs["SRS"] = df_srs["SRS"].astype(str)

    all_srs = df_srs["SRS"].dropna().unique().tolist()
    print(f"[INFO] Total unique SRS: {len(all_srs)}")

    if resume and os.path.exists(out_csv):
        print(f"[INFO] Resume enabled, loading existing map from {out_csv}")
        df_existing = pd.read_csv(out_csv, dtype=str)
        if "SRS" not in df_existing.columns:
            raise ValueError(f"{out_csv} must have a 'SRS' column")
        done = set(df_existing["SRS"].dropna())
        print(f"[INFO] Already mapped: {len(done)} SRS")
    else:
        df_existing = pd.DataFrame(columns=["SRS", "host_subject_id"])
        done = set()

    todo = [s for s in all_srs if s not in done]
    print(f"[INFO] Remaining to map: {len(todo)} SRS")

    if not todo:
        print("[INFO] Nothing left to do.")
        return df_existing

    def worker(srs: str) -> dict:
        subj = get_host_subject_id(srs)
        return {"SRS": srs, "host_subject_id": subj}

    rows = []
    batch_size = 200
    start = time.time()

    print(f"[INFO] Starting ENA XML mapping with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())

            if i % 50 == 0:
                elapsed = time.time() - start
                print(f"[PROGRESS] {i}/{len(todo)} new SRS in {elapsed:.1f}s")

            if i % batch_size == 0:
                df_new = pd.DataFrame(rows)
                df_existing = pd.concat([df_existing, df_new], ignore_index=True)
                df_existing.drop_duplicates(subset="SRS", inplace=True)
                df_existing.to_csv(out_csv, index=False)
                rows = []

    # Flush any remaining rows
    if rows:
        df_new = pd.DataFrame(rows)
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        df_existing.drop_duplicates(subset="SRS", inplace=True)

    df_existing.to_csv(out_csv, index=False)
    print(f"[INFO] Final SRS map saved to {out_csv} with {len(df_existing)} rows")

    return df_existing


# ---------------------------------------------------------------------
# Join with metadata and build labels
# ---------------------------------------------------------------------

def join_with_metadata_and_label(
    map_csv: str,
    metadata_csv: str,
    allergen_cols: list[str],
    time_col: str = TIME_COL,
    master_out: str = MASTER_OUT_CSV,
) -> pd.DataFrame:
    """
    - Read SRS->host_subject_id map and metadata.csv
    - Join on subjectID
    - Build 'label' = 1 if ANY allergy col is positive, else 0
    - Aggregate to one row per (SRS, subjectID, time_col)
    - Save master file and return DataFrame
    """
    print(f"[INFO] Loading SRS map from {map_csv}")
    df_map = pd.read_csv(map_csv, dtype=str)

    print(f"[INFO] Loading metadata from {metadata_csv}")
    df_meta = pd.read_csv(metadata_csv, dtype=str)

    # Normalize subjectID column name in metadata
    subj_col = None
    for c in df_meta.columns:
        if c.lower() in ("subjectid", "subject_id", "host_subject_id", "host_subjectid"):
            subj_col = c
            break

    if subj_col is None:
        raise ValueError(
            f"Could not find subjectID column in metadata. "
            f"Columns: {df_meta.columns.tolist()}"
        )

    df_meta = df_meta.rename(columns={subj_col: "subjectID"})
    if "host_subject_id" in df_map.columns:
        df_map = df_map.rename(columns={"host_subject_id": "subjectID"})

    print("[INFO] Joining SRS map with metadata on subjectID")
    df_merged = df_map.merge(df_meta, on="subjectID", how="left")
    print(f"[INFO] Merged columns: {df_merged.columns.tolist()}")

    # Sanity check allergy columns
    missing = [c for c in allergen_cols if c not in df_merged.columns]
    if missing:
        raise ValueError(f"Missing allergy columns in merged data: {missing}")

    # Build label
    def any_allergy(row) -> int:
        for col in allergen_cols:
            v = str(row[col]).strip().lower()
            if v in {"1", "true", "yes"}:
                return 1
        return 0

    print("[INFO] Computing binary 'label' from allergy columns")
    df_merged["label"] = df_merged.apply(any_allergy, axis=1)

    # Normalize time column
    if time_col not in df_merged.columns:
        raise ValueError(
            f"Time column '{time_col}' not found in merged data. "
            f"Columns: {df_merged.columns.tolist()}"
        )

    df_merged = df_merged.dropna(subset=[time_col, "SRS"]).copy()
    df_merged[time_col] = (
        pd.to_numeric(df_merged[time_col], errors="coerce")
        .round()
        .astype("Int64")
    )
    df_merged = df_merged.dropna(subset=[time_col]).copy()
    df_merged[time_col] = df_merged[time_col].astype(int)

    print("[INFO] Aggregating to one row per (SRS, subjectID, time)")
    agg = (
        df_merged
        .groupby(["SRS", "subjectID", time_col], as_index=False)
        .agg(label=("label", "max"))  # any positive => 1
    )

    agg.to_csv(master_out, index=False)
    print(f"[INFO] Master labeled file written: {master_out}")
    print(agg.head())

    return agg


# ---------------------------------------------------------------------
# Write per-month sample_id,label CSVs
# ---------------------------------------------------------------------

def write_per_month_files(
    agg: pd.DataFrame,
    time_col: str = TIME_COL,
    prefix: str = "Month_",
):
    """
    For each unique time (month) in agg, write a CSV:
        Month_<time>.csv with columns: sample_id,label
    where sample_id is SRS.
    """
    print("[INFO] Writing per-month sample_id,label CSVs")

    for t, grp in agg.groupby(time_col):
        out = grp[["SRS", "label"]].drop_duplicates().copy()
        out = out.rename(columns={"SRS": "sample_id"})
        fname = f"{prefix}{t}.csv"
        out.to_csv(fname, index=False)
        print(f"[INFO] Wrote {fname} with {len(out)} rows")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MicrobeAtlas SRS + DIABIMMUNE metadata for AI-for-Allergies."
    )
    parser.add_argument(
        "--srs-tsv",
        default=DEFAULT_SRS_TSV,
        help=f"TSV file with SRS list (default: {DEFAULT_SRS_TSV})",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_META_CSV,
        help=f"Metadata CSV with subjectID and allergy info (default: {DEFAULT_META_CSV})",
    )
    parser.add_argument(
        "--map-csv",
        default=DEFAULT_MAP_CSV,
        help=f"Output CSV for SRS->host_subject_id mapping (default: {DEFAULT_MAP_CSV})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of threads to use for ENA XML requests (default: 16)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing mapping CSV; rebuild from scratch.",
    )

    args = parser.parse_args()

    df_map = build_srs_subject_map(
        srs_tsv=args.srs_tsv,
        out_csv=args.map_csv,
        max_workers=args.max_workers,
        resume=not args.no_resume,
    )

    agg = join_with_metadata_and_label(
        map_csv=args.map_csv,
        metadata_csv=args.metadata,
        allergen_cols=ALLERGY_COLS,
        time_col=TIME_COL,
        master_out=MASTER_OUT_CSV,
    )

    write_per_month_files(agg, time_col=TIME_COL, prefix="Month_")


if __name__ == "__main__":
    main()