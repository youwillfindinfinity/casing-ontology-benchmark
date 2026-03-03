"""
build_synonym_eval.py
---------------------
Extracts synonym→canonical name pairs from NCBI Taxonomy names.dmp.

Downloads taxdump.tar.gz from NCBI FTP (~50 MB), extracts names.dmp,
then builds paired queries:
    query        : synonym / equivalent name / common name
    canonical    : scientific name (what the FAISS index stores)
    ncbi_taxid   : NCBI TaxID (ground truth)
    synonym_type : synonym | equivalent name | common name | ...

Outputs:
    OUTPUTS/ncbi_synonym_eval_setB.csv  — title-case queries
    OUTPUTS/ncbi_synonym_eval_setA.csv  — ALL-CAPS queries (stress test)

Usage:
    python build_synonym_eval.py --output-dir OUTPUTS --max-queries 5000
"""

import os
import re
import io
import gzip
import tarfile
import argparse
import logging
import urllib.request
import pickle
import random

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"

# Name classes to use as queries (synonyms for the canonical scientific name)
QUERY_CLASSES = {
    "synonym",
    "equivalent name",
    "common name",
    "genbank common name",
    "misspelling",          # intentionally included — tests robustness
}

# Patterns for filtering low-quality synonyms
_NUMERIC    = re.compile(r"^\d+$")
_TOO_SHORT  = re.compile(r"^.{1,3}$")
_NO_LETTERS = re.compile(r"^[^a-zA-Z]+$")


def download_taxdump(cache_dir: str) -> str:
    """Download taxdump.tar.gz, return local path. Skips if cached."""
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, "taxdump.tar.gz")
    if os.path.exists(local):
        log.info(f"Using cached taxdump: {local}")
        return local
    log.info(f"Downloading taxdump from NCBI FTP (~50 MB) ...")
    urllib.request.urlretrieve(TAXDUMP_URL, local)
    size_mb = os.path.getsize(local) / 1e6
    log.info(f"Downloaded {size_mb:.1f} MB → {local}")
    return local


def parse_names_dmp(taxdump_path: str) -> pd.DataFrame:
    """
    Parse names.dmp from taxdump archive.
    Returns DataFrame with columns: taxid, name_txt, name_class
    """
    log.info("Parsing names.dmp ...")
    rows = []
    with tarfile.open(taxdump_path, "r:gz") as tar:
        member = tar.getmember("names.dmp")
        f = tar.extractfile(member)
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            # Format: taxid\t|\tname_txt\t|\tunique_name\t|\tname_class\t|
            parts = [p.strip() for p in line.split("\t|\t")]
            if len(parts) < 4:
                continue
            taxid, name_txt, _, name_class = parts[0], parts[1], parts[2], parts[3].rstrip("\t|\n")
            rows.append({"taxid": taxid, "name_txt": name_txt,
                         "name_class": name_class.strip()})
    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df):,} name entries for "
             f"{df['taxid'].nunique():,} unique taxa")
    return df


def is_good_synonym(syn: str, canonical: str) -> bool:
    s = syn.strip()
    if not s:
        return False
    if _NUMERIC.match(s) or _TOO_SHORT.match(s) or _NO_LETTERS.match(s):
        return False
    if s.lower() == canonical.lower():
        return False
    return True


def build_eval_set(output_dir: str, cache_dir: str,
                   max_queries: int = 5000, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: download + parse
    taxdump = download_taxdump(cache_dir)
    names   = parse_names_dmp(taxdump)

    # Step 2: build canonical name lookup (scientific name per taxid)
    sci = names[names["name_class"] == "scientific name"].copy()
    sci = sci.set_index("taxid")["name_txt"].to_dict()
    log.info(f"Scientific names: {len(sci):,} taxa")

    # Step 3: extract query-worthy synonyms
    log.info("Building synonym pairs ...")
    synonyms = names[names["name_class"].isin(QUERY_CLASSES)].copy()

    rows = []
    for _, row in synonyms.iterrows():
        taxid    = row["taxid"]
        syn_text = row["name_txt"].strip()
        syn_type = row["name_class"]
        canonical = sci.get(taxid)
        if canonical is None:
            continue
        if not is_good_synonym(syn_text, canonical):
            continue
        rows.append({
            "query":        syn_text,
            "canonical":    canonical,
            "ncbi_taxid":   taxid,
            "synonym_type": syn_type,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["query", "ncbi_taxid"])
    log.info(f"Total valid synonym pairs: {len(df):,}")
    log.info(f"  Unique taxa with synonyms: {df['ncbi_taxid'].nunique():,}")

    for cls, grp in df.groupby("synonym_type"):
        log.info(f"  {cls:<25}: {len(grp):,}")

    # Step 4: cross-reference with the taxon IDs actually in our FAISS index
    # (only keep queries whose ground-truth taxid is in our index)
    pkl_path = "INPUT/taxon_data_r.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        indexed_ids = set()
        for entry in raw:
            iri = entry[2] if len(entry) > 2 else None
            if iri and "_" in str(iri):
                indexed_ids.add(str(iri).split("_")[-1])
        before = len(df)
        df = df[df["ncbi_taxid"].isin(indexed_ids)]
        log.info(f"Filtered to taxa in FAISS index: {before:,} → {len(df):,} pairs")
    else:
        log.warning("taxon_data_r.pkl not found — skipping FAISS index filter")

    # Step 5: stratified sample
    # Prioritise: synonym > equivalent name > common name > misspelling
    priority = ["synonym", "equivalent name", "common name",
                "genbank common name", "misspelling"]
    quotas = {
        "synonym":           int(max_queries * 0.45),
        "equivalent name":   int(max_queries * 0.25),
        "common name":       int(max_queries * 0.20),
        "genbank common name": int(max_queries * 0.05),
        "misspelling":       int(max_queries * 0.05),
    }
    parts = []
    for cls in priority:
        grp = df[df["synonym_type"] == cls]
        n   = min(len(grp), quotas.get(cls, 0))
        if n > 0:
            parts.append(grp.sample(n, random_state=seed))

    sampled = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    log.info(f"\nSampled {len(sampled):,} queries:")
    for cls, grp in sampled.groupby("synonym_type"):
        log.info(f"  {cls:<25}: {len(grp)}")

    # Step 6: build SetA (ALL-CAPS) and SetB (title-case) variants
    setB = sampled.copy()
    setB["query"] = setB["query"].str.title()

    setA = sampled.copy()
    setA["query"] = setA["query"].str.upper()

    path_b = os.path.join(output_dir, "ncbi_synonym_eval_setB.csv")
    path_a = os.path.join(output_dir, "ncbi_synonym_eval_setA.csv")
    setB.to_csv(path_b, index=False)
    setA.to_csv(path_a, index=False)

    log.info(f"\nSaved SetB (title-case) → {path_b}")
    log.info(f"Saved SetA (ALL-CAPS)   → {path_a}")

    log.info("\n=== Synonym Eval Summary ===")
    log.info(f"  Queries (each set):            {len(sampled):,}")
    log.info(f"  Unique taxa covered:            {sampled['ncbi_taxid'].nunique():,}")
    log.info(f"  Mean query length (words):      "
             f"{sampled['query'].str.split().str.len().mean():.1f}")

    log.info("\n  Sample pairs:")
    for _, r in sampled.sample(10, random_state=seed).iterrows():
        log.info(f"    [{r['synonym_type']:20s}]  '{r['query']}' "
                 f"→ '{r['canonical']}' (taxid={r['ncbi_taxid']})")

    return sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",  default="OUTPUTS")
    parser.add_argument("--cache-dir",   default="INPUT/taxonomy_cache")
    parser.add_argument("--max-queries", type=int, default=5000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    build_eval_set(args.output_dir, args.cache_dir,
                   args.max_queries, args.seed)
