"""
Experiment 2: MIMIC-IV Organism Standardisation Evaluation
Preprocesses microbiologyevents.csv.gz, resolves NCBI TaxIDs, then re-uses
OntologyMatchingEvaluator from exp1 to evaluate all models.
"""

import os
import re
import time
import logging
import argparse
import warnings
from typing import Tuple
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# Biopython Entrez for NCBI TaxID lookup
try:
    from Bio import Entrez
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Organism strings that are NOT organism names
EXCLUDE_STRINGS = {
    "NO GROWTH", "CANCELLED", "CONTAMINANT", "MIXED BACTERIAL FLORA",
    "YEAST", "GRAM POSITIVE COCCI", "GRAM NEGATIVE RODS",
    "GRAM POSITIVE RODS", "GRAM NEGATIVE COCCI", "MIXED FLORA",
    "COMMENSAL FLORA", "PRESUMPTIVE IDENTIFICATION",
    "NOT APPLICABLE", "PENDING", "REJECTED", "CANCELLED",
    "UNABLE TO PROCESS", "INSUFFICIENT SAMPLE",
}

# Suffixes to strip before NCBI lookup
TRAILING_SUFFIXES = re.compile(
    r"\s+(SPECIES|SP\.|SPP\.|SP|SUBSPECIES|SUBSP\.|SUBSP|VAR\.|VAR|TYPE|GROUP|COMPLEX)\s*$",
    re.IGNORECASE
)


def normalize_organism_name(name: str) -> str:
    """Convert all-caps clinical names to title case and strip trailing noise."""
    name = name.strip()
    # Title-case if all-caps
    if name == name.upper():
        name = name.title()
    # Strip trailing suffixes
    name = TRAILING_SUFFIXES.sub("", name).strip()
    # Remove abundance markers like (10^3)
    name = re.sub(r"\s*\(\d+\^?\d*\)\s*", " ", name).strip()
    return name


def resolve_taxid_via_entrez(organism_name: str, email: str, max_retries: int = 3) -> str:
    """Query NCBI Entrez taxonomy database for a TaxID."""
    if not HAS_BIOPYTHON:
        return ""
    Entrez.email = email
    for attempt in range(max_retries):
        try:
            handle = Entrez.esearch(db="taxonomy", term=f'"{organism_name}"[Scientific Name]', retmax=5)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            if len(ids) == 1:
                return ids[0]
            elif len(ids) > 1:
                return f"ambiguous:{','.join(ids)}"
            # Fallback: search without quotes
            handle = Entrez.esearch(db="taxonomy", term=organism_name, retmax=5)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            if len(ids) == 1:
                return ids[0]
            elif len(ids) > 1:
                return f"ambiguous:{','.join(ids)}"
            return ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"Entrez lookup failed for '{organism_name}': {e}")
    return ""


class MIMICPreprocessor:

    def __init__(self, mimic_dir: str, output_dir: str,
                 entrez_email: str = "r.v.bumbuc@amsterdamumc.nl"):
        self.mimic_dir = mimic_dir
        self.output_dir = output_dir
        self.entrez_email = entrez_email
        os.makedirs(output_dir, exist_ok=True)

    def load_microbiologyevents(self) -> pd.DataFrame:
        gz_path = os.path.join(self.mimic_dir, "microbiologyevents.csv.gz")
        csv_path = os.path.join(self.mimic_dir, "microbiologyevents.csv")
        if os.path.exists(gz_path):
            logger.info(f"Loading {gz_path} ...")
            df = pd.read_csv(gz_path, compression="gzip", low_memory=False)
        elif os.path.exists(csv_path):
            logger.info(f"Loading {csv_path} ...")
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            raise FileNotFoundError(f"No microbiologyevents file found in {self.mimic_dir}")
        logger.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")
        return df

    def extract_unique_organisms(self, df: pd.DataFrame) -> pd.DataFrame:
        # Find organism name column
        org_col = None
        for candidate in ["org_name", "ORG_NAME", "organism", "ORGANISM"]:
            if candidate in df.columns:
                org_col = candidate
                break
        if org_col is None:
            raise ValueError(f"No organism name column found. Available: {list(df.columns)}")

        logger.info(f"Using column: {org_col}")
        counts = df[org_col].value_counts()
        total_isolates = len(df)

        # Filter exclusions
        mask = (
            df[org_col].notna() &
            ~df[org_col].str.strip().str.upper().isin(EXCLUDE_STRINGS) &
            ~df[org_col].str.strip().str.match(r"^\d+$").fillna(False)  # purely numeric
        )
        filtered = df.loc[mask, org_col]
        unique_names = filtered.unique().tolist()
        logger.info(f"Total rows: {total_isolates:,}")
        logger.info(f"After filtering: {len(filtered):,} rows, {len(unique_names)} unique organisms")

        # Build organism DataFrame with frequency
        freq = df[org_col].value_counts()
        records = []
        for name in unique_names:
            records.append({
                "org_name_original": name,
                "org_name_normalized": normalize_organism_name(name),
                "frequency": freq.get(name, 0),
            })
        org_df = pd.DataFrame(records).sort_values("frequency", ascending=False).reset_index(drop=True)
        return org_df

    def resolve_taxids(self, org_df: pd.DataFrame, max_organisms: int = None) -> pd.DataFrame:
        if not HAS_BIOPYTHON:
            logger.warning("Biopython not available; skipping NCBI Entrez lookup")
            org_df["ncbi_taxid"] = ""
            org_df["resolution_status"] = "skipped"
            return org_df

        if max_organisms:
            org_df = org_df.head(max_organisms).copy()

        taxids = []
        statuses = []

        logger.info(f"Resolving TaxIDs for {len(org_df)} organisms via NCBI Entrez...")
        for _, row in tqdm(org_df.iterrows(), total=len(org_df), desc="NCBI Entrez"):
            taxid = resolve_taxid_via_entrez(row["org_name_normalized"], self.entrez_email)
            if taxid.startswith("ambiguous:"):
                status = "ambiguous"
                taxid = taxid.split(":")[1].split(",")[0]  # take first
            elif taxid:
                status = "resolved"
            else:
                status = "unresolved"
                # Try original name
                taxid = resolve_taxid_via_entrez(row["org_name_original"], self.entrez_email)
                if taxid.startswith("ambiguous:"):
                    status = "ambiguous"
                    taxid = taxid.split(":")[1].split(",")[0]
                elif taxid:
                    status = "resolved_original"
            taxids.append(taxid)
            statuses.append(status)
            time.sleep(0.34)  # NCBI rate limit: max 3 requests/sec

        org_df = org_df.copy()
        org_df["ncbi_taxid"] = taxids
        org_df["resolution_status"] = statuses
        return org_df

    def create_evaluation_sets(self, org_df: pd.DataFrame) -> Tuple:
        """Create Set A (original all-caps) and Set B (normalized title-case)."""
        resolved = org_df[org_df["resolution_status"].isin(["resolved", "resolved_original"])].copy()
        logger.info(f"Resolved organisms: {len(resolved)}/{len(org_df)}")

        set_a = resolved[["org_name_original", "ncbi_taxid"]].rename(
            columns={"org_name_original": "query"}
        )
        set_b = resolved[["org_name_normalized", "ncbi_taxid"]].rename(
            columns={"org_name_normalized": "query"}
        )
        return set_a, set_b

    def report_stats(self, org_df: pd.DataFrame):
        total = len(org_df)
        resolved = (org_df["resolution_status"].isin(["resolved", "resolved_original"])).sum()
        ambiguous = (org_df["resolution_status"] == "ambiguous").sum()
        unresolved = (org_df["resolution_status"] == "unresolved").sum()
        skipped = (org_df["resolution_status"] == "skipped").sum()
        logger.info("=" * 50)
        logger.info(f"MIMIC Organisms Summary:")
        logger.info(f"  Total unique organisms:  {total}")
        logger.info(f"  Resolved:                {resolved} ({100*resolved/total:.1f}%)")
        logger.info(f"  Ambiguous:               {ambiguous} ({100*ambiguous/total:.1f}%)")
        logger.info(f"  Unresolved:              {unresolved} ({100*unresolved/total:.1f}%)")
        logger.info(f"  Skipped (no Biopython):  {skipped}")
        logger.info("=" * 50)

    def run(self, max_organisms: int = None):
        df = self.load_microbiologyevents()
        org_df = self.extract_unique_organisms(df)

        # Save raw organism list first
        raw_path = os.path.join(self.output_dir, "mimic_organisms_raw.csv")
        org_df.to_csv(raw_path, index=False)
        logger.info(f"Raw organism list saved: {raw_path}")

        # Resolve TaxIDs
        org_df = self.resolve_taxids(org_df, max_organisms=max_organisms)

        # Save full organism table
        org_path = os.path.join(self.output_dir, "mimic_organisms.csv")
        org_df.to_csv(org_path, index=False)
        logger.info(f"Organism table with TaxIDs saved: {org_path}")

        self.report_stats(org_df)

        # Create evaluation sets
        set_a, set_b = self.create_evaluation_sets(org_df)
        set_a.to_csv(os.path.join(self.output_dir, "mimic_eval_set_a.csv"), index=False)
        set_b.to_csv(os.path.join(self.output_dir, "mimic_eval_set_b.csv"), index=False)
        logger.info(f"Evaluation Set A (original): {len(set_a)} queries")
        logger.info(f"Evaluation Set B (normalized): {len(set_b)} queries")

        return org_df, set_a, set_b


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: MIMIC-IV Preprocessing")
    parser.add_argument("--mimic-dir",     default="MIMIC",   help="Directory containing microbiologyevents.csv.gz")
    parser.add_argument("--output-dir",    default="OUTPUTS", help="Output directory")
    parser.add_argument("--email",         default="r.v.bumbuc@amsterdamumc.nl",
                        help="Email for NCBI Entrez")
    parser.add_argument("--max-organisms", type=int, default=None,
                        help="Limit number of organisms to resolve (for testing)")
    parser.add_argument("--run-eval",      action="store_true",
                        help="After preprocessing, run embedding evaluation")
    parser.add_argument("--base-dir",      default="INPUT",         help="Base dir with FAISS indices")
    parser.add_argument("--indices-dir",   default="INPUT/indices", help="Indices dir")
    args = parser.parse_args()

    preprocessor = MIMICPreprocessor(
        mimic_dir=args.mimic_dir,
        output_dir=args.output_dir,
        entrez_email=args.email,
    )
    org_df, set_a, set_b = preprocessor.run(max_organisms=args.max_organisms)

    if args.run_eval:
        from exp1_embedding_benchmark import OntologyMatchingEvaluator
        evaluator = OntologyMatchingEvaluator(
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            indices_dir=args.indices_dir,
        )
        evaluator.load_all_indices()

        for setname, eval_df in [("mimic_setA", set_a), ("mimic_setB", set_b)]:
            if eval_df.empty:
                continue
            queries = eval_df["query"].tolist()
            gt_ids  = eval_df["ncbi_taxid"].astype(str).tolist()
            logger.info(f"Running evaluation on {setname} ({len(queries)} queries)")
            summary_df = evaluator.run_all(
                dataset=setname,
                queries_file=os.path.join(args.output_dir, f"mimic_eval_{setname.split('_')[1].lower()}.csv"),
            )


if __name__ == "__main__":
    main()
