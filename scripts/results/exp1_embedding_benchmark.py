"""
Experiment 1: Comprehensive Embedding Model Benchmarking
Evaluates all 15 FAISS indices + 4 baselines on parafac4microbiome and MIMIC-IV datasets.
Metrics: Hits@1, Hits@5, Hits@10, Recall@100, MRR, NDCG@10, avg query time.
"""

import os
import time
import pickle
import argparse
import logging
import warnings
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Statistical tests
from scipy import stats
try:
    from scikit_posthocs import posthoc_nemenyi_friedman
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

# Fuzzy matching
try:
    from rapidfuzz import process as rfprocess, fuzz as rffuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# ETE3 for taxonomy lookup
try:
    from ete3 import NCBITaxa
    HAS_ETE3 = True
except ImportError:
    HAS_ETE3 = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry: index filename -> HuggingFace model ID
# ---------------------------------------------------------------------------
INDEX_TO_MODEL = {
    # General — kept
    "ncbi_faiss_allminilml6v2.index":                   "sentence-transformers/all-MiniLM-L6-v2",
    "ncbi_faiss_allmpnetbasev2.index":                  "sentence-transformers/all-mpnet-base-v2",
    "ncbi_faiss_allmpnetbasev2bioasqmatryoshka.index":  "juanpablomesa/all-mpnet-base-v2-bioasq-matryoshka",
    "ncbi_faiss_bgebaseenv15.index":                    "BAAI/bge-base-en-v1.5",
    "ncbi_faiss_e5smallv2.index":                       "intfloat/e5-small-v2",
    "ncbi_faiss_e5largev2.index":                       "intfloat/e5-large-v2",
    "ncbi_faiss_multilinguale5large.index":             "intfloat/multilingual-e5-large",
    # Biomedical — kept + new
    "ncbi_faiss_biosminilm.index":                      "menadsa/BioS-MiniLM",
    "ncbi_faiss_pubmedbertbaseembeddings.index":        "NeuML/pubmedbert-base-embeddings",
    "ncbi_faiss_spubmedbertmsmarco.index":              "pritamdeka/S-PubMedBert-MS-MARCO",
    "ncbi_faiss_sapbertfrompubmedbertfulltext.index":   "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    # Clinical — kept as casing-sensitivity evidence
    "ncbi_faiss_clinicalbert.index":                    "medicalai/ClinicalBERT",
    "ncbi_faiss_bio_clinicalbert.index":                "emilyalsentzer/Bio_ClinicalBERT",
    # Removed: BioM-ELECTRA (0.000), SciBERT (~0.000), BiomedNLP-fulltext (redundant),
    #          S-BioBert-snli (0.008 setA — extreme casing failure, clinical models fill this role)
}

MODEL_CATEGORIES = {
    # General-purpose sentence encoders
    "sentence-transformers/all-MiniLM-L6-v2":                              "general",
    "sentence-transformers/all-mpnet-base-v2":                             "general",
    "juanpablomesa/all-mpnet-base-v2-bioasq-matryoshka":                   "general",
    "BAAI/bge-base-en-v1.5":                                               "general",
    "intfloat/e5-small-v2":                                                "general",
    "intfloat/e5-large-v2":                                                "general",
    "intfloat/multilingual-e5-large":                                      "general",
    # Biomedical / domain-specific
    "menadsa/BioS-MiniLM":                                                 "biomedical",
    "NeuML/pubmedbert-base-embeddings":                                     "biomedical",
    "pritamdeka/S-PubMedBert-MS-MARCO":                                    "biomedical",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext":                       "biomedical",
    # Clinical — retained as casing-sensitivity evidence
    "medicalai/ClinicalBERT":                                              "clinical",
    "emilyalsentzer/Bio_ClinicalBERT":                                      "clinical",
}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def hits_at_k(retrieved_ids: List, ground_truth_id, k: int) -> float:
    return float(ground_truth_id in retrieved_ids[:k])


def reciprocal_rank(retrieved_ids: List, ground_truth_id) -> float:
    try:
        rank = retrieved_ids.index(ground_truth_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(retrieved_ids: List, ground_truth_id, k: int = 10) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == ground_truth_id:
            dcg += 1.0 / np.log2(i + 2)
    # Ideal DCG: correct answer at position 1
    idcg = 1.0 / np.log2(2)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: List, ground_truth_ids: List, k: int = 100) -> float:
    if not ground_truth_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(ground_truth_ids))
    return hits / len(ground_truth_ids)


def compute_metrics(retrieved_ids: List, ground_truth_id, ground_truth_ids: List = None) -> Dict:
    if ground_truth_ids is None:
        ground_truth_ids = [ground_truth_id]
    return {
        "hits@1":      hits_at_k(retrieved_ids, ground_truth_id, 1),
        "hits@5":      hits_at_k(retrieved_ids, ground_truth_id, 5),
        "hits@10":     hits_at_k(retrieved_ids, ground_truth_id, 10),
        "recall@100":  recall_at_k(retrieved_ids, ground_truth_ids, 100),
        "mrr":         reciprocal_rank(retrieved_ids, ground_truth_id),
        "ndcg@10":     ndcg_at_k(retrieved_ids, ground_truth_id, 10),
    }


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

class RapidFuzzBaseline:
    def __init__(self, taxon_names: List[str], taxon_ids: List[str]):
        self.taxon_names = taxon_names
        self.taxon_ids = taxon_ids

    def query(self, query: str, k: int = 100) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        results = rfprocess.extract(
            query, self.taxon_names, scorer=rffuzz.WRatio, limit=k
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        retrieved_ids = []
        for match, score, idx in results:
            retrieved_ids.append(self.taxon_ids[idx])
        return retrieved_ids, elapsed_ms


class BM25Baseline:
    def __init__(self, taxon_names: List[str], taxon_ids: List[str]):
        self.taxon_names = taxon_names
        self.taxon_ids = taxon_ids
        tokenized = [name.lower().split() for name in taxon_names]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, query: str, k: int = 100) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[::-1][:k]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return [self.taxon_ids[i] for i in top_k_idx], elapsed_ms


class ETE3Baseline:
    def __init__(self, taxon_names: List[str], taxon_ids: List[str]):
        self.name_to_id = {name.lower(): tid for name, tid in zip(taxon_names, taxon_ids)}
        if HAS_ETE3:
            try:
                self.ncbi = NCBITaxa()
            except Exception:
                self.ncbi = None
        else:
            self.ncbi = None

    def query(self, query: str, k: int = 100) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        retrieved = []
        q_lower = query.lower()
        # Exact match first
        if q_lower in self.name_to_id:
            retrieved.append(self.name_to_id[q_lower])
        # ETE3 fuzzy lookup
        if self.ncbi and len(retrieved) < k:
            try:
                name2taxid = self.ncbi.get_name_translator([query])
                for name, ids in name2taxid.items():
                    for tid in ids:
                        if str(tid) not in retrieved:
                            retrieved.append(str(tid))
            except Exception:
                pass
        # Substring fallback
        if len(retrieved) < k:
            for name, tid in self.name_to_id.items():
                if q_lower in name or name in q_lower:
                    if tid not in retrieved:
                        retrieved.append(tid)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return retrieved[:k], elapsed_ms


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class OntologyMatchingEvaluator:

    def __init__(self, base_dir: str, output_dir: str, indices_dir: str):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.indices_dir = indices_dir
        os.makedirs(output_dir, exist_ok=True)

        self.taxon_names: List[str] = []
        self.taxon_ids: List[str] = []     # NCBI TaxIDs
        self.indices: Dict[str, faiss.Index] = {}
        self.models: Dict[str, SentenceTransformer] = {}
        self._load_taxon_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_taxon_data(self):
        names_file = os.path.join(self.base_dir, "taxon_names.pkl")
        data_file  = os.path.join(self.base_dir, "taxon_data_r.pkl")

        if os.path.exists(names_file):
            with open(names_file, "rb") as f:
                self.taxon_names = pickle.load(f)
            logger.info(f"Loaded {len(self.taxon_names)} taxon names")
        else:
            logger.warning(f"taxon_names.pkl not found at {names_file}")

        if os.path.exists(data_file):
            with open(data_file, "rb") as f:
                raw = pickle.load(f)
            # taxon_data_r: list of (label, rank, iri) tuples
            # IRI contains NCBI TaxID: e.g. NCBITaxon_9606 -> "9606"
            if raw and isinstance(raw[0], (tuple, list)):
                self.taxon_data = raw
                self.taxon_ids = []
                for entry in raw:
                    iri = entry[2] if len(entry) > 2 else None
                    if iri and "_" in str(iri):
                        tid = str(iri).split("_")[-1]
                    else:
                        tid = str(iri) if iri else ""
                    self.taxon_ids.append(tid)
            else:
                self.taxon_data = raw
                self.taxon_ids = [str(i) for i in range(len(self.taxon_names))]
            logger.info(f"Loaded {len(self.taxon_ids)} taxon IDs")
        else:
            logger.warning(f"taxon_data_r.pkl not found at {data_file}")
            self.taxon_ids = [str(i) for i in range(len(self.taxon_names))]

    def load_all_indices(self):
        """Register which indices exist — load lazily one at a time during evaluation."""
        for fname in INDEX_TO_MODEL:
            fpath = os.path.join(self.indices_dir, fname)
            if os.path.exists(fpath):
                self.indices[fname] = fpath   # store path only, load on demand
                logger.info(f"Found index: {fname}")
            else:
                logger.warning(f"Index not found: {fpath}")

    def _load_index(self, fname: str) -> faiss.Index:
        """Load a single FAISS index from disk."""
        path = self.indices[fname]
        index = faiss.read_index(path)
        logger.info(f"Loaded {fname} ({index.ntotal:,} vectors, dim={index.d})")
        return index

    def load_model(self, model_name: str) -> Optional[SentenceTransformer]:
        if model_name not in self.models:
            try:
                self.models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return None
        return self.models[model_name]

    def load_baselines(self) -> Dict:
        baselines = {}
        if HAS_ETE3 or True:  # ETE3 fallback always available
            baselines["ETE3"] = ETE3Baseline(self.taxon_names, self.taxon_ids)
            logger.info("Loaded ETE3 baseline")
        if HAS_RAPIDFUZZ:
            baselines["RapidFuzz"] = RapidFuzzBaseline(self.taxon_names, self.taxon_ids)
            logger.info("Loaded RapidFuzz baseline")
        if HAS_BM25:
            baselines["BM25"] = BM25Baseline(self.taxon_names, self.taxon_ids)
            logger.info("Loaded BM25 baseline")
        return baselines

    # ------------------------------------------------------------------
    # Evaluation logic
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model_name: str,
        index: faiss.Index,
        queries: List[str],
        ground_truth_ids: List[str],
        k: int = 100,
        prefix_query: bool = False,
    ) -> List[Dict]:
        model = self.load_model(model_name)
        if model is None:
            return []

        q_texts = [f"query: {q}" if prefix_query else q for q in queries]

        # Batch-encode all queries at once for GPU efficiency, then measure per-query FAISS latency.
        # Reported time_ms = amortised encoding cost + FAISS search time (realistic online latency).
        t_enc_start = time.perf_counter()
        all_embs = model.encode(q_texts, batch_size=256, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        t_enc_total = time.perf_counter() - t_enc_start
        per_query_encode_ms = (t_enc_total / len(queries)) * 1000
        faiss.normalize_L2(all_embs)

        results = []
        for query, gt_id, q_emb in tqdm(zip(queries, ground_truth_ids, all_embs),
                                         total=len(queries), desc=model_name[:40], leave=False):
            t0 = time.perf_counter()
            _, faiss_indices = index.search(q_emb.reshape(1, -1), min(k, index.ntotal))
            faiss_ms = (time.perf_counter() - t0) * 1000
            elapsed_ms = per_query_encode_ms + faiss_ms

            retrieved_ids = [self.taxon_ids[i] for i in faiss_indices[0] if 0 <= i < len(self.taxon_ids)]
            m = compute_metrics(retrieved_ids, gt_id)
            m["query"] = query
            m["gt_id"] = gt_id
            m["model"] = model_name
            m["time_ms"] = elapsed_ms
            results.append(m)
        return results

    def evaluate_baseline(
        self,
        baseline_name: str,
        baseline,
        queries: List[str],
        ground_truth_ids: List[str],
        k: int = 100,
    ) -> List[Dict]:
        results = []
        for query, gt_id in tqdm(zip(queries, ground_truth_ids),
                                  total=len(queries), desc=baseline_name, leave=False):
            retrieved_ids, elapsed_ms = baseline.query(query, k)
            m = compute_metrics(retrieved_ids, gt_id)
            m["query"] = query
            m["gt_id"] = gt_id
            m["model"] = baseline_name
            m["time_ms"] = elapsed_ms
            results.append(m)
        return results

    def aggregate_metrics(self, records: List[Dict]) -> Dict:
        if not records:
            return {}
        keys = ["hits@1", "hits@5", "hits@10", "recall@100", "mrr", "ndcg@10", "time_ms"]
        agg = {}
        for k in keys:
            vals = [r[k] for r in records if k in r]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"]  = float(np.std(vals))
        return agg

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def run_statistical_tests(self, results_df: pd.DataFrame, metric: str = "hits@1") -> pd.DataFrame:
        models = results_df["model"].unique()
        # Build matrix: rows = queries, cols = models
        pivot = results_df.pivot_table(index="query", columns="model", values=metric, aggfunc="mean")
        pivot = pivot.dropna()
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            logger.warning("Not enough data for statistical tests")
            return pd.DataFrame()

        groups = [pivot[m].values for m in pivot.columns]
        stat, p_value = stats.friedmanchisquare(*groups)
        logger.info(f"Friedman test on {metric}: chi2={stat:.3f}, p={p_value:.4f}")

        stats_records = [{"test": "friedman", "metric": metric, "statistic": stat, "p_value": p_value}]

        if HAS_POSTHOCS and p_value < 0.05:
            nemenyi = posthoc_nemenyi_friedman(pivot.values)
            nemenyi.index = pivot.columns
            nemenyi.columns = pivot.columns
            nemenyi_path = os.path.join(self.output_dir, f"exp1_nemenyi_{metric}.csv")
            nemenyi.to_csv(nemenyi_path)
            logger.info(f"Nemenyi results saved to {nemenyi_path}")

        return pd.DataFrame(stats_records)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run_all(
        self,
        dataset: str = "microbiome",
        queries_file: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        query_transform: str = "none",
    ):
        logger.info(f"Starting Experiment 1 — dataset: {dataset}, query_transform: {query_transform}")
        self.load_all_indices()

        # Load queries and ground truth
        if queries_file and os.path.exists(queries_file):
            qdf = pd.read_csv(queries_file)
            queries = qdf["query"].tolist() if "query" in qdf.columns else qdf.iloc[:, 0].tolist()
            ground_truth_ids = qdf["ncbi_taxid"].astype(str).tolist() if "ncbi_taxid" in qdf.columns else qdf.iloc[:, 1].tolist()
        else:
            # Use taxon names as self-retrieval queries (sanity check)
            logger.info("No query file provided — using taxon names as self-retrieval queries (sample of 500)")
            sample_idx = np.random.choice(len(self.taxon_names), size=min(500, len(self.taxon_names)), replace=False)
            queries = [self.taxon_names[i] for i in sample_idx]
            ground_truth_ids = [self.taxon_ids[i] for i in sample_idx]

        # Apply query transform (Set C: str.lower(), or re-derive Set B: str.title())
        if query_transform == "lower":
            queries = [q.lower() for q in queries]
            logger.info(f"Applied str.lower() to {len(queries)} queries (Set C)")
        elif query_transform == "title":
            queries = [q.title() for q in queries]
            logger.info(f"Applied str.title() to {len(queries)} queries (Set B re-derive)")

        all_records = []
        summary_rows = []

        # Evaluate embedding models — load one index at a time to stay within RAM
        for fname, model_name in INDEX_TO_MODEL.items():
            if fname not in self.indices:
                continue
            index = self._load_index(fname)   # load from disk
            prefix = model_name.startswith("intfloat/e5") or model_name.startswith("BAAI/bge")
            records = self.evaluate_model(model_name, index, queries, ground_truth_ids, prefix_query=prefix)
            del index   # free RAM before loading next index
            if records:
                all_records.extend(records)
                agg = self.aggregate_metrics(records)
                agg["model"] = model_name
                agg["category"] = MODEL_CATEGORIES.get(model_name, "unknown")
                agg["dataset"] = dataset
                agg["n_queries"] = len(records)
                summary_rows.append(agg)
                logger.info(f"{model_name[:50]:50s}  H@1={agg.get('hits@1_mean', 0):.3f}  MRR={agg.get('mrr_mean', 0):.3f}")

        # Evaluate baselines
        baselines = self.load_baselines()
        for bname, baseline in baselines.items():
            records = self.evaluate_baseline(bname, baseline, queries, ground_truth_ids)
            if records:
                all_records.extend(records)
                agg = self.aggregate_metrics(records)
                agg["model"] = bname
                agg["category"] = "baseline"
                agg["dataset"] = dataset
                agg["n_queries"] = len(records)
                summary_rows.append(agg)
                logger.info(f"{bname:50s}  H@1={agg.get('hits@1_mean', 0):.3f}  MRR={agg.get('mrr_mean', 0):.3f}")

        # Save raw results
        raw_df = pd.DataFrame(all_records)
        raw_path = os.path.join(self.output_dir, f"exp1_raw_{dataset}.csv")
        raw_df.to_csv(raw_path, index=False)
        logger.info(f"Raw results saved: {raw_path}")

        # Save summary
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(self.output_dir, f"exp1_ontology_results_{dataset}.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved: {summary_path}")
        print(summary_df[["model", "hits@1_mean", "hits@5_mean", "mrr_mean", "ndcg@10_mean"]].to_string())

        # Statistical tests
        if len(summary_rows) >= 2 and not raw_df.empty:
            stats_df = self.run_statistical_tests(raw_df, metric="hits@1")
            stats_path = os.path.join(self.output_dir, f"exp1_stats_{dataset}.csv")
            stats_df.to_csv(stats_path, index=False)
            logger.info(f"Stats saved: {stats_path}")

        return summary_df


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Embedding Model Benchmarking")
    parser.add_argument("--base-dir",       default="INPUT",         help="Directory with taxon pkl files")
    parser.add_argument("--output-dir",     default="OUTPUTS",       help="Output directory")
    parser.add_argument("--indices-dir",    default="INPUT/indices", help="Directory with FAISS indices")
    parser.add_argument("--dataset",          default="microbiome",  help="Dataset name tag")
    parser.add_argument("--queries",          default=None,          help="CSV file with query,ncbi_taxid columns")
    parser.add_argument("--ground-truth",     default=None,          help="Alias for --queries")
    parser.add_argument("--query-transform",  default="none",
                        choices=["none", "lower", "title"],
                        help="Transform applied to query strings before evaluation: "
                             "none = Set A (raw ALL-CAPS), lower = Set C (str.lower()), "
                             "title = Set B re-derive (str.title())")
    args = parser.parse_args()

    evaluator = OntologyMatchingEvaluator(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        indices_dir=args.indices_dir,
    )
    queries_file = args.queries or args.ground_truth
    evaluator.run_all(
        dataset=args.dataset,
        queries_file=queries_file,
        query_transform=args.query_transform,
    )


if __name__ == "__main__":
    main()
