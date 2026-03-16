#!/bin/bash
#SBATCH --job-name=bm25_sapbert_check
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bm25_sapbert_%j.out
#SBATCH --error=logs/bm25_sapbert_%j.err

source "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}/.venv/bin/activate"
cd "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}"

# -----------------------------------------------------------------------
# Part 1: BM25 baseline (L4)
# Run exp1 in BM25-only mode by temporarily patching the model registry.
# We call a small inline script that reuses the existing evaluator.
# -----------------------------------------------------------------------
python - <<'PYEOF'
import os, sys, pickle, time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, ".")

# Load taxon data
with open("INPUT/taxon_names.pkl", "rb") as f:
    taxon_names = pickle.load(f)
with open("INPUT/taxon_data_r.pkl", "rb") as f:
    raw = pickle.load(f)

taxon_ids = []
for entry in raw:
    iri = entry[2] if len(entry) > 2 else None
    if iri and "_" in str(iri):
        tid = str(iri).split("_")[-1]
    else:
        tid = str(iri) if iri else ""
    taxon_ids.append(tid)

print(f"Loaded {len(taxon_names)} taxon names")

from rank_bm25 import BM25Okapi

print("Building BM25 index (this may take a few minutes)...")
t0 = time.perf_counter()
tokenized = [name.lower().split() for name in taxon_names]
bm25 = BM25Okapi(tokenized)
print(f"BM25 index built in {time.perf_counter()-t0:.1f}s")

def hits_at_k(retrieved, gt, k):
    return float(gt in retrieved[:k])

def mrr(retrieved, gt):
    try:
        return 1.0 / (retrieved.index(gt) + 1)
    except ValueError:
        return 0.0

def ndcg_at_k(retrieved, gt, k=10):
    import numpy as np
    for i, r in enumerate(retrieved[:k]):
        if r == gt:
            return 1.0 / np.log2(i + 2)
    return 0.0

def query_bm25(query, k=100):
    t0 = time.perf_counter()
    toks = query.lower().split()
    scores = bm25.get_scores(toks)
    top_k = np.argsort(scores)[::-1][:k]
    elapsed = (time.perf_counter() - t0) * 1000
    return [taxon_ids[i] for i in top_k], elapsed

for dataset, query_file, label in [
    ("mimic_setA", "OUTPUTS/mimic_eval_set_a.csv", "mimic_setA"),
    ("mimic_setB", "OUTPUTS/mimic_eval_set_b.csv", "mimic_setB"),
]:
    qdf = pd.read_csv(query_file)
    queries = qdf["query"].tolist()
    gts = qdf["ncbi_taxid"].astype(str).tolist()

    records = []
    for q, gt in tqdm(zip(queries, gts), total=len(queries), desc=f"BM25 {dataset}"):
        retrieved, elapsed = query_bm25(q)
        records.append({
            "query": q, "gt_id": gt,
            "hits@1":     hits_at_k(retrieved, gt, 1),
            "hits@5":     hits_at_k(retrieved, gt, 5),
            "hits@10":    hits_at_k(retrieved, gt, 10),
            "recall@100": hits_at_k(retrieved, gt, 100),
            "mrr":        mrr(retrieved, gt),
            "ndcg@10":    ndcg_at_k(retrieved, gt, 10),
            "model":      "BM25",
            "category":   "baseline",
            "dataset":    dataset,
            "time_ms":    elapsed,
        })

    df = pd.DataFrame(records)
    raw_path = f"OUTPUTS/bm25_raw_{dataset}.csv"
    df.to_csv(raw_path, index=False)

    agg = {k: df[k].mean() for k in ["hits@1","hits@5","hits@10","recall@100","mrr","ndcg@10","time_ms"]}
    print(f"\n=== BM25 {dataset} ===")
    print(f"  H@1={agg['hits@1']:.3f}  H@5={agg['hits@5']:.3f}  MRR={agg['mrr']:.3f}  NDCG@10={agg['ndcg@10']:.3f}  ms/q={agg['time_ms']:.1f}")

print("\nBM25 done.")
PYEOF

# -----------------------------------------------------------------------
# Part 2: SapBERT spot-check (L6)
# Encode a handful of canonical taxon names with SapBERT and search
# its own index — confirms whether 0.000 is a bug or true failure.
# -----------------------------------------------------------------------
python - <<'PYEOF'
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

INDEX_PATH = "INPUT/indices/ncbi_faiss_sapbertfrompubmedbertfulltext.index"
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

print(f"\n=== SapBERT spot-check ===")
print(f"Loading index: {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)
print(f"Index: {index.ntotal:,} vectors, dim={index.d}")

with open("INPUT/taxon_names.pkl", "rb") as f:
    taxon_names = pickle.load(f)
with open("INPUT/taxon_data_r.pkl", "rb") as f:
    raw = pickle.load(f)

taxon_ids = []
for entry in raw:
    iri = entry[2] if len(entry) > 2 else None
    tid = str(iri).split("_")[-1] if (iri and "_" in str(iri)) else str(iri)
    taxon_ids.append(tid)

name_to_idx = {n.lower(): i for i, n in enumerate(taxon_names)}

model = SentenceTransformer(MODEL_NAME)

test_cases = [
    ("Homo sapiens",        "9606"),
    ("Escherichia coli",    "562"),
    ("ESCHERICHIA COLI",    "562"),   # all-caps version
    ("Mus musculus",        "10090"),
    ("Staphylococcus aureus", "1280"),
    ("STAPHYLOCOCCUS AUREUS", "1280"),
]

for query, expected_id in test_cases:
    vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(vec.astype("float32"), 5)
    top5_ids  = [taxon_ids[i] for i in I[0] if 0 <= i < len(taxon_ids)]
    top5_names = [taxon_names[i] for i in I[0] if 0 <= i < len(taxon_names)]
    hit = expected_id in top5_ids
    print(f"  Query: '{query:<30s}'  expected_id={expected_id}  hit@5={hit}")
    print(f"    Top-5: {list(zip(top5_names[:3], top5_ids[:3]))}")

print("\nSapBERT spot-check done.")
PYEOF

echo "Job complete."
