"""
regen_matched_figures.py — regenerate paper figures on the MATCHED n=377 subset.

Rationale: the raw Set A (530) and Set B (515) query sets differ by more than casing
(the title-case pipeline additionally stripped SP./SPECIES/COMPLEX/GROUP qualifiers).
To isolate casing as the sole variable, all Set A / Set B figures are recomputed on the
subset of queries present in BOTH sets (uppercase-identical), n = 377.

Reuses the plotting functions in analyse_results.py by building a summary dataframe in the
same schema and monkeypatching the output directory. Error bars for Hits@1 are bootstrap
95% CI half-widths (not SD), matching the manuscript table/caption convention.

Usage:  python3 scripts/figures/regen_matched_figures.py   (run from the repository root)
"""
import os, sys, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # scripts/figures/
ROOT = os.path.dirname(os.path.dirname(HERE))               # repository root
OUTPUTS = os.path.join(ROOT, "OUTPUTS")
FIGDIR = os.path.join(ROOT, "FIGURES")
os.makedirs(FIGDIR, exist_ok=True)
sys.path.insert(0, HERE)

import analyse_results as AR
AR.FIGURES = FIGDIR                      # redirect save()
AR.OUTPUTS = OUTPUTS

CATEGORY = {
    "menadsa/BioS-MiniLM": "biomedical", "NeuML/pubmedbert-base-embeddings": "biomedical",
    "pritamdeka/S-PubMedBert-MS-MARCO": "biomedical",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "biomedical",
    "intfloat/e5-small-v2": "general", "intfloat/e5-large-v2": "general",
    "intfloat/multilingual-e5-large": "general",
    "sentence-transformers/all-MiniLM-L6-v2": "general",
    "sentence-transformers/all-mpnet-base-v2": "general",
    "juanpablomesa/all-mpnet-base-v2-bioasq-matryoshka": "general",
    "BAAI/bge-base-en-v1.5": "general",
    "emilyalsentzer/Bio_ClinicalBERT": "clinical", "medicalai/ClinicalBERT": "clinical",
    "BM25": "baseline", "RapidFuzz": "baseline", "ETE3": "baseline",
}

METR = ["hits@1", "hits@5", "hits@10", "recall@100", "mrr", "ndcg@10"]


def _load_perquery(path):
    """model -> {UPPER_QUERY: {metric: value}}. Last occurrence of a query wins,
    identical to the authoritative dict semantics used for the manuscript tables."""
    import csv
    d = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m = r["model"]; qu = r["query"].upper()
            rec = {k: float(r[k]) for k in METR if k in r}
            rec["time_ms"] = float(r["time_ms"])
            d.setdefault(m, {})[qu] = rec
    return d


def matched_rows():
    """Per-model summary on the 377 paired queries, for mimic_setA and mimic_setB."""
    A = _load_perquery(os.path.join(OUTPUTS, "exp1_raw_mimic_setA.csv"))
    B = _load_perquery(os.path.join(OUTPUTS, "exp1_raw_mimic_setB.csv"))
    for m, path in (("BM25", "bm25_raw_mimic_setA.csv"),):
        A.update(_load_perquery(os.path.join(OUTPUTS, path)))
    B.update(_load_perquery(os.path.join(OUTPUTS, "bm25_raw_mimic_setB.csv")))

    ref = "intfloat/e5-large-v2"
    paired = sorted(set(A[ref]) & set(B[ref]))
    assert len(paired) == 377, len(paired)

    rng = np.random.default_rng(0)

    def ci_half(vec):
        vec = np.asarray(vec, float)
        idx = rng.integers(0, len(vec), (10000, len(vec)))
        m = vec[idx].mean(1)
        return (np.percentile(m, 97.5) - np.percentile(m, 2.5)) / 2.0

    rows = []
    for ds, src in (("mimic_setA", A), ("mimic_setB", B)):
        for m, tbl in src.items():
            recs = [tbl[q] for q in paired if q in tbl]
            row = {"model": m, "category": CATEGORY.get(m, "baseline"), "dataset": ds,
                   "time_ms_mean": np.mean([r["time_ms"] for r in recs]),
                   "time_ms_std": np.std([r["time_ms"] for r in recs])}
            for k in METR:
                row[f"{k}_mean"] = np.mean([r[k] for r in recs])
                row[f"{k}_std"] = 0.0
            row["hits@1_std"] = ci_half([r["hits@1"] for r in recs])  # error bar = CI half-width
            rows.append(row)
    return rows


def summary_rows():
    """microbiome + synonym_setB summaries (casing-independent) from ontology CSVs."""
    rows = []
    for f in glob.glob(os.path.join(OUTPUTS, "exp1_ontology_results_*.csv")):
        ds = os.path.basename(f).replace("exp1_ontology_results_", "").replace(".csv", "")
        if ds in ("mimic_setA", "mimic_setB"):
            continue
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            rows.append({
                "model": r["model"], "category": CATEGORY.get(r["model"], "baseline"),
                "dataset": ds,
                "hits@1_mean": r.get("hits@1_mean", np.nan), "hits@1_std": r.get("hits@1_std", 0.0),
                "hits@5_mean": r.get("hits@5_mean", np.nan), "hits@10_mean": r.get("hits@10_mean", np.nan),
                "recall@100_mean": r.get("recall@100_mean", np.nan),
                "mrr_mean": r.get("mrr_mean", np.nan), "ndcg@10_mean": r.get("ndcg@10_mean", np.nan),
                "time_ms_mean": r.get("time_ms_mean", np.nan), "time_ms_std": r.get("time_ms_std", 0.0),
            })
    return rows


def main():
    df = pd.DataFrame(matched_rows() + summary_rows())
    print(f"Built matched dataframe: {df['model'].nunique()} models, "
          f"datasets={sorted(df['dataset'].unique())}")
    AR.fig_embedding_comparison(df)
    AR.fig_casing_delta(df)
    AR.fig_hits_at_k(df)
    AR.fig_speed_accuracy(df)
    AR.fig_cd_diagram(df)
    AR.fig_avg_rank(df)
    print("Done — figures written to", FIGDIR)


if __name__ == "__main__":
    main()
