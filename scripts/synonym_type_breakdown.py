#!/usr/bin/env python3
"""Per-synonym-type Hits@1 breakdown for synonym_setB (review item R-8 / D3).

Joins the per-query raw results (exp1_raw_synonym_setB.csv) with the synonym-type
annotation (ncbi_synonym_eval_setB.csv) on the query string and reports mean Hits@1
per (model, synonym_type). Writes OUTPUTS/synonym_type_h1.csv.

The low overall synonym ceiling (best H@1 = 0.322) is partly an artefact of synonym
types that never occur in clinical text (common name, genbank common name); reporting
per-type H@1 shows models do much better on clinically plausible types.
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "OUTPUTS"

raw = pd.read_csv(OUT / "exp1_raw_synonym_setB.csv")
ev = pd.read_csv(OUT / "ncbi_synonym_eval_setB.csv")

merged = raw.merge(ev[["query", "synonym_type"]], on="query", how="left")
assert merged["synonym_type"].isna().sum() == 0, "unmatched queries after merge"

tbl = (
    merged.groupby(["model", "synonym_type"])["hits@1"]
    .mean()
    .unstack("synonym_type")
    .round(4)
)
# order columns from clinically-plausible to clinically-irrelevant
col_order = ["synonym", "equivalent name", "common name", "genbank common name"]
tbl = tbl[[c for c in col_order if c in tbl.columns]]
tbl["overall"] = merged.groupby("model")["hits@1"].mean().round(4)
tbl = tbl.sort_values("overall", ascending=False)

tbl.to_csv(OUT / "synonym_type_h1.csv")
print(tbl.to_string())
print(f"\nType counts: {ev['synonym_type'].value_counts().to_dict()}")
