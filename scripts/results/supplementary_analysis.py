"""
Supplementary analysis addressing review items:
  3.1 — Friedman test (CORRECTED: dataset-level blocking per Demšar 2006)
  3.3 — synonym_setB bootstrap 95% CIs
  3.4 — Full Cliff's delta pairwise effect size tables
  8.6 — Bootstrap CIs for BM25 and RapidFuzz (deterministic baselines)
  L5  — Error analysis (failure case sampling and categorisation)
  L7  — MIMIC ground truth coverage and taxonomic composition stats

Key correction (3.1):
  Previous code used queries as blocks → chi² ≈ 3900–8000 (impossible at dataset scale).
  Correct Demšar (2006) application: blocks = datasets, treatments = models.
  With k=3–4 datasets, chi²(N-1) is bounded to ~40–50 at most.

Outputs saved as CSVs in OUTPUTS/ and appended to results.md.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

try:
    from scikit_posthocs import posthoc_nemenyi_friedman
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("WARNING: scikit_posthocs not available — Nemenyi test skipped")

OUTPUTS = "OUTPUTS"
RESULTS_MD = "results.md"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(arr, n_boot=10000, ci=0.95, seed=42):
    """
    Bootstrap 95% CI via percentile method (10,000 resamples).
    Reflects query-level sampling uncertainty regardless of model stochasticity.
    Valid for both neural models and deterministic baselines (BM25, RapidFuzz, ETE3).
    """
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def cliffs_delta(a, b):
    """
    Non-parametric effect size: proportion of (a_i, b_j) pairs where a_i > b_j
    minus proportion where a_i < b_j.

    For binary (0/1) H@1 vectors of equal length this simplifies to O(n):
      d = (n_{a=1}*n_{b=0} - n_{a=0}*n_{b=1}) / n^2
    For non-binary (e.g. MRR), uses vectorised outer comparison O(n^2).

    Interpretation (Romano 2006):
      |d| < 0.147 = negligible
      0.147 <= |d| < 0.33  = small
      0.33  <= |d| < 0.474 = medium
      |d| >= 0.474          = large
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    unique_a = np.unique(a)
    # Fast path for binary arrays (H@1 is always 0/1)
    if np.array_equal(unique_a, [0.0, 1.0]) or np.array_equal(unique_a, [0.0]) or \
       np.array_equal(unique_a, [1.0]):
        n = len(a)
        n_a1 = (a == 1).sum()
        n_b1 = (b == 1).sum()
        concordant = int(n_a1) * int(n - n_b1)
        discordant = int(n - n_a1) * int(n_b1)
        return (concordant - discordant) / (n * n)
    # General path: vectorised outer comparison
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def effect_label(d):
    d = abs(d)
    if d < 0.147: return "negligible"
    if d < 0.330: return "small"
    if d < 0.474: return "medium"
    return "large"


def short_name(model):
    return model.split("/")[-1][:35]


# ---------------------------------------------------------------------------
# 3.1 — Friedman test: CORRECTED dataset-level blocking (Demšar 2006)
# ---------------------------------------------------------------------------

def run_friedman_dataset_level(datasets_dict, metric="hits@1", test_name="neural_models"):
    """
    Apply the Friedman test as specified by Demšar (2006):
      - blocks    = datasets  (k = number of datasets)
      - treatments = models   (N = number of models)

    datasets_dict: {dataset_name: raw_df}
    Each raw_df must contain columns ['model', metric].

    Rationale for correction:
      The previous implementation used pivot_table(index='query') → chi² ≈ 3900.
      That placed each query as a block (k=530 or 4750), which is statistically
      incoherent: the Friedman test compares classifiers, not query instances.
      Demšar (2006) Sec. 3: "each dataset provides one measurement per classifier."

    Power note:
      With k=3 datasets, the test has very limited statistical power.
      The chi²(N-1) critical value at p=0.05 is printed for reference.
      Non-significance at k=3 does NOT imply equivalence.
    """
    print(f"\n  Friedman test (dataset-level blocking) — {test_name}, metric={metric}")
    print(f"  Blocks (datasets): {list(datasets_dict.keys())}")

    # Keep only models present in ALL datasets
    model_sets = [set(df["model"].unique()) for df in datasets_dict.values()]
    shared_models = sorted(model_sets[0].intersection(*model_sets[1:]))
    print(f"  Models (treatments): n={len(shared_models)}")

    if len(shared_models) < 2:
        print("  ERROR: fewer than 2 shared models across all datasets — skipping")
        return {}

    dataset_names = list(datasets_dict.keys())
    k = len(dataset_names)
    N = len(shared_models)

    # Build score matrix: rows=datasets (blocks), cols=models (treatments)
    score_matrix = np.zeros((k, N))
    for i, dname in enumerate(dataset_names):
        df = datasets_dict[dname]
        for j, model in enumerate(shared_models):
            vals = df[df["model"] == model][metric].values
            score_matrix[i, j] = vals.mean() if len(vals) > 0 else np.nan

    # Drop dataset rows with any NaN (model absent from that dataset)
    valid_rows = ~np.isnan(score_matrix).any(axis=1)
    score_matrix = score_matrix[valid_rows]
    k_actual = score_matrix.shape[0]
    if k_actual < 2:
        print("  ERROR: fewer than 2 complete dataset rows — skipping")
        return {}

    print(f"  Score matrix: {k_actual} datasets × {N} models")
    critical_val = stats.chi2.ppf(0.95, N - 1)
    print(f"  chi²({N - 1}) critical value at p=0.05: {critical_val:.2f}")

    # scipy.stats.friedmanchisquare: pass one array per treatment (column)
    args = [score_matrix[:, j] for j in range(N)]
    stat, p = stats.friedmanchisquare(*args)
    print(f"  Friedman chi²({N - 1}) = {stat:.4f},  p = {p:.4e}")
    print(f"  (k={k_actual} dataset blocks, N={N} model treatments)")

    if k_actual < 5:
        print(f"  WARNING: k={k_actual} blocks → limited power. "
              f"Non-significance does not imply equivalence.")

    result = {
        "test_name":             test_name,
        "metric":                metric,
        "n_datasets_blocks":     k_actual,
        "n_models_treatments":   N,
        "datasets":              dataset_names,
        "models":                shared_models,
        "score_matrix":          score_matrix,
        "friedman_stat":         stat,
        "friedman_p":            p,
        "df":                    N - 1,
    }

    # Nemenyi post-hoc (only if Friedman is significant)
    if p < 0.05 and HAS_POSTHOCS:
        nem = posthoc_nemenyi_friedman(score_matrix)
        nem.index   = [short_name(m) for m in shared_models]
        nem.columns = [short_name(m) for m in shared_models]
        nem_path = os.path.join(OUTPUTS, f"nemenyi_{test_name}_{metric}.csv")
        nem.to_csv(nem_path)
        print(f"  Nemenyi matrix saved → {nem_path}")
        result["nemenyi_df"] = nem
    elif p >= 0.05:
        print(f"  Nemenyi post-hoc skipped (p={p:.4f} >= 0.05)")
    else:
        print(f"  Nemenyi post-hoc skipped (scikit_posthocs not installed)")

    # Average ranks per Demšar (2006) — rank 1 = best
    rank_matrix = np.zeros_like(score_matrix)
    for i in range(k_actual):
        order = np.argsort(-score_matrix[i])   # descending: best first
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, N + 1)
        rank_matrix[i] = ranks
    avg_ranks = rank_matrix.mean(axis=0)

    rank_df = pd.DataFrame({
        "model":      [short_name(m) for m in shared_models],
        "model_full": shared_models,
        "avg_rank":   avg_ranks.round(3),
        "avg_score":  score_matrix.mean(axis=0).round(4),
    }).sort_values("avg_rank")
    rank_path = os.path.join(OUTPUTS, f"avg_ranks_{test_name}_{metric}.csv")
    rank_df.to_csv(rank_path, index=False)
    print(f"  Average ranks saved → {rank_path}")
    result["rank_df"] = rank_df

    return result


# ---------------------------------------------------------------------------
# Bootstrap CIs — all models including deterministic baselines (8.6 + 3.2 + 3.3)
# ---------------------------------------------------------------------------

def compute_bootstrap_cis(raw_df, dataset_name, metric="hits@1"):
    """
    Compute 95% bootstrap CIs for all models in raw_df (10,000 resamples).

    CIs reflect query-level sampling uncertainty: if a different sample of queries
    were drawn from the population, the model's mean H@1 would differ. This
    uncertainty is equally valid for deterministic baselines (BM25, RapidFuzz, ETE3).
    The footnote claim "not applicable for deterministic baselines" is incorrect.
    """
    print(f"\n  Bootstrap CIs — {dataset_name}, metric={metric}")
    ci_rows = []
    for model, grp in raw_df.groupby("model"):
        vals = grp[metric].values
        lo, hi = bootstrap_ci(vals)
        ci_rows.append({
            "dataset":    dataset_name,
            "model":      short_name(model),
            "model_full": model,
            "metric":     metric,
            "n_queries":  len(vals),
            "mean":       round(float(vals.mean()), 4),
            "ci_lo":      round(float(lo), 4),
            "ci_hi":      round(float(hi), 4),
            "ci_str":     f"{vals.mean():.3f} [{lo:.3f}–{hi:.3f}]",
        })
    ci_df = pd.DataFrame(ci_rows).sort_values("mean", ascending=False)
    ci_path = os.path.join(OUTPUTS, f"bootstrap_ci_{dataset_name}.csv")
    ci_df.to_csv(ci_path, index=False)
    print(f"  {len(ci_df)} models — saved → {ci_path}")
    return ci_df


# ---------------------------------------------------------------------------
# 3.4 — Cliff's delta: full pairwise table
# ---------------------------------------------------------------------------

def compute_cliffs_delta_table(raw_df, dataset_name, metric="hits@1"):
    """
    Compute Cliff's delta for ALL pairwise model comparisons.
    Uses raw per-query H@1 vectors (NOT derived from Nemenyi p-values).
    """
    print(f"\n  Cliff's delta (all pairs) — {dataset_name}, metric={metric}")
    pivot = raw_df.pivot_table(index="query", columns="model", values=metric, aggfunc="mean")
    pivot = pivot.dropna()
    models = list(pivot.columns)
    print(f"  {len(models)} models, {len(pivot)} queries → {len(list(combinations(models, 2)))} pairs")

    rows = []
    for m1, m2 in combinations(models, 2):
        d = cliffs_delta(pivot[m1].values, pivot[m2].values)
        rows.append({
            "dataset":      dataset_name,
            "model_a":      short_name(m1),
            "model_b":      short_name(m2),
            "cliffs_delta": round(float(d), 4),
            "abs_delta":    round(abs(float(d)), 4),
            "effect_size":  effect_label(d),
            "direction":    (f"{short_name(m1)} > {short_name(m2)}" if d > 0 else
                             f"{short_name(m2)} > {short_name(m1)}" if d < 0 else "tied"),
        })

    delta_df = pd.DataFrame(rows).sort_values("abs_delta", ascending=False)
    delta_path = os.path.join(OUTPUTS, f"cliffs_delta_{dataset_name}_{metric}.csv")
    delta_df.to_csv(delta_path, index=False)

    summary = delta_df["effect_size"].value_counts()
    print(f"  Effect size distribution: {summary.to_dict()}")
    print(f"  {len(delta_df)} pairs saved → {delta_path}")
    return delta_df


# ---------------------------------------------------------------------------
# L5 — Error analysis
# ---------------------------------------------------------------------------

def error_analysis(raw_a, raw_b, n_sample=25):
    print("\n  Error analysis — sampling failure cases")
    results = {}

    model_a = "menadsa/BioS-MiniLM"
    failures_a = raw_a[(raw_a["model"] == model_a) & (raw_a["hits@1"] == 0)].copy()
    print(f"  BioS-MiniLM SetA failures: {len(failures_a)} / {len(raw_a[raw_a['model']==model_a])}")

    success_b = set(raw_b[(raw_b["model"] == model_a) & (raw_b["hits@1"] == 1)]["query"].str.title())
    failures_a["query_title"] = failures_a["query"].str.title()
    failures_a["casing_failure"] = failures_a["query_title"].isin(success_b)

    def categorise(row):
        q = row["query"]
        if len(q.split()) <= 2 or any(len(w) <= 2 for w in q.split()):
            return "abbreviation/short"
        return "true_retrieval_failure"

    failures_a["category"] = failures_a.apply(
        lambda r: "casing_failure" if r["casing_failure"] else categorise(r), axis=1
    )

    sample_a = failures_a.sample(min(n_sample, len(failures_a)), random_state=42)[
        ["query", "gt_id", "category", "casing_failure"]
    ].sort_values("category")

    casing_pct = failures_a["casing_failure"].mean() * 100
    abbrev_pct = (failures_a["category"] == "abbreviation/short").mean() * 100
    true_pct   = (failures_a["category"] == "true_retrieval_failure").mean() * 100
    print(f"    Casing failures: {casing_pct:.1f}%  |  Abbrev: {abbrev_pct:.1f}%  |  True failures: {true_pct:.1f}%")

    sample_path = os.path.join(OUTPUTS, "error_analysis_biosminilm_setA.csv")
    sample_a.to_csv(sample_path, index=False)
    print(f"  Saved sample → {sample_path}")

    model_cb = "medicalai/ClinicalBERT"
    failures_cb = raw_b[(raw_b["model"] == model_cb) & (raw_b["hits@1"] == 0)].copy()
    print(f"\n  ClinicalBERT SetB failures: {len(failures_cb)} / {len(raw_b[raw_b['model']==model_cb])}")

    results.update({
        "biosminilm_seta_failures": failures_a,
        "casing_pct":               casing_pct,
        "abbrev_pct_biosminilm":    abbrev_pct,
    })
    return results


# ---------------------------------------------------------------------------
# L7 — MIMIC coverage statistics
# ---------------------------------------------------------------------------

def coverage_stats():
    print("\n  MIMIC coverage statistics")
    orgs_path = os.path.join(OUTPUTS, "mimic_organisms.csv")
    raw_path  = os.path.join(OUTPUTS, "mimic_organisms_raw.csv")

    if not os.path.exists(orgs_path):
        print("  mimic_organisms.csv not found — skipping")
        return {}

    df = pd.read_csv(orgs_path)
    resolved   = df[df["resolution_status"] == "resolved"]
    unresolved = df[df["resolution_status"] != "resolved"]
    print(f"  Total unique: {len(df)} | Resolved: {len(resolved)} ({len(resolved)/len(df)*100:.1f}%)")

    total_records    = df["frequency"].sum()
    resolved_records = resolved["frequency"].sum()

    df["name_len"] = df["org_name_normalized"].str.split().str.len()
    print(f"  Name length — mean: {df['name_len'].mean():.1f}, median: {df['name_len'].median():.0f}")
    print(f"  Record coverage: {resolved_records/total_records*100:.1f}%")

    stats_out = {
        "total_unique":         len(df),
        "resolved":             len(resolved),
        "unresolved":           len(unresolved),
        "resolution_rate_pct":  round(len(resolved) / len(df) * 100, 1),
        "record_coverage_pct":  round(resolved_records / total_records * 100, 1),
        "name_len_mean":        round(df["name_len"].mean(), 1),
    }
    cov_path = os.path.join(OUTPUTS, "mimic_coverage_stats.csv")
    pd.DataFrame([stats_out]).to_csv(cov_path, index=False)
    print(f"  Coverage stats saved → {cov_path}")

    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        print(f"  Raw microbiologyevents rows processed: {len(raw):,}")

    return stats_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASELINES = {"BM25", "RapidFuzz", "ETE3"}


def main():
    print("=" * 70)
    print("Supplementary Analysis (3.1, 3.3, 3.4, 8.6, L5, L7)")
    print("=" * 70)

    # ── Load per-query raw results ─────────────────────────────────────────
    raw_a   = pd.read_csv(os.path.join(OUTPUTS, "exp1_raw_mimic_setA.csv"))
    raw_b   = pd.read_csv(os.path.join(OUTPUTS, "exp1_raw_mimic_setB.csv"))
    raw_m   = pd.read_csv(os.path.join(OUTPUTS, "exp1_raw_microbiome.csv"))
    raw_syn = pd.read_csv(os.path.join(OUTPUTS, "exp1_raw_synonym_setB.csv"))

    # BM25 raw files (separate from neural raw CSVs)
    bm25_a = pd.read_csv(os.path.join(OUTPUTS, "bm25_raw_mimic_setA.csv"))
    bm25_b = pd.read_csv(os.path.join(OUTPUTS, "bm25_raw_mimic_setB.csv"))

    # Full frames including baselines for CI and Cliff's delta
    raw_a_full = pd.concat([raw_a, bm25_a], ignore_index=True)
    raw_b_full = pd.concat([raw_b, bm25_b], ignore_index=True)

    # ── 3.1 — Friedman test: dataset-level blocking ────────────────────────
    print("\n" + "=" * 70)
    print("3.1 — Friedman Test (dataset-level blocking per Demšar 2006)")
    print("=" * 70)
    print("  Previous chi² values (~3900): produced by using queries as blocks.")
    print("  Correct application: blocks = datasets, treatments = models.")

    # Neural models only (consistent with original BM25 exclusion; see review 3.6)
    neural_a   = raw_a[~raw_a["model"].isin(BASELINES)]
    neural_b   = raw_b[~raw_b["model"].isin(BASELINES)]
    neural_m   = raw_m[~raw_m["model"].isin(BASELINES)]
    neural_syn = raw_syn[~raw_syn["model"].isin(BASELINES)]

    # Primary: 3 main datasets (highest-quality blocking)
    friedman_3 = run_friedman_dataset_level(
        {"mimic_setA": neural_a, "mimic_setB": neural_b, "microbiome": neural_m},
        metric="hits@1", test_name="neural_3datasets",
    )

    # Secondary: 4 datasets including synonym_setB (different query distribution)
    friedman_4 = run_friedman_dataset_level(
        {"mimic_setA": neural_a, "mimic_setB": neural_b,
         "microbiome": neural_m, "synonym_setB": neural_syn},
        metric="hits@1", test_name="neural_4datasets",
    )

    # ── Bootstrap CIs (all models including BM25) ──────────────────────────
    print("\n" + "=" * 70)
    print("3.2 / 8.6 / 3.3 — Bootstrap 95% CIs (10,000 resamples, all models)")
    print("=" * 70)

    ci_a   = compute_bootstrap_cis(raw_a_full, "mimic_setA")
    ci_b   = compute_bootstrap_cis(raw_b_full, "mimic_setB")
    ci_m   = compute_bootstrap_cis(raw_m,      "microbiome")
    ci_syn = compute_bootstrap_cis(raw_syn,     "synonym_setB")

    # ── 3.4 — Cliff's delta (all pairs) ───────────────────────────────────
    print("\n" + "=" * 70)
    print("3.4 — Cliff's Delta (all pairwise comparisons)")
    print("=" * 70)

    delta_a   = compute_cliffs_delta_table(raw_a_full, "mimic_setA")
    delta_b   = compute_cliffs_delta_table(raw_b_full, "mimic_setB")
    delta_syn = compute_cliffs_delta_table(raw_syn,    "synonym_setB")

    # ── L5 — Error analysis ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L5 — Error Analysis")
    print("=" * 70)
    err = error_analysis(raw_a, raw_b)

    # ── L7 — Coverage stats ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L7 — MIMIC Coverage Statistics")
    print("=" * 70)
    cov = coverage_stats()

    # ── Append summary to results.md ───────────────────────────────────────
    print("\n  Appending supplementary section to results.md ...")

    f3 = friedman_3
    f4 = friedman_4

    def sig_label(p):
        if p < 0.001: return "Yes (p < 0.001, ***)"
        if p < 0.01:  return "Yes (p < 0.01, **)"
        if p < 0.05:  return "Yes (p < 0.05, *)"
        return "No (ns)"

    f3_stat = f3.get("friedman_stat", float("nan"))
    f3_p    = f3.get("friedman_p",    float("nan"))
    f4_stat = f4.get("friedman_stat", float("nan"))
    f4_p    = f4.get("friedman_p",    float("nan"))
    k3 = f3.get("n_datasets_blocks", 3)
    k4 = f4.get("n_datasets_blocks", 4)
    N3 = f3.get("n_models_treatments", 13)
    N4 = f4.get("n_models_treatments", 13)

    supp = f"""
---

## Statistical Significance (corrected)

### Friedman Test — Dataset-level blocking (Demšar 2006)

**Correction note:** Previously reported chi² values (3905, 3365, 3936, 8377)
were produced by treating each query as a block (k=530 or k=4750 blocks).
This is a fundamental misapplication: Demšar (2006, Section 3) specifies that
one block = one dataset. The corrected values below use datasets as blocks.

**Power caveat:** With k={k3}–{k4} datasets as blocks, the Friedman test has
very limited power. The chi²({N3-1}) critical value at p=0.05 is
{stats.chi2.ppf(0.95, N3-1):.1f}. Non-significance does not imply model equivalence;
it reflects insufficient replication at the dataset level, not lack of effect.

| Comparison | k blocks | N models | chi²(N-1) | p | Significant? |
|---|---|---|---|---|---|
| 3 main datasets (mimic A+B, microbiome) | {k3} | {N3} | {f3_stat:.4f} | {f3_p:.4e} | {sig_label(f3_p)} |
| 4 datasets incl. synonym_setB | {k4} | {N4} | {f4_stat:.4f} | {f4_p:.4e} | {sig_label(f4_p)} |

Average ranks per dataset saved to `OUTPUTS/avg_ranks_*.csv`.
Nemenyi post-hoc matrices (where applicable) saved to `OUTPUTS/nemenyi_*.csv`.

### Bootstrap 95% CI — Hits@1, Set A (all-caps, incl. BM25)

"""
    for _, row in ci_a.iterrows():
        supp += f"- **{row['model']}**: {row['ci_str']}\n"

    supp += "\n### Bootstrap 95% CI — Hits@1, Set B (normalised, incl. BM25)\n\n"
    for _, row in ci_b.iterrows():
        supp += f"- **{row['model']}**: {row['ci_str']}\n"

    supp += "\n### Bootstrap 95% CI — Hits@1, synonym_setB\n\n"
    for _, row in ci_syn.iterrows():
        supp += f"- **{row['model']}**: {row['ci_str']}\n"

    supp += "\n### Cliff's Delta — Full Pairwise Effect Sizes\n\n"
    supp += "Full tables: `OUTPUTS/cliffs_delta_*.csv`\n\n"
    supp += "Interpretation (Romano 2006): |d| < 0.147 negligible · 0.147–0.33 small · 0.33–0.474 medium · ≥0.474 large\n\n"
    for ds_name, delta_df in [("mimic_setA", delta_a), ("mimic_setB", delta_b), ("synonym_setB", delta_syn)]:
        if not delta_df.empty:
            counts = delta_df["effect_size"].value_counts().to_dict()
            supp += f"**{ds_name}:** {counts.get('large',0)} large · {counts.get('medium',0)} medium · "
            supp += f"{counts.get('small',0)} small · {counts.get('negligible',0)} negligible "
            supp += f"(of {len(delta_df)} pairs)\n"

    casing_pct = err.get("casing_pct", 0)
    abbrev_pct = err.get("abbrev_pct_biosminilm", 0)
    true_pct   = 100 - casing_pct - abbrev_pct

    supp += f"""
---

## Error Analysis

### BioS-MiniLM — Set A Failure Cases

| Category | Proportion |
|---|---|
| Casing failures (normalisation recovers H@1=1) | {casing_pct:.1f}% |
| Abbreviation / short name | {abbrev_pct:.1f}% |
| True retrieval failure | {true_pct:.1f}% |

Sampled cases: `OUTPUTS/error_analysis_biosminilm_setA.csv`

---

## MIMIC-IV Ground Truth Coverage

"""
    if cov:
        supp += (f"| Statistic | Value |\n|---|---|\n"
                 f"| Unique organism names | {cov['total_unique']} |\n"
                 f"| Resolved to NCBI TaxID | {cov['resolved']} ({cov['resolution_rate_pct']}%) |\n"
                 f"| Unresolved / excluded | {cov['unresolved']} |\n"
                 f"| Record-level coverage | {cov['record_coverage_pct']}% |\n"
                 f"| Mean organism name length | {cov['name_len_mean']} tokens |\n")
    else:
        supp += "_Coverage stats unavailable (mimic_organisms.csv not found)._\n"

    with open(RESULTS_MD, "a") as f:
        f.write(supp)

    print(f"\nDone. Outputs:")
    print(f"  OUTPUTS/avg_ranks_*          — Demšar (2006) average ranks")
    print(f"  OUTPUTS/nemenyi_*            — post-hoc Nemenyi (if Friedman significant)")
    print(f"  OUTPUTS/bootstrap_ci_*.csv   — 95% CIs all models incl. BM25/RapidFuzz")
    print(f"  OUTPUTS/cliffs_delta_*.csv   — full pairwise Cliff's delta tables")
    print(f"  OUTPUTS/error_analysis_*.csv — sampled failure cases")
    print(f"  {RESULTS_MD}                 — supplementary section appended")


if __name__ == "__main__":
    main()
