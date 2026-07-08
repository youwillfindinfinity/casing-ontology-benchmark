# Data Availability & FAIR Statement

This benchmark combines **public-domain** reference data, **openly-licensed** derived
artefacts, and **PhysioNet-credentialed** (MIMIC-IV-derived) data governed by the
PhysioNet Data Use Agreement (DUA). The split below is what makes the FAIR claims in the
paper (Table 6) true rather than aspirational.

> **How the split is enforced.** All result tables live under `OUTPUTS/`, but the
> MIMIC-IV-derived files listed under *PhysioNet-credentialed only* are excluded from
> the public Git repository and the Zenodo archive via `.gitignore`. Cloning the public
> repository therefore yields only the openly-redistributable subset; credentialed
> PhysioNet users regenerate the restricted files with the provided scripts.

## Openly redistributable (this repository / Zenodo, MIT or public domain)

| Artefact | License | Location |
|---|---|---|
| All evaluation, preprocessing, statistics, and plotting code | MIT | `scripts/` |
| NCBI Taxonomy `names.dmp` reference (2.69 M canonical names) | Public domain | NCBI FTP (not redistributed; download script provided) |
| FAISS index build script + SHA-256 checksums | MIT | `index/` |
| Prebuilt FAISS indices (per model) | CC-BY-4.0 | Zenodo DOI 10.5281/zenodo.21268379 |
| **synonym_setB** (4,750 NCBI synonym→canonical pairs) | Public domain (derived from `names.dmp`) | `OUTPUTS/` |
| **microbiome** self-retrieval set (500 taxa, Fujita 2023) | CC-BY-4.0 | `OUTPUTS/` |
| Aggregate per-model result tables (Hits@k means, bootstrap CIs, Cliff's δ, Nemenyi) | CC-BY-4.0 | `OUTPUTS/` |
| Per-synonym-type Hits@1 (`synonym_type_h1.csv`, Table S4) | CC-BY-4.0 | `OUTPUTS/` |
| All six figures (PDF/PNG) | CC-BY-4.0 | `FIGURES/` |

## PhysioNet-credentialed only (MIMIC-IV DUA — NOT in the public repository)

These files contain organism name strings extracted from the MIMIC-IV
`microbiologyevents` table and are therefore MIMIC-IV-derived. Under the PhysioNet DUA
they may be shared **only** with other credentialed PhysioNet users, not posted openly.

| Artefact | Why restricted |
|---|---|
| `mimic_eval_set_a.csv`, `mimic_eval_set_b.csv` | ALL-CAPS / title-case organism query strings from MIMIC-IV |
| `mimic_organisms*.csv`, `mimic_coverage_stats.csv` | MIMIC-IV organism inventory + resolution |
| `exp1_raw_mimic_setA/B.csv`, `bm25_raw_mimic_setA/B.csv` | per-query results keyed by MIMIC organism strings |
| `exp1_ontology_results_mimic_setA/B.csv`, `exp1_stats_mimic_*.csv` | MIMIC per-model aggregates keyed to the restricted query sets |
| `cliffs_delta_mimic_*`, `nemenyi_mimic_*`, `bootstrap_ci_mimic_*`, `paired_mimic_summary.json` | derived from the restricted per-query MIMIC vectors |

**Re-derivation for credentialed users.** The `str.title()` derivation of Set B and the
matched-subset construction are fully scripted; a credentialed PhysioNet user can
regenerate every MIMIC artefact from raw MIMIC-IV v3.1 by running
`scripts/results/exp2_mimic_prep.py` followed by `exp1_embedding_benchmark.py`.

## FAIR mapping

- **Findable** — public artefacts carry a persistent Zenodo DOI (10.5281/zenodo.21268379);
  the article links to it.
- **Accessible** — open artefacts are downloadable without registration; MIMIC-derived
  artefacts are accessible to any credentialed PhysioNet user via the same scripts.
- **Interoperable** — CSV with documented schemas; FAISS binary (faiss ≥ 1.7);
  TaxID columns use NCBI Taxonomy identifiers.
- **Reusable** — MIT/CC-BY licensing, pinned dependencies, single-command reproduction.

## Ethics

MIMIC-IV is a publicly available, de-identified critical-care database used under the
PhysioNet DUA; no additional ethics approval was required. NCBI Taxonomy data are public
domain.
