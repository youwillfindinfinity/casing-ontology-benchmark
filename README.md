# Who Actually Breaks on ALL-CAPS? Casing Sensitivity in Embedding-Based Biomedical Ontology Alignment

Benchmark code and derived artefacts for:

> **Who Actually Breaks on ALL-CAPS? Casing Sensitivity in Embedding-Based
> Biomedical Ontology Alignment.**
> Roland V. Bumbuc. *Health Information Science and Systems*, {{YEAR}}.
> DOI: {{ARTICLE_DOI}}

[![DOI](https://zenodo.org/badge/1294101912.svg)](https://doi.org/10.5281/zenodo.21268379)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)

This repository benchmarks 13 open-weight sentence-transformer models plus three
baselines (BM25, RapidFuzz, ETE3) for zero-shot retrieval of organism names against the
full **NCBI Taxonomy** (2.69 M canonical names), with **input casing isolated as a
controlled variable** using a matched-query design.

**Headline finding.** On the matched query subset where casing is the *only* difference,
modern uncased contrastively-fine-tuned encoders are already casing-robust
(ΔHits@1 = 0.000 for 12 of 16 systems, including the case-insensitive ETE3 control).
Casing is the dominant failure mode **only** for case-sensitive systems — the fuzzy
matcher RapidFuzz (ΔHits@1 = +0.732) and the cased, lowercased-pretrained clinical
BERTs (ClinicalBERT +0.406, Bio_ClinicalBERT +0.366). SapBERT fails structurally on
NCBI Taxonomy regardless of casing (species-epithet collapse).

---

## Repository layout

```
.
├── README.md                     ← this file
├── LICENSE                       ← MIT (code)
├── CITATION.cff                  ← how to cite
├── .zenodo.json                  ← Zenodo deposition metadata
├── DATA_AVAILABILITY.md          ← what is open vs. PhysioNet-credentialed (DUA)
├── Makefile                      ← `make reproduce` and other entry points
├── requirements.txt              ← pinned Python dependencies
├── scripts/
│   ├── results/                  ← experiment drivers (produce OUTPUTS/)
│   │   ├── exp2_mimic_prep.py            NCBI TaxID resolution of MIMIC organisms
│   │   ├── build_synonym_eval.py         synonym_setB construction from names.dmp
│   │   ├── embedding_evaluator.py        FAISS IndexFlatIP + Hits@k evaluation core
│   │   ├── exp1_embedding_benchmark.py   main 16-system benchmark
│   │   └── supplementary_analysis.py     Friedman / Nemenyi / bootstrap / Cliff's δ
│   ├── figures/
│   │   ├── analyse_results.py            figure functions
│   │   └── regen_matched_figures.py      regenerate figs on the matched n=377 subset
│   ├── plot_style.py                     Okabe-Ito publication style
│   ├── synonym_type_breakdown.py         per-synonym-type Hits@1 (Table S4)
│   └── slurm/                            Snellius HPC batch scripts
├── OUTPUTS/                      ← result tables; MIMIC-derived files are .gitignored (DUA, see DATA_AVAILABILITY.md)
├── FIGURES/                      ← final PDF/PNG figures (matched n=377)
└── index/
    ├── build_index.py            ← build a FAISS IndexFlatIP over NCBI Taxonomy (one model)
    ├── make_checksums.py         ← emit CHECKSUMS.sha256 for the archived indices
    └── CHECKSUMS.sha256          ← (generated) SHA-256 of each hosted index
```

> **Note on the matched subset.** The raw `mimic_setA` (530 queries) and `mimic_setB`
> (515 queries) files differ by more than casing: the title-case pipeline additionally
> stripped qualifier tokens (`SP.`, `SPECIES`, `COMPLEX`, `GROUP`). All Set A / Set B
> comparisons in the paper are therefore computed on the **matched n = 377 subset**
> present in both sets, where Set B is exactly `str.title()` of Set A. Reproduce with
> `scripts/figures/regen_matched_figures.py`.

---

## Reproducing the results

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.10–3.11
```

### 2. Build the FAISS index (GPU recommended; ~2–8 h/model on an A100)

```bash
python scripts/results/exp1_embedding_benchmark.py --build-index --model all-MiniLM-L6-v2
```

The 2.69 M-name NCBI Taxonomy `names.dmp` is public domain
(<https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/>). A prebuilt index per model
(~8.2 GB for 768-d) is archived on Zenodo (DOI 10.5281/zenodo.21268379); verify with
`index/CHECKSUMS.sha256`.

### 3. Run the benchmark

```bash
# open datasets (microbiome self-retrieval + NCBI synonym set) — no credentials needed
python scripts/results/exp1_embedding_benchmark.py --dataset microbiome
python scripts/results/exp1_embedding_benchmark.py --dataset synonym_setB

# MIMIC-IV datasets — requires credentialed PhysioNet access (see DATA_AVAILABILITY.md)
python scripts/results/exp2_mimic_prep.py           # resolve MIMIC organisms → TaxID
python scripts/results/exp1_embedding_benchmark.py --dataset mimic_setA
python scripts/results/exp1_embedding_benchmark.py --dataset mimic_setB
```

### 4. Statistics and figures

```bash
python scripts/results/supplementary_analysis.py    # Friedman, Cliff's δ, bootstrap CIs
python scripts/figures/regen_matched_figures.py      # all six figures on matched n=377
```

Single-command reproduction of the analysis from released per-query vectors:

```bash
make reproduce      # runs supplementary_analysis + regen_matched_figures
```

---

## Recommended models (zero-shot deployment)

| Scenario | Recommended | Hits@1 (ALL-CAPS, matched) | Notes |
|---|---|---|---|
| General single model | **e5-small-v2** | 0.859 | Pareto-optimal (299 ms/query); casing-invariant |
| Highest accuracy | e5-large-v2 | 0.867 | +0.008 over e5-small at ~2.5× latency |
| Strong lexical baseline | BM25 | 0.857 | statistically tied; always include as a baseline |
| **Do not use** | SapBERT, ClinicalBERT, Bio_ClinicalBERT | 0.000 | structural / casing failure on raw ALL-CAPS |

---

## Citing

See [`CITATION.cff`](CITATION.cff). If you use the benchmark artefacts, please cite both
the article ({{ARTICLE_DOI}}) and the archived dataset (10.5281/zenodo.21268379).

## License

Code: MIT (see [`LICENSE`](LICENSE)). NCBI Taxonomy data: public domain.
MIMIC-IV-derived artefacts: PhysioNet Data Use Agreement — see
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).
