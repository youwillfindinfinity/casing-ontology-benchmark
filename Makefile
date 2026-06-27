# Makefile — reproduction entry points for the casing-sensitivity benchmark.
#
# Tiers of reproduction (cheapest first):
#   make reproduce   → re-derive all statistics + figures from released per-query vectors
#                      (no GPU, no PhysioNet credentials; uses OUTPUTS/)
#   make figures     → regenerate the six figures on the matched n=377 subset
#   make stats       → Friedman / Cliff's delta / bootstrap CIs
#   make index       → build a FAISS IndexFlatIP for one model over NCBI Taxonomy (GPU)
#   make benchmark   → full zero-shot benchmark of one model (GPU; MIMIC needs credentials)
#   make checksums   → (re)compute index/CHECKSUMS.sha256
#
# Config
PYTHON  ?= python3
MODEL   ?= all-MiniLM-L6-v2
DATASET ?= microbiome            # microbiome | synonym_setB | mimic_setA | mimic_setB
NAMESDMP ?= data/names.dmp       # NCBI Taxonomy dump (public domain; see 'make names')

.PHONY: reproduce figures stats index benchmark checksums names clean help

help:
	@grep -E '^#|^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | sed 's/^# \{0,1\}//'

## --- Tier 1: no GPU, no credentials -----------------------------------------
reproduce: stats figures
	@echo ">> Reproduction complete (stats + figures from released vectors)."

figures:
	$(PYTHON) scripts/figures/regen_matched_figures.py

stats:
	$(PYTHON) scripts/results/supplementary_analysis.py

## --- Tier 2: GPU (index construction + inference) ---------------------------
names:
	@mkdir -p data
	@echo ">> Download NCBI Taxonomy names.dmp (public domain):"
	@echo "   curl -L https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz | tar -xz -C data names.dmp"

index: $(NAMESDMP)
	$(PYTHON) index/build_index.py --model $(MODEL) --names-dmp $(NAMESDMP) --out index/$(MODEL).faiss

benchmark:
	@echo ">> mimic_* datasets require credentialed PhysioNet access (see DATA_AVAILABILITY.md)."
	$(PYTHON) scripts/results/exp1_embedding_benchmark.py --model $(MODEL) --dataset $(DATASET)

## --- Utilities --------------------------------------------------------------
checksums:
	$(PYTHON) index/make_checksums.py index/ > index/CHECKSUMS.sha256
	@echo ">> Wrote index/CHECKSUMS.sha256"

clean:
	rm -f FIGURES/*.png
	@echo ">> Removed regenerated PNGs (PDFs retained)."
