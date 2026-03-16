#!/bin/bash
#SBATCH --job-name=syn_eval_cont
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/syn_eval_cont_%j.out
#SBATCH --error=logs/syn_eval_cont_%j.err

source "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}/.venv/bin/activate"
cd "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}"

# Step 1: synonym CSVs already exist from previous run — rebuild only if missing
if [ ! -f OUTPUTS/ncbi_synonym_eval_setB.csv ]; then
    echo "=== Step 1: Download NCBI taxdump and extract synonym pairs ==="
    python build_synonym_eval.py \
        --output-dir  OUTPUTS \
        --cache-dir   INPUT/taxonomy_cache \
        --max-queries 5000 \
        --seed        42
    if [ ! -f OUTPUTS/ncbi_synonym_eval_setB.csv ]; then
        echo "ERROR: synonym eval CSV not created — aborting"
        exit 1
    fi
else
    echo "=== Step 1: Synonym CSVs already exist — skipping ==="
    wc -l OUTPUTS/ncbi_synonym_eval_setB.csv OUTPUTS/ncbi_synonym_eval_setA.csv
fi

echo ""
echo "=== Step 2: Embedding benchmark — synonym_setB (title-case) ==="
python exp1_embedding_benchmark.py \
    --base-dir    INPUT \
    --output-dir  OUTPUTS \
    --indices-dir INPUT/indices \
    --dataset     synonym_setB \
    --queries     OUTPUTS/ncbi_synonym_eval_setB.csv

echo ""
echo "=== Step 3: Embedding benchmark — synonym_setA (ALL-CAPS) ==="
python exp1_embedding_benchmark.py \
    --base-dir    INPUT \
    --output-dir  OUTPUTS \
    --indices-dir INPUT/indices \
    --dataset     synonym_setA \
    --queries     OUTPUTS/ncbi_synonym_eval_setA.csv

echo ""
echo "=== Synonym evaluation complete ==="
ls -lh OUTPUTS/exp1_ontology_results_synonym_*.csv
