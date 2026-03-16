#!/bin/bash
#SBATCH --job-name=exp1_embedding_bench
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err

source "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}/.venv/bin/activate"
cd "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}"

# Run all three datasets sequentially (lazy index loading — one index at a time)
for DATASET in microbiome mimic_setA mimic_setB; do
    echo "=== Dataset: $DATASET ==="
    EXTRA_ARGS=""
    if [ "$DATASET" = "mimic_setA" ]; then
        EXTRA_ARGS="--queries OUTPUTS/mimic_eval_set_a.csv"
    elif [ "$DATASET" = "mimic_setB" ]; then
        EXTRA_ARGS="--queries OUTPUTS/mimic_eval_set_b.csv"
    fi

    python exp1_embedding_benchmark.py \
        --base-dir INPUT \
        --output-dir OUTPUTS \
        --indices-dir INPUT/indices \
        --dataset "$DATASET" \
        $EXTRA_ARGS
done

echo "Exp1 complete."
