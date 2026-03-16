#!/bin/bash
#SBATCH --job-name=exp2_mimic_setB
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err

source "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}/.venv/bin/activate"
cd "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}"

# SetA already done by Exp1 — only run setB here
python exp1_embedding_benchmark.py \
    --base-dir INPUT \
    --output-dir OUTPUTS \
    --indices-dir INPUT/indices \
    --dataset mimic_setB \
    --queries OUTPUTS/mimic_eval_set_b.csv
