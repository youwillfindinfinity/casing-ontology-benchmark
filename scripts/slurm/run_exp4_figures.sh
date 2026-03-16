#!/bin/bash
#SBATCH --job-name=exp4_figures
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp4_%j.out
#SBATCH --error=logs/exp4_%j.err

source "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}/.venv/bin/activate"
cd "${PROJECT_ROOT:?export PROJECT_ROOT to your repository checkout}"

python exp4_figures.py \
    --results-dir OUTPUTS \
    --figures-dir FIGURES \
    --format pdf
