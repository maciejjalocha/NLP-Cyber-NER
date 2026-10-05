#!/bin/bash
#SBATCH --job-name=bert_baseline_cyner_roberta
#SBATCH --chdir=/home/jhsc/NLP-Cyber-NER-latest
#SBATCH --output=logs/bert_baseline_roberta_cyner_%j.out
#SBATCH --error=logs/bert_baseline_roberta_cyner_%j.err
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --exclude=cn6

module load GCCcore/12.3.0
module load CUDA/12.1.

~/miniconda3/envs/cyber-ner/bin/python -m models.BERT_CYNER
