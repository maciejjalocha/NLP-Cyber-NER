#!/bin/bash
#SBATCH --job-name=lst_ner_attackner_roberta
#SBATCH --chdir=/home/jhsc/NLP-Cyber-NER-latest
#SBATCH --output=logs/lst_ner_roberta_attackner_%j.out
#SBATCH --error=logs/lst_ner_roberta_attackner_%j.err
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --exclude=cn6

module load GCCcore/12.3.0
module load CUDA/12.1.

~/miniconda3/envs/cyber-ner/bin/python -m models.train_lst_ner_attackner
