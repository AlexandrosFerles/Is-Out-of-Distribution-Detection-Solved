#!/usr/bin/env bash
#SBATCH --mem  8GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/dense0%J.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/dense0%J.err
#SBATCH --job-name dense0

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/SanityGenODIN.py
