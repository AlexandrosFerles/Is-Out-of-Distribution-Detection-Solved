#!/usr/bin/env bash
#SBATCH --mem  8GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/baseline-monte-carlo-MEL.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/baseline-monte-carlo-MEL.err
#SBATCH --job-name baseline-monte-carlo-MEL

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/baseline.py --mc /Midgard/home/ferles/Dermatology/src/checkpoints/custom_exclude_MEL_step_lr_cutout_eb0-best-balanced-accuracy-model.pth --ex MEL
