#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/exclude_MEL_step_lr_cutout_gt.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/exclude_MEL_step_lr_cutout_gt.err
#SBATCH --job-name exclude_MEL_step_lr_cutout_gt

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/exclude_train.py --c /Midgard/home/ferles/Dermatology/src/configs/exlude_ISIC/MEL_exclude.json
