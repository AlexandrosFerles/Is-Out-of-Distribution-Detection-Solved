#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/exclude_NV_step_lr_cutout_gt.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/exclude_NV_step_lr_cutout_gt.err
#SBATCH --job-name exclude_NV_step_lr_cutout_gt
#SBATCH --constrain rivendell|shire|khazadum|gondor

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/exclude_train.py --c /Midgard/home/ferles/Dermatology/src/configs/exlude_ISIC/NV_exclude.json
