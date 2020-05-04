#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/rot_exclude_VASC_step_lr_cutout_gt.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/rot_exclude_VASC_step_lr_cutout_gt.err
#SBATCH --job-name rot_exclude_VASC_step_lr_cutout_gt
#SBATCH --constrain rivendell|khazadum|shire|gondor

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/ss_rot.py --c /Midgard/home/ferles/Dermatology/src/configs/rot_ISIC/VASC_exclude.json
