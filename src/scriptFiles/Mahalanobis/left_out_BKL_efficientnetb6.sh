#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/leftout_BKL_step_lr_cutout_gt.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/leftout_BKL_step_lr_cutout_gt.err
#SBATCH --job-name leftout_BKL_step_lr_cutout_gt

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ../val_left_out_ood.py --mc checkpoints/exclude_BKL_step_lr_cutout-best-balanced-accuracy-model.pth --exclude_class BKL
