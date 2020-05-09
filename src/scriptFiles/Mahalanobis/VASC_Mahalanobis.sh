#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/gen_mah_VASC.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/gen_mah_VASC.err
#SBATCH --job-name mah_VASC

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python eval_left_out_ood.py --mc checkpoints/exclude_VASC_step_lr_cutout-best-balanced-accuracy-model.pth --ex VASC