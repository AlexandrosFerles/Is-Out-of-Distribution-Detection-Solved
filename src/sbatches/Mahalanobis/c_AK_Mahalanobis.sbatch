#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/gen_mah_AK.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/gen_mah_AK.err
#SBATCH --job-name mah_AK

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/eval_left_out_ood.py --mc /Midgard/home/ferles/Dermatology/src/checkpoints/exclude_AK_step_lr_cutout_eb0-best-balanced-accuracy-model.pth --ex AK
