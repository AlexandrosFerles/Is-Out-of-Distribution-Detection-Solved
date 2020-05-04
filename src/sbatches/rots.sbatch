#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/rot_exclude_AK_step_lr_cutout_gt.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/rot_exclude_AK_step_lr_cutout_gt.err
#SBATCH --job-name rot_exclude_AK_step_lr_cutout_gt

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python score_ss_rot.py --mc checkpoints/exclude_AK_step_lr_cutout_eb0_rot-best-val-loss-model.pth --ex AK
python score_ss_rot.py --mc checkpoints/exclude_BCC_step_lr_cutout_eb0_rot-best-val-loss-model.pth --ex BCC
python score_ss_rot.py --mc checkpoints/exclude_BKL_step_lr_cutout_eb0_rot-best-val-loss-model.pth --ex BKL
python score_ss_rot.py --mc checkpoints/exclude_DF_step_lr_cutout_eb0_rot-best-val-loss-model.pth --ex DF
python score_ss_rot.py --mc checkpoints/exclude_MEL_step_lr_cutout_eb0_rot-best-val-loss-model.pth --ex MEL
python score_ss_rot.py --mc checkpoints/exclude_NV_step_lr_cutout_eb0_rot-best-val-loss-model.pth --ex NV
python score_ss_rot.py --mc checkpoints/exclude_SCC_step_lr_cutout_eb0_rot-best-val-loss-model.pth --ex SCC
python score_ss_rot.py --mc checkpoints/exclude_VASC_step_lr_cutout_eb0_rot-best-val-loss-model.pth --ex VASC
