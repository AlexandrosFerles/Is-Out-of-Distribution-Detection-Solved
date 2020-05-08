#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/baseline_MEL.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/baseline_MEL.err
#SBATCH --job-name baseline_MEL

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method baseline \
              --num_classes 7 \
              --in_distribution_dataset ISIC \
              --out_distribution_dataset ISIC \
              --model_checkpoint checkpoints/isic_classifiers/custom_exclude_MEL_step_lr_cutout_eb0-best-auc-model_old.pth \
              --exclude_class MEL > results/txts/ISICvsISIC/MEL/baseline_MEL_old_auc.txt