#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/baseline_VASC.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/baseline_VASC.err
#SBATCH --job-name baseline_VASC

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method baseline \
              --num_classes 7 \
              --in_distribution_dataset ISIC \
              --out_distribution_dataset ISIC \
              --model_checkpoint checkpoints/isic_classifiers/custom_exclude_VASC_step_lr_cutout_eb0-best-balanced-accuracy-model_new.pth \
              --exclude_class VASC > results/txts/ISICvsISIC/VASC/baseline_VASC_new_bac.txt
