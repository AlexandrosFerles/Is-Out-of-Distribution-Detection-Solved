#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/baseline_NV_mcdo.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/baseline_NV_mcdo.err
#SBATCH --job-name baseline_NV_mcdo

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method baseline \
              --num_classes 7 \
              --in_distribution_dataset ISIC \
              --out_distribution_dataset ISIC \
              --model_checkpoint custom_exclude_NV_step_lr_cutout_eb0-best-balanced-accuracy-model.pth \
              --exclude_class NV \
              --monte_carlo_steps 100 > results/txts/ISICvsISIC/NV/baseline_NV_mcdo.txt