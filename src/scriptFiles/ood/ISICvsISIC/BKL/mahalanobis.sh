#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/mahalanobis_BKL.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/mahalanobis_BKL.err
#SBATCH --job-name mahalanobis_BKL

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method mahalanobis \
              --num_classes 7 \
              --in_distribution_dataset ISIC \
              --out_distribution_dataset ISIC \
              --model_checkpoint custom_exclude_BKL_step_lr_cutout_eb0-best-balanced-accuracy-model.pth \
              --exclude_class BKL \
              --with_FGSM True \
              --batch_size 10 > results/txts/ISICvsISIC/BKL/mahalanobis_BKL.txt