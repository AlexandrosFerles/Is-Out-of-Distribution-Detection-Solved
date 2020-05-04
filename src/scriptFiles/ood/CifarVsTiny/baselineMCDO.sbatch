#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/baseline_cifar10_tinyimagenet_mcdo.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/baseline_cifar10_tinyimagenet_mcdo.err
#SBATCH --job-name baseline_cifar10_tinyimagenet_mcdo

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method baseline \
              --num_classes 10 \
              --in_distribution_dataset Cifar10 \
              --out_distribution_dataset TinyImagenet \
              --model_checkpoint Cifar10_seed1.pth \
              --monte_carlo_steps 100 > results/txts/baseline_sanity_mcdo.txt
