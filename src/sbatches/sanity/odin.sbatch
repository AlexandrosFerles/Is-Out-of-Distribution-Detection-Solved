#!/usr/bin/env bash
#SBATCH --mem  5GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-type END
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/sanity_odin.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/sanity_odin.err
#SBATCH --job-name sanity_odin

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/ood_cifar_imagenet.py --mc /Midgard/home/ferles/Dermatology/src/checkpoints/eb0Cifar10.pth
