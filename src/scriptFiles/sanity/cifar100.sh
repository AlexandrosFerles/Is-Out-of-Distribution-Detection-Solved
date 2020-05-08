#!/usr/bin/env bash
#SBATCH --mem  5GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/cifar100.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/cifar100.err
#SBATCH --job-name cifar100

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/Cifars.py --c /Midgard/home/ferles/Dermatology/src/configs/sanity/Eb0Cifar100.json
