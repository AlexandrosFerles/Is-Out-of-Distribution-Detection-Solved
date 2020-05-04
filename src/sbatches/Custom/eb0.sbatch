#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/eb0Custom.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/eb0Custom.err
#SBATCH --job-name eb0Custom

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/custom_train.py --c /Midgard/home/ferles/Dermatology/src/configs/Custom_ISIC/eb0.json
