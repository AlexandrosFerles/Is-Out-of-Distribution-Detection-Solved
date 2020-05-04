#!/usr/bin/env bash
#SBATCH --mem  12GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/eb6_focal.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/eb6_focal.err
#SBATCH --job-name eb6_focal

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/custom_train.py --c /Midgard/home/ferles/Dermatology/src/configs/Custom_ISIC/custom_EfficientNetB6.json
