#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/AK_custom_oexposure_eb0.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/AK_custom_oexposure_eb0.err
#SBATCH --job-name AK_custom_oexposure_eb0

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/ensemble.py --c /Midgard/home/ferles/Dermatology/src/configs/OExposure/AK_exclude.json
