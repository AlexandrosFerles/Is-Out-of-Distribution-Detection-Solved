#!/usr/bin/env bash
#SBATCH --mem  12GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/BKL_custom_oexposure_eb0.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/BKL_custom_oexposure_eb0.err
#SBATCH --job-name BKL_custom_oexposure_eb0

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/ensemble.py --c /Midgard/home/ferles/Dermatology/src/configs/OExposure/BKL_exclude.json
