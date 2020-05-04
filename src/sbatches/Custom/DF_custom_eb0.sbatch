#!/usr/bin/env bash
#SBATCH --mem  12GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/DF_custom_eb0.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/DF_custom_eb0.err
#SBATCH --job-name DF_custom_eb0
#SBATCH --constrain rivendell|khazadum|shire|gondor

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/custom_train.py --c /Midgard/home/ferles/Dermatology/src/configs/Custom_ISIC/DF_exclude.json
