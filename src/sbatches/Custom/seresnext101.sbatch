#!/usr/bin/env bash
#SBATCH --mem  12GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/seresnext101.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/seresnext101.err
#SBATCH --job-name seresnext101

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/custom_train.py --c /Midgard/home/ferles/Dermatology/src/configs/Custom_ISIC/SE-ResNext101.json
