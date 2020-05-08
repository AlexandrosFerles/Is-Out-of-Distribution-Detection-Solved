#!/usr/bin/env bash
#SBATCH --mem  12GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/ResNet101.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/ResNet101.err
#SBATCH --job-name ResNet101

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ../train.py --config configs/ResNet101.json
