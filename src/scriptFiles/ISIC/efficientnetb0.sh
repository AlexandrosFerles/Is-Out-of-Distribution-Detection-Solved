#!/usr/bin/env bash
#SBATCH --mem  12GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/efficientNetB0-step-lr-auto-cutout.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/efficientNetB0-step-lr-auto-cutout.err
#SBATCH --job-name efficientNetB0-step-lr-auto-cutout

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/train.py --config /Midgard/home/ferles/Dermatology/src/configs/ISIC/EfficientNetB0.json
