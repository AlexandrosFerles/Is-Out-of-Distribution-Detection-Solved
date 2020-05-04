#!/usr/bin/env bash
#SBATCH --mem  12GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/eb0_rot.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/eb0_rot.err
#SBATCH --job-name eb0_rot
#SBATCH --constrain rivendell|khazadum|shire|gondor

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/ss_rot.py --c /Midgard/home/ferles/Dermatology/src/configs/rot_ISIC/eb0.json
