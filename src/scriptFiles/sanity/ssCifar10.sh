#!/usr/bin/env bash
#SBATCH --mem  16GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 2
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/ebNetRot.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/ebNetRot.err
#SBATCH --job-name ebNetRot
#SBATCH --constrain rivendell|khazadum|shire|gondor

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/EBnetsRotation.py --c /Midgard/home/ferles/Dermatology/src/configs/sanity/Eb0Cifar10Rotation.json
