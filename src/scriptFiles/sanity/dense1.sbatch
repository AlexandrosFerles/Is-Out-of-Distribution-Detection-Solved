#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/dense1.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/dense1.err
#SBATCH --job-name dense1
#SBATCH --constrain rivendell|khazadum|gondor|shire

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/SanityGenODIN.py --mode 1
