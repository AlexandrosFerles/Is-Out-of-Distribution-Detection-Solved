#!/usr/bin/env bash
#SBATCH --mem  5GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/dense2.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/dense2.err
#SBATCH --job-name dense2
#SBATCH --constrain rivendell|khazadum|shire|gondor

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/SanityGenODIN.py --mode 2
