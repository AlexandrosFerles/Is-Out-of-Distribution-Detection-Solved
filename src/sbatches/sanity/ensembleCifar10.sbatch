#!/usr/bin/env bash
#SBATCH --mem  5GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/EnsembleCifar10.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/EnsembleCifar10.err
#SBATCH --job-name EnsembleCifar10

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/Cifar-10-Leave_Out_Ensemble.py --c /Midgard/home/ferles/Dermatology/src/configs/sanity/Cifar10Ensemble1.json
