#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/senet.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/senet.err
#SBATCH --job-name SENetWCE

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ../train.py --c configs/SENet154.json
