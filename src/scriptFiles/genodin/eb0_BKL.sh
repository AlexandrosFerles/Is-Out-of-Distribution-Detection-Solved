#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/genOdinEB0BKL.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/genOdinEB0BKL.err
#SBATCH --job-name genOdinEB0BKL

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/generalized_odin_train.py --c /Midgard/home/ferles/Dermatology/src/configs/Gen_odin/gen_odin_eb0_BKL.json
