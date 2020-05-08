#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/gen_odin_eb0_extended.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/gen_odin_eb0_extended.err
#SBATCH --job-name gen_odin_eb0_extended

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/generalized_odin_train.py --c /Midgard/home/ferles/Dermatology/src/configs/Gen_odin/eb0.json
