#!/usr/bin/env bash
#SBATCH --mem  12GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 2
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/genOdinEB0AK.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/genOdinEB0AK.err
#SBATCH --job-name genOdinEB0AK

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/generalized_odin_train.py --c /Midgard/home/ferles/Dermatology/src/configs/Gen_odin/gen_odin_eb0_AK.json
