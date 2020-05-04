#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/eval_log_regs.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/eval_log_regs.err
#SBATCH --job-name eval_log_regs

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python log_regs_predict_test_set.py --mf logistic_regressors.txt --no 7
