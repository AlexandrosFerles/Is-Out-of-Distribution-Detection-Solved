#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/ensemble_accuracy.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/ensemble_accuracy.err
#SBATCH --job-name ensemble_accuracy

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method ensemble \
              --num_classes 10 \
              --in_distribution_dataset Cifar10 \
              --out_distribution_dataset TinyImagenet \
              --model_checkpoints_file Ensemble_checkpoints.txt \
              --ensemble_mode accuracy \
              --batch_size 32 > results/txts/ensemble_accuracy.txt
