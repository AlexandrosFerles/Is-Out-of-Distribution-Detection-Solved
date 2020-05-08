#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/mahalanobis_cifar.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/mahalanobis_cifar.err
#SBATCH --job-name mahalanobis_cifar

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method mahalanobis \
              --num_classes 10 \
              --in_distribution_dataset Cifar10 \
              --out_distribution_dataset TinyImagenet \
              --model_checkpoint Cifar10_seed1.pth \
              --with_FGSM True \
              --batch_size 10 > results/txts/mahalanobis.txt
