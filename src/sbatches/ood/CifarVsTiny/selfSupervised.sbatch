#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/selfSupervisedEb0Cifar.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/selfSupervisedEb0Cifar.err
#SBATCH --job-name selfSupervisedEb0Cifar

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method self-supervision \
              --num_classes 10 \
              --in_distribution_dataset Cifar10 \
              --out_distribution_dataset TinyImagenet \
              --model_checkpoint eb0Cifar10rotationSeed1.pth \
              --batch_size 1 > results/txts/selfSupervisedEb0Cifar.txt
