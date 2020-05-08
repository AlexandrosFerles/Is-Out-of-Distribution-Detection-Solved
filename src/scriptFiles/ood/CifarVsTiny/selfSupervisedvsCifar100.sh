#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/selfSupervisedEb0Cifar10vsCifar100.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/selfSupervisedEb0CifarvsCifar100.err
#SBATCH --job-name selfSupervisedEb0CifarvsCifar100 

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method self-supervision \
              --num_classes 10 \
              --in_distribution_dataset Cifar10 \
              --out_distribution_dataset Cifar100 \
              --model_checkpoint sanity/eb0Cifar10rotationBestLoss.pth \
              --batch_size 1 > results/txts/selfSupervisedEb0CifarvsCifar100.txt
