#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/odin_cifar10_tinyimagenet_best_paper.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/odin_cifar10_tinyimagenet_best_paper.err
#SBATCH --job-name odin_cifar10_tinyimagenet_best_paper

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method odin \
              --num_classes 10 \
              --in_distribution_dataset Cifar10 \
              --out_distribution_dataset TinyImagenet \
              --model_checkpoint Cifar10_seed1.pth \
              --temperature 1000 \
              --epsilon 0.0014 > results/txts/odin_best_paper.txt
