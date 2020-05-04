#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/gen_odin_cifar_2.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/gen_odin_cifar_2.err
#SBATCH --job-name gen_odin_cifar_2

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method generalized-odin \
              --num_classes 10 \
              --in_distribution_dataset Cifar10 \
              --out_distribution_dataset TinyImagenet \
              --model_checkpoint genOdinCifar2.pth \
              --gen_odin_mode 2 \
              --batch_size 32 > results/txts/gen_odin_cifar_2.txt
