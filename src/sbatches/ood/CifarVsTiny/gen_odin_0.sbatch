#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/gen_odin_cifar_0.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/gen_odin_cifar_0.err
#SBATCH --job-name gen_odin_cifar_0

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method generalized-odin \
              --num_classes 10 \
              --in_distribution_dataset Cifar10 \
              --out_distribution_dataset TinyImagenet \
              --model_checkpoint genOdinCifar0.pth \
              --gen_odin_mode 0 \
              --batch_size 32 > results/txts/gen_odin_cifar_0.txt
