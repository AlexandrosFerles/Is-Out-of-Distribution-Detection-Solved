#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user ferles@kth.se
#SBATCH --output /Midgard/home/%u/Dermatology/logs/odin_SCC_fgsm.out
#SBATCH --error  /Midgard/home/%u/Dermatology/logs/odin_SCC_fgsm.err
#SBATCH --job-name odin_SCC_fgsm

. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python ood.py --ood_method odin \
              --num_classes 7 \
              --in_distribution_dataset ISIC \
              --out_distribution_dataset ISIC \
              --model_checkpoint custom_exclude_SCC_step_lr_cutout_eb0-best-balanced-accuracy-model.pth \
              --exclude_class SCC \
              --with_FGSM True > results/txts/ISICvsISIC/SCC/odin_fgsm_SCC.txt