#!/usr/bin/env bash
. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /home/ferles/medusa/src/custom_train.py --c /home/ferles/medusa/src/configs/Custom_ISIC/MEL_exclude.json
