#!/usr/bin/env bash
. ~/anaconda3/etc/profile.d/conda.sh
conda activate isic
python /Midgard/home/ferles/Dermatology/src/custom_train.py --c /Midgard/home/ferles/Dermatology/src/configs/Custom_ISIC/MEL_exclude.json
