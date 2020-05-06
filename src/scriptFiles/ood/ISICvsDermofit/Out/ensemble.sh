python /home/ferles/medusa/src/ood.py \
        --ood_method ensemble \
        --num_classes 7 \
        --in_distribution_dataset ISIC \
        --out_distribution_dataset Dermofit-Out \
        --dv $DEV \
        --model_checkpoints_file /home/ferles/medusa/src/EnsembleCheckpoints.txt > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/Out/ensemble.tx