python /home/ferles/medusa/src/ood.py \
        --ood_method self-supervision \
        --num_classes 8 \
        --in_distribution_dataset ISIC \
        --out_distribution_dataset Dermofit-In \
        --model_checkpoint /home/ferles/medusa/src/checkpoints/eb0_rot-best-balanced-accuracy-model.pth \
        --batch_size 1 > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/In/selfSupervised.txt