python /home/ferles/medusa/src/ood.py \
        --ood_method baseline \
        --num_classes 8 \
        --in_distribution_dataset ISIC \
        --out_distribution_dataset Dermofit-In \
        --model_checkpoint /home/ferles/medusa/src/checkpoints/isic_classifiers/eb0Custom-best-balanced-accuracy-model_new.pth > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/In/baseline.txt
