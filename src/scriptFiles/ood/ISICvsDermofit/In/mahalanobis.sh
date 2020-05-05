python /home/ferles/medusa/src/ood.py \
        --ood_method mahalanaobis \
        --num_classes 8 \
        --in_distribution_dataset ISIC \
        --out_distribution_dataset Dermofit-In \
        --model_checkpoint /home/ferles/medusa/src/checkpoints/isic_classifiers/eb0Custom-best-balanced-accuracy-model_new.pth \
        --with_FGSM True \
        --batch_size 10 > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/In/mahalanobis.txt