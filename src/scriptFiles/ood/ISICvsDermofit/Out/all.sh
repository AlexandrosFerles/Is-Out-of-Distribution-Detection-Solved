python /home/ferles/medusa/src/ood.py \
        --ood_method baseline \
        --num_classes 8 \
        --in_distribution_dataset ISIC \
        --out_distribution_dataset Dermofit-out \
        --model_checkpoint /home/ferles/medusa/src/checkpoints/isic_classifiers/eb0Custom-best-balanced-accuracy-model_new.pth > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/Out/baseline.txt

python /home/ferles/medusa/src/ood.py \
          --ood_method odin \
          --num_classes 8 \
          --in_distribution_dataset ISIC \
          --out_distribution_dataset Dermofit-out \
          --model_checkpoint /home/ferles/medusa/src/checkpoints/isic_classifiers/eb0Custom-best-balanced-accuracy-model_new.pth \
          --temperature 1000 \
          --epsilon 0.0014 > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/Out/odin_best_paper.txt

python /home/ferles/medusa/src/ood.py \
        --ood_method odin \
        --num_classes 8 \
        --in_distribution_dataset ISIC \
        --out_distribution_dataset Dermofit-out \
        --model_checkpoint /home/ferles/medusa/src/checkpoints/isic_classifiers/eb0Custom-best-balanced-accuracy-model_new.pth \
        --with_FGSM True > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/Out/odin_fgsm.txt

python /home/ferles/medusa/src/ood.py \
        --ood_method mahalanaobis \
        --num_classes 8 \
        --in_distribution_dataset ISIC \
        --out_distribution_dataset Dermofit-out \
        --model_checkpoint /home/ferles/medusa/src/checkpoints/isic_classifiers/eb0Custom-best-balanced-accuracy-model_new.pth \
        --with_FGSM True \
        --batch_size 10 > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/Out/mahalanobis.txt

python /home/ferles/medusa/src/ood.py \
        --ood_method selfSupervised \
        --num_classes 8 \
        --in_distribution_dataset ISIC \
        --out_distribution_dataset Dermofit-out \
        --model_checkpoint /home/ferles/medusa/src/checkpoints/eb0_rot-best-balanced-accuracy-model.pth \
        --batch_size 1 > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/Out/selfSupervised.txt

