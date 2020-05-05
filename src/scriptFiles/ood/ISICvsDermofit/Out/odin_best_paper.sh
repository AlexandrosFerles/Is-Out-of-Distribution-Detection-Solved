conda activate isic
python /home/ferles/medusa/src/ood.py \
          --ood_method odin \
          --num_classes 8 \
          --in_distribution_dataset ISIC \
          --out_distribution_dataset Dermofit-out \
          --model_checkpoint /home/ferles/medusa/src/checkpoints/isic_classifiers/eb0Custom-best-balanced-accuracy-model_new.pth \
          --temperature 1000 \
          --epsilon 0.0014 > /home/ferles/medusa/src/results/txts/ISICvsDermoFit/Out/odin_best_paper.txt