python ood.py --m baseline --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_AK_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex AK --nc 7 --dv 3  > results/isic/panelD/AK/baseline.txt
python ood.py --m odin --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_AK_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex AK --nc 7 --dv 3  > results/isic/panelD/AK/odin.txt
python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_AK_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex AK --nc 7 --dv 3  --bs 20  > results/isic/panelD/AK/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-_AK-best-model.pth --in isic --val imagenet --out isic --ex AK --nc 7 --dv 3  --bs 1 > results/isic/panelD/AK/rotation.txt
python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISICAK-best-model.pth --in isic --val imagenet --out isic --ex AK --nc 7 --dv 3  > results/isic/panelD/AK/generalizedodin.txt


