python ood.py --in isic --val isic --out dermofit --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_NV_step_lr_cutout_eb0-best-model.pth --m baseline --nc 7 --bs 50 --ex NV --dv 3 > baseline_val_NV.txt
python ood.py --in isic --val isic --out dermofit --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_NV_step_lr_cutout_eb0-best-model.pth --m odin --nc 7 --bs 50 --ex NV --dv 3 > odin_val_NV.txt
python ood.py --in isic --val isic --out dermofit --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_NV_step_lr_cutout_eb0-best-model.pth --m mahalanobis --nc 7 --bs 50 --ex NV --dv 3 > mahalanobis_val_NV.txt
python ood.py --in isic --val isic --out dermofit --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-_NV-best-model.pth --m rotation --nc 7 --bs 1 --ex NV --dv 3 > rotation_val_NV.txt
python ood.py --in isic --val isic --out dermofit --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISIC_NV-best-model.pth --m generalizedodin --nc 7  --ex NV --dv 3 > GenOdin_val_NV.txt