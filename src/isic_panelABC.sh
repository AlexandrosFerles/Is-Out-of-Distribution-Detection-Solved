#python ood.py --m baseline --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out places --nc 8 --dv 2  > results/isic/panelA/baseline.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out places --nc 8 --dv 2  > results/isic/panelA/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out places --nc 8 --dv 2  --bs 20  > results/isic/panelA/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-best-model.pth --in isic --val imagenet --out places --nc 8 --dv 2  --bs 1 > results/isic/panelA/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISICeb0-best-model.pth --in isic --val imagenet --out places --nc 8 --dv 2  > results/isic/panelA/generalizedodin.txt
#
#python ood.py --m baseline --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out dermofit-in --nc 8 --dv 2  > results/isic/panelB/baseline.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out dermofit-in --nc 8 --dv 2  > results/isic/panelB/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out dermofit-in --nc 8 --dv 2  --bs 20  > results/isic/panelB/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-best-model.pth --in isic --val imagenet --out dermofit-in --nc 8 --dv 2  --bs 1 > results/isic/panelB/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISICeb0-best-model.pth --in isic --val imagenet --out dermofit-in --nc 8 --dv 2  > results/isic/panelB/generalizedodin.txt

python ood.py --m baseline --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out dermofit-out --nc 8 --dv 2  > results/isic/panelC/baseline.txt
python ood.py --m odin --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out dermofit-out --nc 8 --dv 2  > results/isic/panelC/odin.txt
python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/isic_classifiers/eb0CustomNoPreproc-best-model.pth --in isic --val imagenet --out dermofit-out --nc 8 --dv 2  --bs 20  > results/isic/panelC/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-best-model.pth --out isic --in imagenet --out dermofit-out --nc 8 --dv 2  --bs 1 > results/isic/panelC/rotation.txt
python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISICeb0-best-model.pth --in isic --val imagenet --out dermofit-out --nc 8 --dv 2  > results/isic/panelC/generalizedodin.txt
