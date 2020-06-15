#python ood.py --m baseline --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_MEL_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex MEL --nc 7 --dv 2  > results/isic/panelD/MEL/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_MEL_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex MEL --nc 7 --dv 2  > results/isic/panelD/MEL/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_MEL_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex MEL --nc 7 --dv 2  > results/isic/panelD/MEL/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_MEL_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex MEL --nc 7 --dv 2  --bs 20  > results/isic/panelD/MEL/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-_MEL-best-model.pth --in isic --val imagenet --out isic --ex MEL --nc 7 --dv 2  --bs 1 > results/isic/panelD/MEL/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISIC_MEL-best-model.pth --in isic --val imagenet --out isic --ex MEL --nc 7 --dv 2  > results/isic/panelD/MEL/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_NV_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex NV --nc 7 --dv 2  > results/isic/panelD/NV/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_NV_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex NV --nc 7 --dv 2  > results/isic/panelD/NV/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_NV_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex NV --nc 7 --dv 2  > results/isic/panelD/NV/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_NV_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex NV --nc 7 --dv 2  --bs 20  > results/isic/panelD/NV/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-_NV-best-model.pth --in isic --val imagenet --out isic --ex NV --nc 7 --dv 2  --bs 1 > results/isic/panelD/NV/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISIC_NV-best-model.pth --in isic --val imagenet --out isic --ex NV --nc 7 --dv 2  > results/isic/panelD/NV/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_SCC_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex SCC --nc 7 --dv 2  > results/isic/panelD/SCC/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_SCC_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex SCC --nc 7 --dv 2  > results/isic/panelD/SCC/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_SCC_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex SCC --nc 7 --dv 2  > results/isic/panelD/SCC/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_SCC_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex SCC --nc 7 --dv 2  --bs 20  > results/isic/panelD/SCC/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-_SCC-best-model.pth --in isic --val imagenet --out isic --ex SCC --nc 7 --dv 2  --bs 1 > results/isic/panelD/SCC/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISIC_SCC-best-model.pth --in isic --val imagenet --out isic --ex SCC --nc 7 --dv 2  > results/isic/panelD/SCC/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_VASC_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex VASC --nc 7 --dv 2  > results/isic/panelD/VASC/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_VASC_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex VASC --nc 7 --dv 2  > results/isic/panelD/VASC/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_VASC_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex VASC --nc 7 --dv 2  > results/isic/panelD/VASC/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/isic_classifiers/custom_exclude_VASC_step_lr_cutout_eb0-best-model.pth --in isic --val imagenet --out isic --ex VASC --nc 7 --dv 2  --bs 20  > results/isic/panelD/VASC/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/isic_classifiers/rot_isic-_VASC-best-model.pth --in isic --val imagenet --out isic --ex VASC --nc 7 --dv 2  --bs 1 > results/isic/panelD/VASC/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/isic_classifiers/GenOdinISIC_VASC-best-model.pth --in isic --val imagenet --out isic --ex VASC --nc 7 --dv 2  > results/isic/panelD/VASC/generalizedodin.txt

