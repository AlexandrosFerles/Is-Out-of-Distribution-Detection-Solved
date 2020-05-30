python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100 --in cifar100  --val cifar10  --dv 5 > results/baseline_cifar100val_cifar10.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100 --in cifar100  --val cifar10  --dv 5 > results/odin_cifar100val_cifar10.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100  --in cifar100  --val cifar10  --dv 5 --bs 10 > results/mahalanobis_cifar100val_cifar10.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/cifar100/rot_eb0Cifar100.pth --nc 100  --in cifar100  --val cifar10  --dv 5 --bs 1 > results/rotation_cifar100val_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar100/genOdinCifar100.pth --nc 100 --in cifar100  --val cifar10  --dv 5 > results/genodin_cifar100val_cifar10.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100 --in cifar100  --val svhn  --dv 5 > results/baseline_cifar100val_svhn.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100 --in cifar100  --val svhn  --dv 5 > results/odin_cifar100val_svhn.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100  --in cifar100  --val svhn  --dv 5 --bs 10 > results/mahalanobis_cifar100val_svhn.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/cifar100/rot_eb0Cifar100.pth --nc 100  --in cifar100  --val svhn  --dv 5 --bs 1 > results/rotation_cifar100val_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar100.pth --nc 100 --in cifar100 --val svhn  --dv 5 > results/genodin_cifar100val_svhn.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100 --in cifar100  --val stl  --dv 5 > results/baseline_cifar100val_stl.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100 --in cifar100  --val stl  --dv 5 > results/odin_cifar100val_stl.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100  --in cifar100  --val stl  --dv 5 --bs 10 > results/mahalanobis_cifar100val_stl.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/cifar100/rot_eb0Cifar100.pth --nc 100  --in cifar100  --val stl  --dv 5 --bs 1 > results/rotation_cifar100val_stl.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar100.pth --nc 100 --in cifar100 --val stl  --dv 5 > results/genodin_cifar100val_stl.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100 --in cifar100  --val tinyimagenet  --dv 5 > results/baseline_cifar100val_tinyimagenet.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100 --in cifar100  --val tinyimagenet  --dv 5 > results/odin_cifar100val_tinyimagenet.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 100  --in cifar100  --val tinyimagenet  --dv 5 --bs 10 > results/mahalanobis_cifar100val_tinyimagenet.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/cifar100/rot_eb0Cifar100.pth --nc 100  --in cifar100  --val tinyimagenet  --dv 5 --bs 1 > results/rotation_cifar100val_tinyimagenet.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar100.pth --nc 100 --in cifar100 --val tinyimagenet  --dv 5 > results/genodin_cifar100val_tinyimagenet.txt