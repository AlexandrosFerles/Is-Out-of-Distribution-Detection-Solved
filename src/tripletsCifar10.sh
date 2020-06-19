python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val cifar100 --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m baseline
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val stl --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m baseline
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val tinyimagenet --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m baseline
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val svhn --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m baseline

python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val cifar100 --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m odin
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val stl --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m odin
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val tinyimagenet --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m odin
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val svhn --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m odin

python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val cifar100 --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m mahalanobis
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val stl --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m mahalanobis
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val tinyimagenet --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m mahalanobis
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val svhn --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --m mahalanobis

python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val cifar100 --mc /raid/ferles/checkpoints/eb0/cifar10/rot_eb0Cifar10.pth --m rotation
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val stl --mc /raid/ferles/checkpoints/eb0/cifar10/rot_eb0Cifar10.pth --m rotation
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val tinyimagenet --mc /raid/ferles/checkpoints/eb0/cifar10/rot_eb0Cifar10.pth --m rotation
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val svhn --mc /raid/ferles/checkpoints/eb0/cifar10/rot_eb0Cifar10.pth --m rotation

python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val cifar100 --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --m rotation
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val stl --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --m rotation
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val tinyimagenet --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --m rotation
python ood_triplets.py --in cifar10 --nc 10 --in cifar10 --val svhn --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --m rotation

python ood_triplets.py --m ensemble --mcf ensemblesCifar.txt --nc 8 --in cifar10 --val cifar100
python ood_triplets.py --m ensemble --mcf ensemblesCifar.txt --nc 8 --in cifar10 --val stl
python ood_triplets.py --m ensemble --mcf ensemblesCifar.txt --nc 8 --in cifar10 --val tinyimagenet
python ood_triplets.py --m ensemble --mcf ensemblesCifar.txt --nc 8 --in cifar10 --val svhn
