python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --in cifar10 --val cifar100 --nc 10 --dv 4 > results/genOdin_cifar10_val_cifar100.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --in cifar10 --val svhn --nc 10 --dv 4 > results/genOdin_cifar10_val_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --in cifar10 --val stl --nc 10 --dv 4 > results/genOdin_cifar10_val_stl.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --in cifar10 --val tinyimagenet --nc 10 --dv 4 > results/genOdin_cifar10_val_tinyimagenet.txt

python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar100/genOdinCifar100.pth --in cifar100 --val cifar10 --nc 10 --dv 4 > results/genOdin_cifar100_val_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar100/genOdinCifar100.pth --in cifar100 --val svhn --nc 10 --dv 4 > results/genOdin_cifar100_val_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar100/genOdinCifar100.pth --in cifar100 --val stl --nc 10 --dv 4 > results/genOdin_cifar100_val_stl.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar100/genOdinCifar100.pth --in cifar100 --val tinyimagenet --nc 10 --dv 4 > results/genOdin_cifar100_val_tinyimagenet.txt

python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/svhn/genOdinSVHN.pth --in svhn --val cifar10 --nc 10 --dv 4 > results/genOdin_svhn_val_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/svhn/genOdinSVHN.pth --in svhn --val cifar100 --nc 10 --dv 4 > results/genOdin_svhn_val_cifar100.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/svhn/genOdinSVHN.pth --in svhn --val stl --nc 10 --dv 4 > results/genOdin_svhn_val_stl.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/svhn/genOdinSVHN.pth --in svhn --val tinyimagenet --nc 10 --dv 4 > results/genOdin_svhn_val_tinyimagenet.txt

python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stl/genOdinSTL.pth --in stl --val cifar10 --nc 10 --dv 4 > results/genOdin_stl_val_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stl/genOdinSTL.pth --in stl --val cifar100 --nc 10 --dv 4 > results/genOdin_stl_val_cifar100.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stl/genOdinSTL.pth --in stl --val svhn --nc 10 --dv 4 > results/genOdin_stl_val_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stl/genOdinSTL.pth --in stl --val tinyimagenet --nc 10 --dv 4 > results/genOdin_stl_val_tinyimagenet.txt

python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/genOdinTinyImageNet.pth --in tinyimagenet --val cifar10 --nc 10 --dv 4 > results/genOdin_tinyimagenet_val_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/genOdinTinyImageNet.pth --in tinyimagenet --val cifar100 --nc 10 --dv 4 > results/genOdin_tinyimagenet_val_cifar100.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/genOdinTinyImageNet.pth --in tinyimagenet --val svhn --nc 10 --dv 4 > results/genOdin_tinyimagenet_val_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/genOdinTinyImageNet.pth --in tinyimagenet --val stl --nc 10 --dv 4 > results/genOdin_tinyimagenet_val_stl.txt