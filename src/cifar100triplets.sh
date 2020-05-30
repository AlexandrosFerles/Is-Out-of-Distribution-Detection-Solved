python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --in cifar100 --nc 10 --val cifar10  --dv 4 > results/baseline_cifar100val_cifar10.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --in cifar100 --nc 10 --val cifar10  --dv 4 > results/odin_cifar100val_cifar10.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 10 --in cifar100 --val cifar10  --dv 4 --bs 10 > results/mahalanobis_cifar100val_cifar10.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/svhn/rot_eb0SVHN.pth --nc 10 --in cifar100 --val cifar10  --dv 4 --bs 1 > results/rotation_cifar100val_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/svhn/genOdinSVHN.pth --nc 10 --in cifar100 --val cifar10  --dv 4 > results/genodin_cifar100val_cifar10.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --in cifar100 --nc 10 --val svhn  --dv 4 > results/baseline_cifar100val_svhn.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --in cifar100 --nc 10 --val svhn  --dv 4 > results/odin_cifar100val_svhn.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 10 --in cifar100 --val svhn  --dv 4 --bs 10 > results/mahalanobis_cifar100val_svhn.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/svhn/rot_eb0SVHN.pth --nc 10 --in cifar100 --val svhn  --dv 4 --bs 1 > results/rotation_cifar100val_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --in cifar100 --val svhn  --dv 4 > results/genodin_cifar100val_svhn.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --in cifar100 --nc 10 --val stl  --dv 4 > results/baseline_cifar100val_stl.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --in cifar100 --nc 10 --val stl  --dv 4 > results/odin_cifar100val_stl.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 10 --in cifar100 --val stl  --dv 4 --bs 10 > results/mahalanobis_cifar100val_stl.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/svhn/rot_eb0SVHN.pth --nc 10 --in cifar100 --val stl  --dv 4 --bs 1 > results/rotation_cifar100val_stl.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --in cifar100 --val stl  --dv 4 > results/genodin_cifar100val_stl.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --in cifar100 --nc 10 --val tiny  --dv 4 > results/baseline_cifar100val_tiny.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --in cifar100 --nc 10 --val tiny  --dv 4 > results/odin_cifar100val_tiny.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar100/eb0Cifar100.pth --nc 10 --in cifar100 --val tiny  --dv 4 --bs 10 > results/mahalanobis_cifar100val_tiny.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/svhn/rot_eb0SVHN.pth --nc 10 --in cifar100 --val tiny  --dv 4 --bs 1 > results/rotation_cifar100val_tiny.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --in cifar100 --val tiny  --dv 4 > results/genodin_cifar100val_tiny.txt