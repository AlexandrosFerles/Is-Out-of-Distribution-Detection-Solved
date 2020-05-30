python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --nc 10 --val cifar100  --dv 3 > results/odin_cifar10val_cifar100.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/cifar10/rot_eb0Cifar10.pth --nc 10 --in cifar10 --val cifar100  --dv 3 --bs 1 > results/rotation_cifar10val_cifar100.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --nc 10 --val svhn  --dv 3 > results/baseline_cifar10val_svhn.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --nc 10 --val svhn  --dv 3 > results/odin_cifar10val_svhn.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --nc 10 --in cifar10 --val svhn  --dv 3 --bs 10 > results/mahalanobis_cifar10val_svhn.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/cifar10/rot_eb0Cifar10.pth --nc 10 --in cifar10 --val svhn  --dv 3 --bs 1 > results/rotation_cifar10val_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --nc 10 --in cifar10 --val svhn  --dv 3 > results/genodin_cifar10val_svhn.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --nc 10 --val stl  --dv 3 > results/baseline_cifar10val_stl.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --nc 10 --val stl  --dv 3 > results/odin_cifar10val_stl.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --nc 10 --in cifar10 --val stl  --dv 3 --bs 10 > results/mahalanobis_cifar10val_stl.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/cifar10/rot_eb0Cifar10.pth --nc 10 --in cifar10 --val stl  --dv 3 --bs 1 > results/rotation_cifar10val_stl.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --nc 10 --in cifar10 --val stl  --dv 3 > results/genodin_cifar10val_stl.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --nc 10 --val tiny  --dv 3 > results/baseline_cifar10val_tiny.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --nc 10 --val tiny  --dv 3 > results/odin_cifar10val_tiny.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --nc 10 --in cifar10 --val tiny  --dv 3 --bs 10 > results/mahalanobis_cifar10val_tiny.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/cifar10/rot_eb0Cifar10.pth --nc 10 --in cifar10 --val tiny  --dv 3 --bs 1 > results/rotation_cifar10val_tiny.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --nc 10 --in cifar10 --val tiny  --dv 3 > results/genodin_cifar10val_tiny.txt