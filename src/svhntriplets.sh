python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn  --val cifar100  --dv 5 > results/baseline_svhnval_cifar100.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn  --val cifar100  --dv 5 > results/odin_svhnval_cifar100.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn --val cifar100  --dv 5 --bs 10 > results/mahalanobis_svhnval_cifar100.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/svhn/rot_eb0SVHN.pth --nc 10 --in svhn --val cifar100  --dv 5 --bs 1 > results/rotation_svhn100val_cifar100.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/svhn/genOdinSVHN.pth --nc 10 --in svhn --val cifar100  --dv 5 > results/genodin_svhn100val_cifar100.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn  --val cifar10  --dv 5 > results/baseline_svhnval_cifar10.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn  --val cifar10  --dv 5 > results/odin_svhnval_cifar10.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn --val cifar10  --dv 5 --bs 10 > results/mahalanobis_svhnval_cifar10.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/svhn/rot_eb0SVHN.pth --nc 10 --in svhn --val cifar10  --dv 5 --bs 1 > results/rotation_svhn100val_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --nc 10 --in svhn --val cifar10  --dv 5 > results/genodin_svhn100val_cifar10.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn  --val stl  --dv 5 > results/baseline_svhnval_stl.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn  --val stl  --dv 5 > results/odin_svhnval_stl.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn --val stl  --dv 5 --bs 10 > results/mahalanobis_svhnval_stl.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/svhn/rot_eb0SVHN.pth --nc 10 --in svhn --val stl  --dv 5 --bs 1 > results/rotation_svhn100val_stl.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --nc 10 --in svhn --val stl  --dv 5 > results/genodin_svhn100val_stl.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn  --val tiny  --dv 5 > results/baseline_svhnval_tiny.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn  --val tiny  --dv 5 > results/odin_svhnval_tiny.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/svhn/eb0SVHN.pth --nc 10 --in svhn --val tiny  --dv 5 --bs 10 > results/mahalanobis_svhnval_tiny.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/svhn/rot_eb0SVHN.pth --nc 10 --in svhn --val tiny  --dv 5 --bs 1 > results/rotation_svhn100val_tiny.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar10.pth --nc 10 --in svhn --val tiny  --dv 5 > results/genodin_svhn100val_tiny.txt