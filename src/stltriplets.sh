python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10 --in stl  --val cifar10  --dv 6 > results/baseline_stlval_cifar10.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10 --in stl  --val cifar10  --dv 6 > results/odin_stlval_cifar10.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10  --in stl  --val cifar10  --dv 6 --bs 10 > results/mahalanobis_stlval_cifar10.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/stl/rot_STL.pth --nc 10  --in stl  --val cifar10  --dv 6 --bs 1 > results/rotation_stlval_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stl/genOdinSTL.pth --nc 10 --in stl  --val cifar10  --dv 6 > results/genodin_stlval_cifar10.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10 --in stl  --val svhn  --dv 6 > results/baseline_stlval_svhn.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10 --in stl  --val svhn  --dv 6 > results/odin_stlval_svhn.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10  --in stl  --val svhn  --dv 6 --bs 10 > results/mahalanobis_stlval_svhn.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/stl/rot_STL.pth --nc 10  --in stl  --val svhn  --dv 6 --bs 1 > results/rotation_stlval_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar100.pth --nc 10 --in stl --val svhn  --dv 6 > results/genodin_stlval_svhn.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10 --in stl  --val cifar100  --dv 6 > results/baseline_stlval_cifar100.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10 --in stl  --val cifar100  --dv 6 > results/odin_stlval_cifar100.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10  --in stl  --val cifar100  --dv 6 --bs 10 > results/mahalanobis_stlval_cifar100.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/stl/rot_STL.pth --nc 10  --in stl  --val cifar100  --dv 6 --bs 1 > results/rotation_stlval_cifar100.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar100.pth --nc 10 --in stl --val cifar100  --dv 6 > results/genodin_stlval_cifar100.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10 --in stl  --val tinyimagenet  --dv 6 > results/baseline_stlval_tinyimagenet.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10 --in stl  --val tinyimagenet  --dv 6 > results/odin_stlval_tinyimagenet.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stl/STL.pth --nc 10  --in stl  --val tinyimagenet  --dv 6 --bs 10 > results/mahalanobis_stlval_tinyimagenet.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/stl/rot_STL.pth --nc 10  --in stl  --val tinyimagenet  --dv 6 --bs 1 > results/rotation_stlval_tinyimagenet.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/cifar10/genOdinCifar100.pth --nc 10 --in stl --val tinyimagenet  --dv 6 > results/genodin_stlval_tinyimagenet.txt