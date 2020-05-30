python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200 --in tinyimagenet  --val cifar10  --dv 6 > results/baseline_tinyimagenet_val_cifar10.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200 --in tinyimagenet  --val cifar10  --dv 6 > results/odin_tinyimagenet_val_cifar10.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200  --in tinyimagenet  --val cifar10  --dv 6 --bs 10 > results/mahalanobis_tinyimagenet_val_cifar10.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/tinyimagenet/rot_eb0tinyimagenet.pth --nc 200  --in tinyimagenet  --val cifar10  --dv 6 --bs 1 > results/rotation_tinyimagenet_val_cifar10.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stl/genOdinSTL.pth --nc 200 --in tinyimagenet  --val cifar10  --dv 6 > results/genodin_tinyimagenet_val_cifar10.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200 --in tinyimagenet  --val svhn  --dv 6 > results/baseline_tinyimagenet_val_svhn.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200 --in tinyimagenet  --val svhn  --dv 6 > results/odin_tinyimagenet_val_svhn.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200  --in tinyimagenet  --val svhn  --dv 6 --bs 10 > results/mahalanobis_tinyimagenet_val_svhn.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/tinyimagenet/rot_eb0tinyimagenet.pth --nc 200  --in tinyimagenet  --val svhn  --dv 6 --bs 1 > results/rotation_tinyimagenet_val_svhn.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/genOdinTinyImageNet.pth --nc 200 --in tinyimagenet --val svhn  --dv 6 > results/genodin_tinyimagenet_val_svhn.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200 --in tinyimagenet  --val cifar100  --dv 6 > results/baseline_tinyimagenet_val_cifar100.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200 --in tinyimagenet  --val cifar100  --dv 6 > results/odin_tinyimagenet_val_cifar100.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200  --in tinyimagenet  --val cifar100  --dv 6 --bs 10 > results/mahalanobis_tinyimagenet_val_cifar100.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/tinyimagenet/rot_eb0tinyimagenet.pth --nc 200  --in tinyimagenet  --val cifar100  --dv 6 --bs 1 > results/rotation_tinyimagenet_val_cifar100.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/genOdinTinyImageNet.pth --nc 200 --in tinyimagenet --val cifar100  --dv 6 > results/genodin_tinyimagenet_val_cifar100.txt

python ood_triplets.py --m baseline --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200 --in tinyimagenet  --in cifar100  --dv 6 > results/baseline_tinyimagenet_val_stl.txt
python ood_triplets.py --m odin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200 --in tinyimagenet  --in cifar100  --dv 6 > results/odin_tinyimagenet_val_stl.txt
python ood_triplets.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/tinyimagenet/eb0tinyimagenet.pth --nc 200  --in tinyimagenet  --in cifar100  --dv 6 --bs 10 > results/mahalanobis_tinyimagenet_val_stl.txt
python ood_triplets.py --m rotation --mc /raid/ferles/checkpoints/eb0/tinyimagenet/rot_eb0tinyimagenet.pth --nc 200  --in tinyimagenet  --in cifar100  --dv 6 --bs 1 > results/rotation_tinyimagenet_val_stl.txt
python ood_triplets.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/tinyimagenet/genOdinTinyImageNet.pth --nc 200 --in tinyimagenet --in cifar100  --dv 6 > results/genodin_tinyimagenet_val_stl.txt