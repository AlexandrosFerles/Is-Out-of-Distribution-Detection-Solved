python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val cifar100 --dv 3  --nc 10 > results/triplets/baseline_cifar100.txt
python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val cifar100 --dv 3  --nc 10 > results/triplets/odin_cifar100.txt
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val cifar100 --bs 10 --dv 3  --nc 10 > results/triplets/mahalanobis_cifar100.sh

python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val svhn --dv 3  --nc 10 > results/triplets/baseline_svhn.txt
python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val svhn --dv 3  --nc 10 > results/triplets/odin_svhn.txt
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val svhn --bs 10 --dv 3  --nc 10 > results/triplets/mahalanobis_svhn.sh

python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val stl --dv 3  --nc 10 > results/triplets/baseline_stl.txt
python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val stl --dv 3  --nc 10 > results/triplets/odin_stl.txt
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val stl --bs 10 --dv 3  --nc 10 > results/triplets/mahalanobis_stl.sh

python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val tinyimagenet --dv 3  --nc 10 > results/triplets/baseline_tinyimagenet.txt
python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val tinyimagenet --dv 3  --nc 10 > results/triplets/odin_tinyimagenet.txt
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/eb0Cifar10.pth --in cifar10 --val tinyimagenet --bs 10 --dv 3  --nc 10 > results/triplets/mahalanobis_tinyimagenet.sh