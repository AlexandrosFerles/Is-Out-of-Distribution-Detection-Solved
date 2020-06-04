python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val cifar100 --dv 1  --nc 10 > results/triplets/baseline_low_dropout_extended_cifar100.txt
python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val cifar100 --dv 1  --nc 10 > results/triplets/odin_low_dropout_extended_cifar100.txt
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val cifar100 --bs 10 --dv 1  --nc 10 > results/triplets/mahalanobis_low_dropout_extended_cifar100.sh

python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val svhn --dv 1  --nc 10 > results/triplets/baseline_low_dropout_extended_svhn.txt
python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val svhn --dv 1  --nc 10 > results/triplets/odin_low_dropout_extended_svhn.txt
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val svhn --bs 10 --dv 1  --nc 10 > results/triplets/mahalanobis_low_dropout_extended_svhn.sh

python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val stl --dv 1  --nc 10 > results/triplets/baseline_low_dropout_extended_stl.txt
python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val stl --dv 1  --nc 10 > results/triplets/odin_low_dropout_extended_stl.txt
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val stl --bs 10 --dv 1  --nc 10 > results/triplets/mahalanobis_low_dropout_extended_stl.sh

python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val tinyimagenet --dv 1  --nc 10 > results/triplets/baseline_low_dropout_extended_tinyimagenet.txt
python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val tinyimagenet --dv 1  --nc 10 > results/triplets/odin_low_dropout_extended_tinyimagenet.txt
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_extended_eb0Cifar10.pth --in cifar10 --val tinyimagenet --bs 10 --dv 1  --nc 10 > results/triplets/mahalanobis_low_dropout_extended_tinyimagenet.sh