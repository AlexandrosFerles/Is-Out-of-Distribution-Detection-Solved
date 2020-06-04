#python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val cifar100 --dv 0 --nc 10 > results/triplets/baseline_low_dropout_cifar100.sh
#python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val cifar100 --dv 0 --nc 10 > results/triplets/odin_low_dropout_cifar100.sh
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val cifar100 --bs 20 --dv 0 --nc 10 > results/triplets/mahalanobis_low_dropout_cifar100.sh

#python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val svhn --dv 0 --nc 10 > results/triplets/baseline_low_dropout_svhn.sh
#python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val svhn --dv 0 --nc 10 > results/triplets/odin_low_dropout_svhn.sh
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val svhn --bs 20 --dv 0 --nc 10 > results/triplets/mahalanobis_low_dropout_svhn.sh

#python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val stl --dv 0 --nc 10 > results/triplets/baseline_low_dropout_stl.sh
#python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val stl --dv 0 --nc 10 > results/triplets/odin_low_dropout_stl.sh
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val stl --bs 20 --dv 0 --nc 10 > results/triplets/mahalanobis_low_dropout_stl.sh

#python ood_triplets.py --m baseline --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val tinyimagenet --dv 0 --nc 10 > results/triplets/baseline_low_dropout_tinyimagenet.sh
#python ood_triplets.py --m odin --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val tinyimagenet --dv 0 --nc 10 > results/triplets/odin_low_dropout_tinyimagenet.sh
python ood_triplets.py --m mahalanobis --mc /home/ferles/checkpoints/eb0/cifar10/low_dropout_eb0Cifar10.pth --in cifar10 --val tinyimagenet --bs 20 --dv 0 --nc 10 > results/triplets/mahalanobis_low_dropout_tinyimagenet.sh