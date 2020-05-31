python ood.py --m baseline --bs 30  --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --nc 555 --dv 6 --in nabirds --val imagenet --out cub200 > baseline_nabirds_imagenet_val.txt
python ood.py --m odin --bs 30  --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --nc 555 --dv 6 --in nabirds --val imagenet --out cub200 > odin_nabirds_imagenet_val.txt
python ood.py --m mahalanobis --bs 10  --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --nc 555 --dv 6 --in nabirds --val imagenet --out cub200 > mahalanobis_nabirds_imagenet_val.txt
python ood.py --m odin --bs 30  --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --nc 555 --dv 6 --in nabirds --val fgsm --out cub200 > odin_nabirds_fgsm_val.txt
python ood.py --m mahalanobis --bs 10  --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --nc 555 --dv 6 --in nabirds --val fgsm --out cub200 > mahalanobis_nabirds_fgsm_val.txt
python ood.py --m generalizedodin --bs 30 --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds.pth --nc 555 --dv 0 --in nabirds --val imagenet --out cub200 > gen_odin_nabirds_imagenet_val.txt
python ood.py --m rotation --bs 1  --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds.pth --nc 555 --dv 6 --in nabirds --val imagenet --out cub200 > rotation_nabirds_imagenet_val.txt
