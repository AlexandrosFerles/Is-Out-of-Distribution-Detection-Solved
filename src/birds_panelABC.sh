#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out places --nc 555 --dv 3  > results/birds/panelA/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out places --nc 555 --dv 3  > results/birds/panelA/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out places --nc 555 --dv 3  > results/birds/panelA/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out places --nc 555 --dv 3  --bs 20  > results/birds/panelA/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds.pth --in nabirds --val imagenet --out places --nc 555 --dv 3  --bs 1 > results/birds/panelA/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds.pth --in nabirds --val imagenet --out places --nc 555 --dv 3  > results/birds/panelA/generalizedodin.txt
#
#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out cub200-in --nc 555 --dv 3  > results/birds/panelB/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out cub200-in --nc 555 --dv 3  > results/birds/panelB/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out cub200-in --nc 555 --dv 3  > results/birds/panelB/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out cub200-in --nc 555 --dv 3  --bs 20  > results/birds/panelB/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds.pth --in nabirds --val imagenet --out cub200-in --nc 555 --dv 3  --bs 1 > results/birds/panelB/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds.pth --in nabirds --val imagenet --out cub200-in --nc 555 --dv 3  > results/birds/panelB/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out cub200-out --nc 555 --dv 3  > results/birds/panelC/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out cub200-out --nc 555 --dv 3  > results/birds/panelC/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out cub200-out --nc 555 --dv 3  > results/birds/panelC/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds.pth --in nabirds --val imagenet --out cub200-out --nc 555 --dv 3  --bs 20  > results/birds/panelC/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds.pth --in nabirds --val imagenet --out cub200-out --nc 555 --dv 3  --bs 1 > results/birds/panelC/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds.pth --in nabirds --val imagenet --out cub200-out --nc 555 --dv 3  > results/birds/panelC/generalizedodin.txt
