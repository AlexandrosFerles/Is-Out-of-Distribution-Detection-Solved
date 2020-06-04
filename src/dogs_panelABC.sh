python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 > results/dogs/panelA/baseline.txt
python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 > results/dogs/panelA/odin.txt
python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 --bs 20  > results/dogs/panelA/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 --bs 1 > results/dogs/panelA/rotation.txt
python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 > results/dogs/panelA/generalizedodin.txt

python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 > results/dogs/panelB/baseline.txt
python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 > results/dogs/panelB/odin.txt
python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 --bs 20  > results/dogs/panelB/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 --bs 1 > results/dogs/panelB/rotation.txt
python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 > results/dogs/panelB/generalizedodin.txt

python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-out --nc 120 > results/dogs/panelC/baseline.txt
python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --out stanforddogs --val imagenet --out oxfordpets-out --nc 120 > results/dogs/panelC/odin.txt
python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --out stanforddogs --val imagenet --out oxfordpets-out --nc 120 --bs 20  > results/dogs/panelC/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs.pth --out stanforddogs --val imagenet --out oxfordpets-out --nc 120 --bs 1 > results/dogs/panelC/rotation.txt
python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs.pth --out stanforddogs --val imagenet --out oxfordpets-out --nc 120 > results/dogs/panelC/generalizedodin.txt
