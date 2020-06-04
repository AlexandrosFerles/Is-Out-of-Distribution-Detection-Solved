python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 > dogs/panelA/baseline.txt
python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 > dogs/panelA/odin.txt
python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 --bs 20  > dogs/panelA/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 --bs 1 > dogs/panelA/rotation.txt
python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs.pth --in stanforddogs --val imagenet --out places --nc 120 > dogs/panelA/generalizedodin.txt

python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 > dogs/panelB/baseline.txt
python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 > dogs/panelB/odin.txt
python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 --bs 20  > dogs/panelB/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 --bs 1 > dogs/panelB/rotation.txt
python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-in --nc 120 > dogs/panelB/generalizedodin.txt

python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --in stanforddogs --val imagenet --out oxfordpets-out --nc 120 > dogs/panelC/baseline.txt
python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --out stanforddogs --val imagenet --out oxfordpets-out --nc 120 > dogs/panelC/odin.txt
python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs.pth --out stanforddogs --val imagenet --out oxfordpets-out --nc 120 --bs 20  > dogs/panelC/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs.pth --out stanforddogs --val imagenet --out oxfordpets-out --nc 120 --bs 1 > dogs/panelC/rotation.txt
python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs.pth --out stanforddogs --val imagenet --out oxfordpets-out --nc 120 > dogs/panelC/generalizedodin.txt
