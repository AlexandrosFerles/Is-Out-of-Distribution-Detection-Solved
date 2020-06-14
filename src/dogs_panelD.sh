#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_1.pth --in stanforddogs --val imagenet --out stanforddogs --sub 1  --nc 96 --dv 6  > results/dogs/panelD/1/baseline.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_1.pth --in stanforddogs --val imagenet --out stanforddogs --sub 1  --nc 96 --dv 6  > results/dogs/panelD/1/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_1.pth --in stanforddogs --val imagenet --out stanforddogs --sub 1  --nc 96 --dv 6  --bs 20  > results/dogs/panelD/1/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs_subset_1.pth --in stanforddogs --val imagenet --out stanforddogs --sub 1  --nc 96 --dv 6  --bs 1 > results/dogs/panelD/1/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs_subset_1.pth --in stanforddogs --val imagenet --out stanforddogs --sub 1  --nc 96 --dv 6  > results/dogs/panelD/1/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_2.pth --in stanforddogs --val imagenet --out stanforddogs --sub 2  --nc 96 --dv 6  > results/dogs/panelD/2/baseline.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_2.pth --in stanforddogs --val imagenet --out stanforddogs --sub 2  --nc 96 --dv 6  > results/dogs/panelD/2/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_2.pth --in stanforddogs --val imagenet --out stanforddogs --sub 2  --nc 96 --dv 6  --bs 20  > results/dogs/panelD/2/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs_subset_2.pth --in stanforddogs --val imagenet --out stanforddogs --sub 2  --nc 96 --dv 6  --bs 1 > results/dogs/panelD/2/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs_subset_2.pth --in stanforddogs --val imagenet --out stanforddogs --sub 2  --nc 96 --dv 6  > results/dogs/panelD/2/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_3.pth --in stanforddogs --val imagenet --out stanforddogs --sub 3  --nc 96 --dv 6  > results/dogs/panelD/3/baseline.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_3.pth --in stanforddogs --val imagenet --out stanforddogs --sub 3  --nc 96 --dv 6  > results/dogs/panelD/3/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_3.pth --in stanforddogs --val imagenet --out stanforddogs --sub 3  --nc 96 --dv 6  --bs 20  > results/dogs/panelD/3/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs_subset_3.pth --in stanforddogs --val imagenet --out stanforddogs --sub 3  --nc 96 --dv 6  --bs 1 > results/dogs/panelD/3/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs_subset_3.pth --in stanforddogs --val imagenet --out stanforddogs --sub 3  --nc 96 --dv 6  > results/dogs/panelD/3/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_4.pth --in stanforddogs --val imagenet --out stanforddogs --sub 4 --nc 96 --dv 6  > results/dogs/panelD/4/baseline.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_4.pth --in stanforddogs --val imagenet --out stanforddogs --sub 4  --nc 96 --dv 6  > results/dogs/panelD/4/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_4.pth --in stanforddogs --val imagenet --out stanforddogs --sub 4  --nc 96 --dv 6  --bs 20  > results/dogs/panelD/4/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs_subset_4.pth --in stanforddogs --val imagenet --out stanforddogs --sub 4  --nc 96 --dv 6  --bs 1 > results/dogs/panelD/4/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs_subset_4.pth --in stanforddogs --val imagenet --out stanforddogs --sub 4  --nc 96 --dv 6  > results/dogs/panelD/4/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_5.pth --in stanforddogs --val imagenet --out stanforddogs --sub 5  --nc 96 --dv 6  > results/dogs/panelD/5/baseline.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_5.pth --in stanforddogs --val imagenet --out stanforddogs --sub 5   --nc 96 --dv 6  > results/dogs/panelD/5/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/stanforddogs/StanfordDogs_subset_5.pth --in stanforddogs --val imagenet --out stanforddogs --sub 5   --nc 96 --dv 6  --bs 20  > results/dogs/panelD/5/mahalanobis.txt
python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/stanforddogs/rot_StanfordDogs_subset_5.pth --in stanforddogs --val imagenet --out stanforddogs --sub 5   --nc 96 --dv 6  --bs 1 > results/dogs/panelD/5/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/stanforddogs/GenOdinStanfordDogs_subset_5.pth --in stanforddogs --val imagenet --out stanforddogs --sub 5   --nc 96 --dv 6  > results/dogs/panelD/5/generalizedodin.txt