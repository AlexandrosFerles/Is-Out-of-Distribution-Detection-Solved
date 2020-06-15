#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_1.pth --in nabirds --val imagenet --out nabirds --sub 1  --nc 460 --dv 4 > results/birds/panelD/1/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_1.pth --in nabirds --val imagenet --out nabirds --sub 1  --nc 460 --dv 4 > results/birds/panelD/1/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_1.pth --in nabirds --val imagenet --out nabirds --sub 1  --nc 460 --dv 4   > results/birds/panelD/1/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_1.pth --in nabirds --val imagenet --out nabirds --sub 1  --nc 460 --dv 4   --bs 20  > results/birds/panelD/1/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds_subset_1.pth --in nabirds --val imagenet --out nabirds --sub 1  --nc 460 --dv 4   --bs 1 > results/birds/panelD/1/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds_subset_1.pth --in nabirds --val imagenet --out nabirds --sub 1  --nc 460 --dv 4   > results/birds/panelD/1/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_2.pth --in nabirds --val imagenet --out nabirds --sub 2  --nc 460 --dv 4   > results/birds/panelD/2/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_2.pth --in nabirds --val imagenet --out nabirds --sub 2  --nc 460 --dv 4   > results/birds/panelD/2/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_2.pth --in nabirds --val imagenet --out nabirds --sub 2  --nc 460 --dv 4   > results/birds/panelD/2/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_2.pth --in nabirds --val imagenet --out nabirds --sub 2  --nc 460 --dv 4   --bs 20  > results/birds/panelD/2/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds_subset_2.pth --in nabirds --val imagenet --out nabirds --sub 2  --nc 460 --dv 4   --bs 1 > results/birds/panelD/2/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds_subset_2.pth --in nabirds --val imagenet --out nabirds --sub 2  --nc 460 --dv 4   > results/birds/panelD/2/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_3.pth --in nabirds --val imagenet --out nabirds --sub 3  --nc 434 --dv 4 > results/birds/panelD/3/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_3.pth --in nabirds --val imagenet --out nabirds --sub 3  --nc 434 --dv 4 > results/birds/panelD/3/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_3.pth --in nabirds --val imagenet --out nabirds --sub 3  --nc 434 --dv 4   > results/birds/panelD/3/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_3.pth --in nabirds --val imagenet --out nabirds --sub 3  --nc 434 --dv 4   --bs 20  > results/birds/panelD/3/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds_subset_3.pth --in nabirds --val imagenet --out nabirds --sub 3  --nc 434 --dv 4   --bs 1 > results/birds/panelD/3/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds_subset_3.pth --in nabirds --val imagenet --out nabirds --sub 3  --nc 434 --dv 4   > results/birds/panelD/3/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_4.pth --in nabirds --val imagenet --out nabirds --sub 4 --nc 434 --dv 4   > results/birds/panelD/4/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_4.pth --in nabirds --val imagenet --out nabirds --sub 4 --nc 434 --dv 4   > results/birds/panelD/4/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_4.pth --in nabirds --val imagenet --out nabirds --sub 4  --nc 434 --dv 4   > results/birds/panelD/4/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_4.pth --in nabirds --val imagenet --out nabirds --sub 4  --nc 434 --dv 4   --bs 20  > results/birds/panelD/4/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds_subset_4.pth --in nabirds --val imagenet --out nabirds --sub 4  --nc 434 --dv 4   --bs 1 > results/birds/panelD/4/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds_subset_4.pth --in nabirds --val imagenet --out nabirds --sub 4  --nc 434 --dv 4   > results/birds/panelD/4/generalizedodin.txt

#python ood.py --m baseline --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_5.pth --in nabirds --val imagenet --out nabirds --sub 5  --nc 433 --dv 4 > results/birds/panelD/5/baseline.txt
python ood.py --m baseline --mcdo 20 --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_5.pth --in nabirds --val imagenet --out nabirds --sub 5  --nc 433 --dv 4 > results/birds/panelD/5/baseline_mcdo.txt
#python ood.py --m odin --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_5.pth --in nabirds --val imagenet --out nabirds --sub 5   --nc 433 --dv 4   > results/birds/panelD/5/odin.txt
#python ood.py --m mahalanobis --mc /raid/ferles/checkpoints/eb0/nabirds/Nabirds_subset_5.pth --in nabirds --val imagenet --out nabirds --sub 5   --nc 433 --dv 4   --bs 20  > results/birds/panelD/5/mahalanobis.txt
#python ood.py --m rotation --mc /raid/ferles/checkpoints/eb0/nabirds/rot_Nabirds_subset_5.pth --in nabirds --val imagenet --out nabirds --sub 5   --nc 433 --dv 4   --bs 1 > results/birds/panelD/5/rotation.txt
#python ood.py --m generalizedodin --mc /raid/ferles/checkpoints/eb0/nabirds/GenOdinNabirds_subset_5.pth --in nabirds --val imagenet --out nabirds --sub 5 --nc 433 --dv 4 > results/birds/panelD/5/generalizedodin.txt