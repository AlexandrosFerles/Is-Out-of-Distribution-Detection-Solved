# Is Out-of-Distribution detection solved?

Code to accompany the "Is Out-of-Distribution detection solved?" paper. We asses 6 Out-of-Distribution (OoD) detection methods ([Baseline](https://arxiv.org/abs/1610.02136), [ODIN](https://arxiv.org/abs/1706.02690), [Mahalanobis](https://arxiv.org/abs/1807.03888), [Self-Supervision](https://arxiv.org/pdf/1906.12340.pdf), [Generalized-ODIN](https://arxiv.org/abs/2002.11297), [Self-Ensemble](https://arxiv.org/abs/1809.03576)) .

Updates to follow:
* Code to run individual OoD detection experiments on standard datasets

## Datasets and pickle files

We provide several pickle files which are expected from our code to generate the train/val splits for each unique experiments of our work.

We provide a few of the datasets used through this work in this [Google Drive folder](https://drive.google.com/drive/folders/1vX7cD33jI\_vsIgBw-05Mshju5HyK51Rm?usp=sharing). In the same folder, we provide specific instructions on how to generate the datasets that are too big to be uploaded there. 

## Creating custom versions of Efficient Net

We use [this version](https://github.com/lukemelas/EfficientNet-PyTorch) of [Efficient Nets](https://arxiv.org/abs/1905.11946) for PyTorch. In order to run customised versions of Efficient Nets that for [self-supervised OOD detection](https://arxiv.org/pdf/1906.12340.pdf) and [Generalized-ODIN](https://arxiv.org/abs/2002.11297), a few files need to be added and modified. We share these files under the path `src/custom_ebnet_files`.

## Training base classifiers

In order to train {EB0, Rot-EB0, Ensemble-EB0} as a base classifier for standard datasets:

```python {trainNaturalImages.py, rotationNaturalImages.py, ensembleNaturalImages.py} --c $CONFIG_FILE_NAME --dv $DEVICE_INDEX --ds $DATASET_NAME```

where you can set a GPU device index if not using device #0, choose one of the config files under the path `configs/standard` and select one of the standard datasets from {cifar10, cifar100, svhn, stl, tinyimagenet, tinyimagenet-cifar10, tinyimagenet-cifar100}. Gen-EB0 is provided as an option by using the appropriate config file on `trainNaturalImages.py`.

Similarly, for fine grained datasets (dog breeds from Stanford Dogs and bird species from NaBirds) you need to run the following script:

```python {trainFineGrained.py, rotationFineGrained.py, trainFineGrainedEnsemble.py} --c $CONFIG_FILE_NAME --dv $DEVICE_INDEX --ds $DATASET_NAME```

If you need to train a classifier on one of the designated subsets of each dataset, simly add the parameter `--sub $SUBSET_INDEX` in the above command. 

Finally, the appropriate commands for ISIC (dermatology data) are the following:

```python {custom_train.py, ss_rot.py, EnsembleISIC.py} --c $CONFIG_FILE_NAME --dv $DEVICE_INDEX```

Subset training is performed by applying the appropriate config file. For instance, `configs/ISIC/AK_exclude.json` trains a base model on all classes except Actinic Keratosis (AK; see the [ISIC 2019 competition page](https://challenge2019.isic-archive.com) for details).

## OoD detection methods on standard datasets

Standard datasets include {cifar10, cifar100, stl, svhn} and one option out of {tinyimagenet, tinyimagenet-cifar10, tinyimagenet-cifar100}.

By executing `ood_ensemble.py` you can apply all OoD detection methods based on a train (InD) and val (OoD) dataset. All methods are optimized on this pair, and scored in between the InD dataset and the remaining 3 datasets which are considered as OoD.

```python ood_ensemble.py --in $InD --val $OoD --dv $DEVICE_INDEX --nc $NUMBER_OF_IND_CLASSES --mcf txt_files/$CHECKPOINTS_FILE```  

where $CHECKPOINTS_FILE refers to a txt file that provides the paths for all the checkpoints trained on the InD dataset and are required for the OoD detection methods.

If you add the parameter `--m $METHOD_NAME` where method name can be any of the options `baseline, odin, mahalanobis, self-supervision, selfsupervision, self_supervision, generalized-odin, generalizedodin, generalized_odin, ensemble`, you can apply an individual OoD detection method on standard datasets. 

If you wish to use either tinyimagenet-cifar10 or tinyimagenet-cifar100, you need to add the parameter `--test {tinyimagenet-cifar10, tinyimagenet-cifar100}`

## OoD detection methods on fine-grained datasets

You can apply one of the included OoD detection method through the following command:

```python ood.py --m $OOD_METHOD --in $InD --val $Val-OoD --out $TEST-OoD --dv $DEVICE_INDEX --mc $CHECKPOINT_FILE --nc $NUMBER_OF_IND_CLASSES```

For our main results, use "imagenet" as the val dataset. If running the self-ensemble method, replace the `--mc` parameter with `--mcf` pointing to the txt file that includes all the ensemble checkpoints. Choose between a subset of a fine-grained dataset and an excluded class from ISIC by including either `--sub $SUBSET_INDEX` or `--ex $EXCLUDED_CLASS`. 

## Acknowledgements

Code from [ODIN](https://github.com/facebookresearch/odin) and [Mahalanobis](https://github.com/pokaxpoka/deep_Mahalanobis_detector/) was used in this work, and we sincerely thank the authors for making their work and code publicly available. We also found the contributions of [Self-Supervision](https://github.com/hendrycks/ss-ood) and [Self-Ensemble](https://github.com/YU1ut/Ensemble-of-Leave-out-Classifiers) pretty helpful to understand their core components.

Most importantly, we deeply thank the authors of [Generalized-ODIN](https://arxiv.org/abs/2002.11297) for answering immediately all the questions that we came up with (and sharing code chunks) when trying to reproduce their work.