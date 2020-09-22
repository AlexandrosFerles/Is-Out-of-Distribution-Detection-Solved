# Is Out-of-Distribution detection solved?

Code to accompany the "Is Out-of-Distribution detection solved?" paper. 

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

Subset training is performed by applying the appropriate config file. For instance, 
