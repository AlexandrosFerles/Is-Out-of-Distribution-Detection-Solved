# Is Out-of-Distribution detection solved?

Code to accompany the "Is Out-of-Distribution detection solved?" paper. 

## Datasets and pickle files

We provide several pickle files which are expected from our code to generate the train/val splits for each unique experiments of our work.

We provide a few of the datasets used through this work in this [Google Drive folder](https://drive.google.com/drive/folders/1vX7cD33jI\_vsIgBw-05Mshju5HyK51Rm?usp=sharing). In the same folder, we provide specific instructions on how to generate the datasets that are too big to be uploaded there. 

## Creating custom version of Efficient Net

We use [this version](https://github.com/lukemelas/EfficientNet-PyTorch) of [Efficient Nets](https://arxiv.org/abs/1905.11946) for PyTorch. In order to run customised versions of Efficient Nets that for [self-supervised OOD detection](https://arxiv.org/pdf/1906.12340.pdf) and [Generalized-ODIN](https://arxiv.org/abs/2002.11297), a few files need to be added and modified. We share these files under the path `src/custom_ebnet_files`.