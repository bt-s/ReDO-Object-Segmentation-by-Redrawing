#  NeurIPS Reproducibility Challenge | Reproduction of _Unsupervised Object Segmentation by Redrawing_ by Chen et al. 2019

An endeavour for the *Deep Learning, Advanced: DT2119* course 2019 at KTH Royal Institute of Technology.

## Authors: Adrian Chmielewski-Anders, Mats Steinweg and Bas Straathof

The objective of this project is to reproduce the results a presented in: 

> Chen, Mickaël, Thierry Artières, and Ludovic Denoyer. "Unsupervised Object Segmentation by Redrawing." arXiv preprint arXiv:1905.13539 (2019).

Currently, only the results for the _Flowers_ data set have been reproduced.

Once finished, the corresponding report will be uploaded here.


## Project file structure

`evaluate_masks.py`\
Script to test the mask generator.

`evaluate_models.py`\
Script to test the models.

`gen_images.py`\
Script to load pre-trained segmentation and generator networks, and create figures as in the original paper.

`gen_datasets.py`\
Script to obtain the _Flowers_, _Labeled Faces in the Wild_ and _CUB-200-2011_ data sets.

`test_all.py`\
Script to test the datasets preprocessing, discriminator, generator, instance normalization and segmentation network of the ReDO application.``` 

`train_gan.py`\
Script to train the ReDO GAN.

`redo`

- `src`

  * `datasets.py`
  A script for handling the supported datasets _Flowers_, _Faces in the Wild_ and _CUB-200-2011_
  * `discriminator.py`
  A script containing an implementation of the discriminator network.
  * `draw.py`
  A script for redrawing images.
  * `generator.py`
  A script containing an implementation of the generator network.
  * `information_network.py`
  A script containing an implementation of the information network.
  * `network_components.py`
  A script containing an implementations of several network components: spectral normalization, instance normalization, the self-attention module and the residual block. 
  * `segmentation_network.py`
  A script containing an implementation of the segmentation network.
  * `train_utils.py`
  A script containing several util functions for training the ReDO GAN.


- `tests`

  * `test_datasets.py`
  A script to test the data set preprocessing.
  * `test_discriminator.py`
  A script to test the discrimnator network.
  * `test_generator.py`
  A script to test the generator network.
  * `test_instance_norm.py`
  A script to test the instance normalization.
  * `test_segmentation_network.py`
  A script to test the segmentation network.
