# ODT_projectionGAN

Official implementation of the paper ***"Complete Removal of Missing Cone Artifacts in ODT using Unsupervised Deep Learning"***.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4434355.svg)](https://doi.org/10.5281/zenodo.4434355)

### Downloading source data

If you only wish to visualize the data,
all reconstruction data provided in the manuscript can be downloaded from the following url [link](https://www.dropbox.com/sh/ci6rw4l2xa1atc5/AACb-Y0VPkr2KqJZxJrdp_Aea?dl=0)

### Environment

The developmental version of the code was created and tested in the following environment.
```
- Linux 18.04
- CUDA 10.0
```

### Prerequisites

All python code were developed in conda ```python 3.7``` environment, and the requirements are listed in ```requirements.txt```.
MATLAB codes were tested with ```MATLAB R2019a```. Be sure to pre-compile CUDA files before running the main scripts.

### Inference

Before performing projectionGAN reconstruction, you first have to download ODT data reconstructed with GP algorithm [download link](https://www.dropbox.com/sh/yiitrugxdo6101c/AACcNavEc2Q_KUJGEAinwE1oa?dl=0). The downloaded data should be placed in the folder ```./GP_recon```. Reconstruction steps are performed in the following order

1. ```./data_processing/{biological cell, microbead, phantom}/prep_{bio, microbead, phantom}.m```
2. ```python Infer_{bio, microbead, phantom}.py```
3. ```./data_processing/{biological cell, microbead, phantom}/FBP_{bio, microbead, phantom}.m```

Optionally, before FBP, you can visualize the enhanced projections with ```visualize_{bio, microbead, phantom}.m```

### Training

If you are interested in training with your own data, run

```
# Uses vanilla patchGAN discriminator
python train.py

# Uses multi-scale patchGAN discriminator
python train_multiD.py
```

Be sure to specify the ```settings``` inside the scripts.





