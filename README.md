# Pixel-wise Guidance for Utilizing Auxiliary Features for Monte Carlo Denoising (I3D 2023)
### [Project Page] | [Paper]
<br>Official PyTorch implementation.</br>
 [Kyu-Beom Han](https://qbhan.oopy.io/)<sup>1</sup>,
 Olivia G. Odenthal<sup>2</sup>,
 Woo-Jae Kim<sup>1</sup>,
 [Sung-eui Yoon](https://sgvr.kaist.ac.kr/~sungeui/)<sup>1</sup> <br>
KAIST, Repulic of Korea<sup>1</sup>  
University of Stuttgart, Germany<sup>2</sup>  

![teaser_image](assets/results.PNG)

<!-- ### Scene credits
- ["Library-Home Office"](https://www.blendswap.com/blend/19984) by [ThePefDispenser](https://blendswap.com/profile/663773) under CC BY 3.0.
- ["My Kitchen"](https://www.blendswap.com/blend/9528) by [tokabilitor](https://www.blendswap.com/profile/41638) under CC0. -->

## Paper Summary

Our pixel-wise guidance framework enhances the per-pixel contribution of each type of auxiliary features (G-buffers and P-buffers[Cho et al. 2021]), achieving considerable test performance increase compared to baseline denoising models. 

Our code is tested on Ubuntu 18.04 with an Intel i9-10940X CPU with NVIDIA Geforce RTX 3090, RTX 2080ti, and Quadro RTX 8000 graphics cards.

## Quick Start

1. Clone the git repository and create a PyTorch Anaconda environment as below:
    ```
    conda create -n pytorch python=3.9
    activate pytorch
    conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolit==11.3.1 -c pytorch
    conda install -c anaconda h5py
    conda install tensorboard tensorboardX lz4
    pip install opencv-python, kornia, scipy, pandas, tqdm, matplotlib
    ```

2. Go to torch_utils and follow the instructions to install per-pixel kernel function for PyTorch

3. Run the shell script in the directory 'scripts/' to train the model.


## Overview
1. Supporting State-of-the-art baselines: 
    - Image-space MC Denoising 
      - Kernel-predicting convolutional network (KPCN) [Bako et al. 2017]
      - Adversarial Monte Carlo Denoising (AdvMCD) [Xu et al. 2019]
    - Sample-space MC Denoising
        - Neural Denoising with Layer Embeddings (NDLE) [Munkberg and Hasselgren 2020]

2. Repository Overview
   - train_kpcn.py         # Train KPCN with G- \& P- buffers
   - train_adv.py          # Train AdvMCD with G- \& P- buffers
   - train_lbmc.py         # Train NDLE with G- \& P- buffers
   - train_ensemble.py     # Train Our Guidance Denoising Framework
   - test_models.py        # Test any model
     - support/            # Codes for baselines \& utilities
       - WCMC/             # Codes for reproducing KPCN & WCMC  
       - AdvMCD/           # Codes for reproducing AdvMCD
       - NDLE/             # Codes for reproducing NDLE
     	- datasets.py      
     	- img_utils.py
     	- interfaces.py
     	- losses.py
     	- metrics.py
     	- networks.py
     	- utils.py
   	- scripts/             # Scripts to train models
   	- torch_utils          # Cuda implementation of per-pixel kernel for PyTorch [Munkberg and Hasselgren 2020]
