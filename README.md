# TinyMachines.jl

[![Build Status](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml?query=branch%3Amain)

A collection of tiny mchine learning models for IoT devices, written in Julia/Flux.

### UNet5, UNet4, UNet2

UNet5 is the classic U-Net architecture, with five encoder/decoder levels. UNet4 and UNet2 have, respectively, four and two. Number of channels can be modulated to increase/decrease size.

The standard implementation follows the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" (arXiv:1505.04597). Paper credits: Ronnenberger, Olaf; Fischer, Philipp; Brox, Thomas.

### Mobile-Unet

Mobile-Unet utilizes inverted residual bottleneck blocks and pointwise convolution.

Implementation follows the paper "Mobile-Unet: An efficient convolutional neural network for fabric defect detection" (arXiv:xxx). Paper credits: Jing, Junfeng; Wang, Zhen; Ratsch, Matthias; Zhang, Huanhuan.

### ESPNet

Under development.

Implementation follows the paper "ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation" (arXiv:1803.06815v3). Paper credits: Mehta, Sachin; Rastegari, Mohammad; Caspi, Anat;
Shapiro, Linda; and Hajishirzi, Hannaneh.

### Credits
Credits for the original architectures go to the papers' authors, as aforementioned.

Credits for this implementation in Julia/Flux go to Ciro B Rosa (cirobr@github).
* LinkedIn: https://www.linkedin.com/in/cirobrosa/
