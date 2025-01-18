![alt text](./images/logo-name-tm.png)

# TinyMachines.jl

[![Build Status](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml?query=branch%3Amain)

A collection of tiny machine learning models for image semantic segmentation in IoT devices, written in Flux.jl

Besides regular mask outputs, all models deliver their internal feature maps as outputs, which are useful for model compression through knowledge distillation.


## UNet5, UNet4

UNet5 is the classic U-Net architecture, with five encoder/decoder levels. UNet4 has four. Number of channels can be modulated to increase/decrease size.

The standard implementation follows the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" ([arXiv](https://arxiv.org/abs/1505.04597)). Paper credits: Ronnenberger, Olaf; Fischer, Philipp; and Brox, Thomas.


## MobileUNet

Mobile-Unet has the same encoder structure as the Mobilenet-V2 classification model, and the same u-shape and skip connection principles as the U-Net.

Implementation follows the following papers:
* "Mobile-Unet: An efficient convolutional neural network for fabric defect detection" ([doi.org](https://doi.org/10.1177/0040517520928604)). Paper credits: Jing, Junfeng; Wang, Zhen; Ratsch, Matthias; and Zhang, Huanhuan.
* MobileNetV2: Inverted Residuals and Linear Bottleneck" ([arxiv]https://doi.org/10.48550/arXiv.1801.04381). Paper credits: Sandler, Mark; Howard, Andrew; Zhu, Menglong; Zhmoginov, Andrey; and Chen, Liang-Chen.


## ESPNet
ESPNet utilizes the Efficient Spatial Pyramid module and the PReLU nonlinearity (replaced by ReLU in this implementation).

Implementation follows the paper "ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation" ([arXiv](https://arxiv.org/abs/1803.06815)). Paper credits: Mehta, Sachin; Rastegari, Mohammad; Caspi, Anat; Shapiro, Linda; and Hajishirzi, Hannaneh.


## Credits
Credits for the original architectures go to the papers' authors, as aforementioned.

Credits for the implementations in Julia/Flux go to Ciro B Rosa.
* GitHub: https://github.com/cirobr
* LinkedIn: https://www.linkedin.com/in/cirobrosa/


## Syntax:

With no arguments, all models accept 3-channels Float32 input and deliver 1-channel mask with sigmoid output activation.

    model = UNet5()

If ch_out > 1, output mask activation becomes softmax. For instance, a model with 3-channels input and 2-channels output becomes:

    model = UNet5(3,2)


### UNet5, UNet4 syntax:

    UNet5(ch_in::Int=3, ch_out::Int=1;   # input/output channels
        activation    = relu,            # activation function
        alpha::Int    = 1,               # channels divider
        verbose::Bool = false,           # output feature maps
    )

UNet5() has internally five encoder/decoder stages, each of them delivering features with respectivelly [64, 128, 256, 512, 1024] channels.

Argument "alpha" modulates the number of internal channels proportionally. For instance, alpha == 2 delivers [32, 64, 128, 256, 512] channels.

Argument "verbose" == false delivers output mask with same (H,W) size as input images.

Argument "verbose" == true delivers a two-elements vector: first element is the same output as verbose == false; and second element are the intermediate feature model outputs, which are useful for knowledge distillation.

    verbose == false => y_hat
    verbose == true  => [ yhat, [encoder[1:5] - decoder[6:9] - logits[10]] ] (UNet5)
    verbose == true  => [ yhat, [encoder[1:4] - decoder[5:7] - logits[8]] ]  (UNet4)


### MobileUnet syntax:

    MobileUNet(ch_in::Int=3, ch_out::Int=1;   # input/output channels
        verbose::Bool = false,                # output feature maps
    )

    verbose == false => y_hat
    verbose == true  => [ yhat, [encoder[1:5] - decoder[6:13] - logits[14]] ]


### ESPNet syntax:

    ESPnet(ch_in::Int=3, ch_out::Int=1;     # input/output channels
        activation=relu,                    # activation function
        alpha2::Int=2, alpha3::Int=3,       # modulation of encoder's expansive blocks
        verbose::Bool=false,                # output feature maps
    )

Arguments alpha2 and alpha3 are modulation parameters of encoder's expansive blocks. User shall refer to original article for details.

    verbose == false => y_hat
    verbose == true  => [ yhat, [encoder[1:10] - bridges[11:13] - decoder[14:17] - logits[18]] ]


## Versions:

### v0.1.1
* Added examples folder.

### v0.1.0
* First public version.
* Cleaned up code.
* UNet2 removed.

### v0.0.19
* Added compatibility with Flux v0.16.

### v0.0.18
* Added compatibility with Flux v0.15.
* Added examples folder.

### v0.0.17
* U-Net feature outputs are revised such that the second conv 3x3 at each encoder/decoder level is finalized with BatchNorm() and a nonlinearity.
* Compatibility frozen with Flux = v0.14.17

### v0.0.16
* Added features output to ESPNet

### v0.0.15
* ESPNet added
* Improved dropouts
* Unfrozen compatibility with Flux

### v0.0.12
* Largely improved MobileUNet.
* Compatibility frozen with Flux = v0.14.16

### v0.0.11
* Intermediate features, besides model output, are made avaliable at UNets.

### v0.0.8
* ESPNet temporalily removed, until development is completed.

### v0.0.7
* UNet5, UNet4, UNet2 are mature models.
* MobileUNet works well. Needs mode experiments.
* ESPNet on probation, performance issues need investigation.
