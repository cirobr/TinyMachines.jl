![alt text](./images/logo-name-tm.png)

# TinyMachines.jl

[![Build Status](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml?query=branch%3Amain)

A collection of tiny machine learning models for semantic image segmentation in IoT devices, written in Julia/Flux.

Besides regular mask outputs, all models deliver their internal feature maps as additional outputs, which are useful for model compression through knowledge distillation.


## UNet5, UNet4

UNet5 is the classic U-Net architecture, with five encoder/decoder levels. UNet4 has four levels.

Reference:
* "U-Net: Convolutional Networks for Biomedical Image Segmentation" ([arXiv](https://arxiv.org/abs/1505.04597)). Credits: Ronnenberger, Olaf; Fischer, Philipp; and Brox, Thomas.


## MobileUNet

Mobile-Unet has the same encoder structure as the Mobilenet-V2 classification model, and the same u-shape and skip connection principles as the U-Net.

Reference:
* "Mobile-Unet: An efficient convolutional neural network for fabric defect detection" ([doi.org](https://doi.org/10.1177/0040517520928604)). Credits: Jing, Junfeng; Wang, Zhen; Ratsch, Matthias; and Zhang, Huanhuan.


## ESPNet
ESPNet utilizes the Efficient Spatial Pyramid module and the PReLU nonlinearity.

Reference:
* Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation" ([arXiv](https://arxiv.org/abs/1803.06815)). Credits: Mehta, Sachin; Rastegari, Mohammad; Caspi, Anat; Shapiro, Linda; and Hajishirzi, Hannaneh.


## PReLU
PReLU is a trainable nonlinearity, which is incorporated in ESPNet.

Reference:
* "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" ([arXiv](https://arxiv.org/abs/1502.01852)). Credits: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.


## Credits
Credits for the original architectures go to the references' authors, as aforementioned.

Credits for the implementations in Julia/Flux go to Ciro B Rosa.
* GitHub: https://github.com/cirobr
* LinkedIn: https://www.linkedin.com/in/cirobrosa/


## Syntax

With no arguments, all models accept 3-channels Float32 input and deliver 2-channels logits output

Remark: Final activation has been removed from all models, from v0.3.7 onwards. Constructors remain unchanged.

```
model = UNet5() = UNet5(3,2)   # three input channels, two output channels

model = UNet4(3,5)   # three input channels, five output channels

```


## Models

```
# Both UNet5() and UNet() calls are the same classic U-Net
UNet5(3, 2;                # input/output channels
    activation = relu,     # activation function
)
```

```
UNet4(3, 2;                # input/output channels
    activation = relu,     # activation function
)
```

```
MobileUNet(3, 2;           # input/output channels
    activation = relu6,    # activation function
)
```

```
# Model calls for alpha2=5, alpha3=8, which differ from default constructor
ESPNet(3, 2;               # input/output channels
    activation = "prelu"   # activation function (if "prelu", use between quotes) 
)
```

## Features

```
model = UNet5()

yhat  = model(x)    # return_features default to false, yhat = logits

yhat  = model(x; return_features=true)
yhat.logits         # output logits (same output for return_features=false)
yhat.encoder.enc1   # output encoder feature first level
yhat.encoder.enc2
yhat.encoder.enc3
yhat.encoder.enc4
yhat.encoder.enc5   # output encoder feature fifth level
```

```
model = MobileUNet()

yhat  = model(x)    # return_features default to false, yhat = logits

yhat  = model(x; return_features=true)
yhat.logits         # output logits (same output for return_features=false)
yhat.encoder.x1     # output encoder feature first level
yhat.encoder.x2
yhat.encoder.x3
yhat.encoder.x4
yhat.encoder.x5     # output encoder feature fifth level
```

```
model = ESPNet()

yhat  = model(x)    # return_features default to false, yhat = logits

yhat  = model(x; return_features=true)
yhat.logits         # output logits (same output for return_features=false)
yhat.encoder.ct1    # output encoder feature first level
yhat.encoder.ct2
yhat.encoder.ct3    # output encoder feature third level
```

## Constructors

Constructors are models which allow access to a multitude of hyperparameters. Each model from above has been build with the aid of these constructors, where hyperparameters are chosen for performance.

```
# Both unet5() and unet() calls are the same classic unet
unet5(3, 1;                               # input/output channels
    activation = relu,                    # activation function
    alpha = 1,                            # channels divider
    edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # dropout rates
    ddrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
)
```

```
unet4(3, 1;                               # input/output channels
    activation = relu,                    # activation function
    alpha = 1,                            # channels divider
    edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # dropout rates
    ddrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
)
```

Both unet5() and unet() call the same classic U-Net with five encoder/decoder stages, each of them delivering features with respectivelly $[64, 128, 256, 512, 1024]$ channels. unet4() has four encoder/decoder stages and $[64, 128, 256, 512]$ channels.

Argument $alpha$ in unets modulates the number of internal channels proportionally. For instance, $alpha == 2$ delivers $[32, 64, 128, 256, 512]$ channels.


```
mobileunet(3, 1;                          # input/output channels
    activation = relu6,                   # activation function
    edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # dropout rates
    ddrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
)
```

```
# ConvPReLU is incorporated, no need to pass activation function
espnet(3, 1;                              # input/output channels
    activation = "prelu",                 # activation function (if "prelu", use between quotes)
    alpha2 = 2,                           # expansion factor in encoder stage 2
    alpha3 = 3,                           # expansion factor in encoder stage 3
    edrops = (0.0, 0.0, 0.0),             # dropout rates for encoder
    ddrops = (0.0, 0.0),                  # dropout rates for decoder
)
```


## PReLU

```
PReLU(ch)   # number of channels
```
