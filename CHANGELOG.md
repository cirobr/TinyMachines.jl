### v0.3.2
* Cleanup, more efficient code.
* ESPNet: ConvPReLU returned.

### v0.3.1
* Fixed sigmoid output activation.

### v0.3.0
* MobileUNet: Recursions eliminated from irblocks.
* ESPNet: ConvPReLU replaced by leakyrelu due to long compilation time.
* ESPNet: Recursions eliminated from ESP blocks (faster compilation).
* ESPNet: Simplified esp blocks.
* Created model constructors: unet5, unet4, mobileunet, espnet.
* Models with preferred hyperparameters and output activations are now built from constructors.
* Updated documentation.

### v0.2.0
* MobileUNet: added activation argument, default to relu6.
* ConvPReLU: add and export new layer (PReLU implementation).
* ESPBlock: simplified.
* ESPNet: ConvPReLU incorporated.
* UNet4, UNet5, MobuleUNet: cleanup.
* Convolutions: cleanup.
* Tests: cleanup.
* Add single epoch tests for all models.

### v0.1.3
* Update github workflows.

### v0.1.2
* Changed UPBlocks in UNets.
* Compatibility expanded to Julia 1.10 - 1.11.

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
