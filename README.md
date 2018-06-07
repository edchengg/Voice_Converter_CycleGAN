# Voice Converter CycleGAN

This repository contains a pytorch implementation of the model: [CycleGAN-VC - Parallel-Data-Free Voice Conversion
Using Cycle-Consistent Adversarial Networks](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/). 

## Introduction

Cycle-consistent adversarial networks (CycleGAN) has been widely used for image conversions. It turns out that it could also be used for voice conversion. This is an implementation of CycleGAN on human speech conversions. The neural network utilized 1D gated convolution neural network (Gated CNN) for generator, and 2D Gated CNN for discriminator. The model takes Mel-cepstral coefficients ([MCEPs](https://github.com/eYSIP-2017/eYSIP-2017_Speech_Spoofing_and_Verification/wiki/Feature-Extraction-for-Speech-Spoofing)) (for spectral envelop) as input for voice conversions.

<p align="center">
    <img src = "./figures/network.png" width="100%">
</p>

## TODO
[x] Generator
[ ] Discriminator
[ ] CycleGAN

