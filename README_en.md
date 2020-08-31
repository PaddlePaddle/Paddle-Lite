# Paddle Lite

[简体中文](README.md) | English

[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle-Lite.svg?branch=develop&longCache=true&style=flat-square)](https://travis-ci.org/PaddlePaddle/Paddle-Lite)  [![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddle-lite.readthedocs.io/zh/latest/)  [![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle-Lite.svg)](https://github.com/PaddlePaddle/Paddle-Lite/releases)  [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


Paddle Lite is an updated version of Paddle-Mobile, an open-open source deep learning framework designed to make it easy to perform inference on mobile, embeded, and IoT devices. It is compatible with PaddlePaddle and pre-trained models from other sources.

For tutorials, please see [PaddleLite Document](https://paddle-lite.readthedocs.io/zh/latest/).

## Key Features

### Light Weight

On mobile devices, execution module can be deployed without third-party libraries, because our excecution module and analysis module are decoupled.

On ARM V7, only 800KB are taken up, while on ARM V8, 1.3MB are taken up with the 80 operators and 85 kernels in the dynamic libraries provided by Paddle Lite.

Paddle Lite enables immediate inference without extra optimization.

### High Performance

Paddle Lite enables device-optimized kernels, maximizing ARM CPU performance.

It also supports INT8 quantizations with [PaddleSlim model compression tools](https://github.com/PaddlePaddle/models/tree/v1.5/PaddleSlim), reducing the size of models and increasing the performance of models.

On Huawei NPU and FPGA, the performance is also boosted.

The latest benchmark is located at [benchmark](https://paddlepaddle.github.io/Paddle-Lite/develop/benchmark/)

### High Compatibility

Hardware compatibility: Paddle Lite supports a diversity of hardwares — ARM CPU, Mali GPU, Adreno GPU, Huawei NPU and FPGA. In the near future, we will also support AI microchips from Cambricon and Bitmain.

Model compatibility: The Op of Paddle Lite is fully compatible to that of PaddlePaddle. The accuracy and performance of 18 models (mostly CV models and OCR models) and 85 operators have been validated. In the future, we will also support other models.

Framework compatibility: In addition to models trained on PaddlePaddle, those trained on Caffe and TensorFlow can also be converted to be used on Paddle Lite, via [X2Paddle](https://github.com/PaddlePaddle/X2Paddle). In the future to come, we will also support models of ONNX format.

## Architecture

Paddle Lite is designed to support a wide range of hardwares and devices, and it enables mixed execution of a single model on multiple devices, optimization on various phases, and leight-weighted applications on devices.

![img](https://user-images.githubusercontent.com/45189361/70908123-6ce4fd00-2045-11ea-97e1-ad08446c5c86.png)

As is shown in the figure above, analysis phase includes Machine IR module, and it enables optimizations like Op fusion and redundant computation pruning. Besides, excecution phase only involves Kernal exevution, so it can be deployed on its own to ensure maximized light-weighted deployment.

## Key Info about the Update

The earlier Paddle-Mobile was designed to be compatible with PaddlePaddle and multiple hardwares, including ARM CPU, Mali GPU, Adreno GPU, FPGA, ARM-Linux and Apple's GPU Metal. Within Baidu, inc, many product lines have been using Paddle-Mobile. For more details, please see: [mobile/README](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/mobile/README.md).

As an update of Paddle-Mobile, Paddle Lite has incorporated many older capabilities into the [new architecture](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite). For the time being, the code of Paddle-mobile will be kept under the directory `mobile/`, before complete transfer to Paddle Lite.

For demands of Apple's GPU Metal and web front end inference, please see `./metal` and `./web` . These two modules will be further developed and maintained.

## Special Thanks

Paddle Lite has referenced the following open-source projects:

- [ARM compute library](https://github.com/ARM-software/ComputeLibrary%29)
- [Anakin](https://github.com/PaddlePaddle/Anakin). The optimizations under Anakin has been incorporated into Paddle Lite, and so there will not be any future updates of Anakin. As another high-performance inference project under PaddlePaddle, Anakin has been forward-looking and helpful to the making of Paddle Lite.  



## Feedback and Community Support

- Questions, reports, and suggestions are welcome through Github Issues!
- Forum: Opinions and questions are welcome at our [PaddlePaddle Forum](https://ai.baidu.com/forum/topic/list/168)！
- WeChat Official Account: PaddlePaddle
- QQ Group Chat: 696965088
<p align="center"><img width="200" height="200"  src="https://user-images.githubusercontent.com/45189361/64117959-1969de80-cdc9-11e9-84f7-e1c2849a004c.jpeg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="200" height="200" margin="500" src="https://user-images.githubusercontent.com/45189361/64117844-cb54db00-cdc8-11e9-8c08-24bbe594608e.jpeg"/></p>
<p align="center">&#8194; WeChat Official Account&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;QQ Group Chat&#8194;&#8194;&#8194;&#8194;&#8194;</p>
