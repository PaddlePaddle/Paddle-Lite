# Paddle Lite

[简体中文](README.md) | English

[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle-Lite.svg?branch=develop&longCache=true&style=flat-square)](https://travis-ci.org/PaddlePaddle/Paddle-Lite)  [![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddle-lite.readthedocs.io/zh/develop/)  [![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle-Lite.svg)](https://github.com/PaddlePaddle/Paddle-Lite/releases)  [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


Paddle Lite is an updated version of Paddle-Mobile, an open-open source deep learning framework designed to make it easy to perform inference on mobile, embeded, and IoT devices. It is compatible with PaddlePaddle and pre-trained models from other sources.

For tutorials, please see [PaddleLite Document](https://paddle-lite.readthedocs.io/zh/develop/).

## Key Features

- Multiple platform support, covering Android and iOS devices, embedded Linux, Windows, macOS and Linux computer.
- Diverse language support, which includes Java, C++, and Python.
- High performance and light weight: optimized for on-device machine learning, reduced model and binary size, efficient inference and reduced memory usage.


## Architecture

Paddle Lite is designed to support a wide range of hardwares and devices, and it enables mixed execution of a single model on multiple devices, optimization on various phases, and light-weighted applications on devices.

![img](https://paddlelite-demo.bj.bcebos.com/devices/generic/paddle_lite_with_nnadapter.png)

As is shown in the figure above, analysis phase includes Machine IR module, and it enables optimizations like Op fusion and redundant computation pruning. Besides, excecution phase only involves Kernal execution, so it can be deployed on its own to ensure maximum light-weighted deployment.

## Key Info about the Update

The earlier Paddle-Mobile was designed to be compatible with PaddlePaddle and multiple hardwares, including ARM CPU, Mali GPU, Adreno GPU, FPGA, ARM-Linux and Apple's GPU Metal. Within Baidu, inc, many product lines have been using Paddle-Mobile.

As an update of Paddle-Mobile, Paddle Lite has incorporated many older capabilities into the [new architecture](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite).

## Special Thanks

Paddle Lite has referenced the following open-source projects:

- [ARM compute library](https://github.com/ARM-software/ComputeLibrary)
- [Anakin](https://github.com/PaddlePaddle/Anakin). The optimizations under Anakin has been incorporated into Paddle Lite, and so there will not be any future updates of Anakin. As another high-performance inference project under PaddlePaddle, Anakin has been forward-looking and helpful to the making of Paddle Lite.  



## Feedback and Community Support

- Questions, reports, and suggestions are welcome through Github Issues!
- Forum: Opinions and questions are welcome at our [PaddlePaddle Forum](https://ai.baidu.com/forum/topic/list/168)！
- WeChat Official Account: PaddlePaddle
- QQ Group Chat: 696965088
<p align="center"><img width="200" height="200"  src="https://user-images.githubusercontent.com/45189361/64117959-1969de80-cdc9-11e9-84f7-e1c2849a004c.jpeg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="200" height="200" margin="500" src="https://user-images.githubusercontent.com/45189361/64117844-cb54db00-cdc8-11e9-8c08-24bbe594608e.jpeg"/></p>
<p align="center">&#8194; WeChat Official Account&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;QQ Group Chat&#8194;&#8194;&#8194;&#8194;&#8194;</p>
