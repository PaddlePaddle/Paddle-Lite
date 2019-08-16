# Paddle-Mobile

[![Build Status](https://travis-ci.org/PaddlePaddle/paddle-mobile.svg?branch=develop&longCache=true&style=flat-square)](https://travis-ci.org/PaddlePaddle/paddle-mobile)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/doc)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

<!--[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle-Mobile.svg)](https://github.com/PaddlePaddle/Paddle-Mobile/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)-->

Welcome to Paddle-Mobile GitHub project。Paddle-Mobile is a project of PaddlePaddle as well as a deep learning framework for embedded platforms.

欢迎来到 Paddle-Mobile GitHub 项目。Paddle-Mobile是PaddlePaddle组织下的项目，是一个致力于嵌入式平台的深度学习的框架。

## Features

- high performance in support of ARM CPU 
- support Mali GPU
- support Andreno GPU
- support the realization of GPU Metal on Apple devices
- support implementation on ZU5、ZU9 and other FPGA-based development boards
- support implementation on Raspberry Pi and other arm-linux development boards

## Features

- 高性能支持ARM CPU 
- 支持Mali GPU
- 支持Andreno GPU
- 支持苹果设备的GPU Metal实现
- 支持ZU5、ZU9等FPGA开发板
- 支持树莓派等arm-linux开发板


## Demo
- [ANDROID](https://github.com/xiebaiyuan/paddle-mobile-demo)

### 原Domo目录

[https://github.com/PaddlePaddle/paddle-mobile/tree/develop/demo](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/demo)

## Documentation

### Documentation of design

If you want to know more details about the documentation of paddle-mobile design, please refer to the link as follows. There are many previous designs and discussion: [issue](https://github.com/PaddlePaddle/paddle-mobile/issues).

[link of documentation of design](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/design_doc.md)

### Documentation of development

Documentation of development is mainly about building, running and other tasks.As a developer,you can use it with the help of contributed documents.
* [iOS](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_ios.md)
* [Android_CPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android.md)
* [Android_GPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android_GPU.md)
* [FPGA](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_fpga.md)
* [ARM_LINUX](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_arm_linux.md)

### How to contribute your documents
- [tutorial link to contribute documents](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/CONTRIBUTING.md)
- Main procedure of contributing code is covered in the document above.If you have other problems during the procedure,please send them as [issue](https://github.com/PaddlePaddle/paddle-mobile/issues). We will deal with it as quickly as possible.

## 文档

### 设计文档

关于paddle-mobile设计文档在下面链接中，如果想了解更多内容。[issue](https://github.com/PaddlePaddle/paddle-mobile/issues)中会有很多早期的设计和讨论过程。
[设计文档链接](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/design_doc.md)

### 开发文档

开发文档主要是关于编译、运行等问题。做为开发者，它可以和贡献文档共同结合使用。
* [iOS](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_ios.md)
* [Android_CPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android.md)
* [Android_GPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android_GPU.md)
* [FPGA](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_fpga.md)
* [ARM_LINUX](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_arm_linux.md)

### 贡献文档
- [贡献文档链接](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/CONTRIBUTING.md)
- 上面文档中涵盖了主要的贡献代码流程，如果在实践中您还遇到了其他问题，可以发[issue](https://github.com/PaddlePaddle/paddle-mobile/issues)。我们看到后会尽快处理。

## Acquision of Models
At present Paddle-Mobile only supports Paddle fluid training model. Models wiil be operated regularly after transformation if you have various models.
### 1. Use Paddle Fluid directly to train
It is the most reliable method to be recommanded
### 2. Transform Caffe to Paddle Fluid model
[caffe2fluid](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/caffe2fluid)
### 3. ONNX
ONNX is expanded as Open Neural Network Exchange. The project is aimed to make a full communication and usage among diffrent nerual network development frameworks.

Except for directly using fluid models trained by PaddlePaddle,you can also get certain Paddle fluid models through onnx transformation.

At present，work in support of onnx is also under operation in Baidu. Related tranformation project can be referred to here：
[https://github.com/PaddlePaddle/paddle-onnx](https://github.com/PaddlePaddle/paddle-onnx)

### 4. Download parts of testing models and testing pictures
[http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip](http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip)

- input data generated by tools from `tools/python/imagetools`.


## 模型获得
目前Paddle-Mobile仅支持Paddle fluid训练的模型。如果你手中的模型是不同种类的模型，需要进行模型转换才可以运行。
### 1. 直接使用Paddle Fluid训练
该方式最为可靠，推荐方式
### 2. caffe转为Paddle Fluid模型
[caffe2fluid](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/caffe2fluid)
### 3. ONNX
ONNX全称为“Open Neural Network Exchange”，即“开放的神经网络切换”。该项目的目的是让不同的神经网络开发框架做到互通互用。

除直接使用PaddlePaddle训练fluid版本的模型外，还可以通过onnx转换得到个别Paddle fluid模型。

目前，百度也在做onnx支持工作。相关转换项目在这里：
[https://github.com/PaddlePaddle/paddle-onnx](https://github.com/PaddlePaddle/paddle-onnx)

### 4. 部分测试模型和测试图片下载
[http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip](http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip)

- 测试输入数据可由本仓库下的脚本`tools/python/imagetools`生成。

## Communication
- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ discussion group: 696965088 (Paddle-Mobile).
- [Forums](http://ai.baidu.com/forum/topic/list/168?pageNo=1): discuss implementations, research, etc.

## 交流与反馈
- 欢迎您通过[Github Issues](https://github.com/PaddlePaddle/Paddle/issues)来提交问题、报告与建议
- QQ群: 696965088 (Paddle-Mobile)
- [论坛](http://ai.baidu.com/forum/topic/list/168): 欢迎大家在PaddlePaddle论坛分享在使用PaddlePaddle中遇到的问题和经验, 营造良好的论坛氛围

## Old version Mobile-Deep-Learning
Original MDL(Mobile-Deep-Learning) project has been transferred to [Mobile-Deep-Learning](https://github.com/allonli/mobile-deep-learning)

## 旧版 Mobile-Deep-Learning
原MDL(Mobile-Deep-Learning)工程被迁移到了这里 [Mobile-Deep-Learning](https://github.com/allonli/mobile-deep-learning)

## Copyright and License
[Apache-2.0 license](LICENSE).
