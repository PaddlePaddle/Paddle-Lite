# Paddle-Mobile
 
[![Build Status](https://travis-ci.org/PaddlePaddle/paddle-mobile.svg?branch=develop&longCache=true&style=flat-square)](https://travis-ci.org/PaddlePaddle/paddle-mobile)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/doc)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

<!--[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle-Mobile.svg)](https://github.com/PaddlePaddle/Paddle-Mobile/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)-->


欢迎来到 Paddle-Mobile GitHub 项目。

Paddle-Moible是PaddlePaddle组织下的项目，是一个致力于嵌入式平台的深度学习的框架。Paddle-Moible设计思想和PaddlePaddle的最新版fluid版本保持了高度一致，同时针对嵌入式做了大量优化。设计之初就对嵌入式的性能、体积、能耗、硬件平台覆盖等方面做了考虑。

## 简单搜索线上效果

如下gif是简单搜索app的线上主体检测应用效果

![ezgif-1-050a733dfb](http://otkwwi4x8.bkt.clouddn.com/2018-07-05-ezgif-1-050a733dfb.gif)

## Demo目录

[点我](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/demo)

## Features

- **ARM CPU**

|mobilenet arm v7|1线程|2线程|4线程|
|------------|----|-----|-----|
|麒麟960(ms)|110.586|63.285|38.215|
|||||
|mobilenetssd arm v7|1线程|2线程|4线程|
|麒麟960(ms)|220.248|128.473|79.334|
|||||
|googlenet(v1) arm v7|1线程|2线程|4线程|
|麒麟960(ms)|341.965|228.724|161.531|
|||||
|squeezenet arm v7|1线程|2线程|4线程|
|麒麟960(ms)|84.080|55.641|37.182|
|||||
|yolo arm v7|1线程|2线程|4线程|
|麒麟960(ms)|129.445|80.627|50.936|

    arm cpu是paddle-mobile的主要支持方向，cpu的通用性一直是其优势。嵌入式深度学习，需要大量的cpu汇编实现。我们正在紧锣密鼓的编码，为的是能充分硬件的每一点加速能力。
    arm cpu的优化工作还在进行中，现在使用了常规的cpu优化。在arm a73上paddle-mobile arm-v7现在单核运行一次mobilenet1.0是110+ms，显然这不是我们的最终目标，我们正在用大量的汇编改写，后续性能仍会有巨大提升空间, 目前只支持armv7, 未来我们也会支持armv8。
    
- **Mali GPU**

    Mali GPU是百度和ARM合作开发的，双方团队近期都在致力于将paddle的op能无缝运行在ACL(arm compute library)。目前已经支持squeezenet，googlenet，resnet等几个网络模型，后续会继续加大力度。使全部移动端paddle op能高效运行在mali gpu上。 

- **苹果设备的GPU Metal实现**

    基于Metal实现的苹果设备的GPU预测库，也已经在实现中，近期也会有相应可运行版本。
     
- **FPGA**

    FPGA实现正在进行中，是基于Xilinx的ZU5目标开发板。

- **灵活性**

    * paddle-mobile cpu版不依赖任何第三库, 可进行快速集成。
    * 使用泛型特化进行平台切换, 可灵活切换 cpu、gpu 和其他协处理器。
    * 可根据特定的常见网络, 进行编译特定的 op, 降低编译时间, 减小包大小。
    * 使用 docker 编译, 提供统一的编译环境。
    * 高可拓展性, 方便拓展其他协处理器, 提供高性能 arm 算子实现, 方便其他协处理器开发者集成开发。
    * 直接兼容 paddle-fluid 模型, 不需要额外的转换操作。

- **体积**

    paddle-mobile从设计之初就深入考虑到移动端的包体积的问题，cpu实现中没有外部依赖。在编译过程中，如果该网络不需要的op是完全不会被打入的。同时编译选项优化也为体积压缩提供了帮助。
    除了二进制体积，我们对代码体积极力避免过大。整个仓库的代码体积也非常小。


## 文档

### 设计文档

关于paddle-mobile设计文档在下面链接中，如果想了解更多内容。[issue](https://github.com/PaddlePaddle/paddle-mobile/issues)中会有很多早期的设计和讨论过程。
[设计文档链接](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/design_doc.md)

### 开发文档

开发文档主要是关于编译、运行等问题。做为开发者，它可以和贡献文档共同结合使用。
[开发文档链接](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_doc.md)

### 贡献文档
- [贡献文档链接](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/CONTRIBUTING.md)
- 上面文档中涵盖了主要的贡献代码流程，如果在实践中您还遇到了其他问题，可以发[issue](https://github.com/PaddlePaddle/paddle-mobile/issues)。我们看到后会尽快处理。


## 模型获得
目前Paddle-Mobile仅支持Paddle fluid训练的模型。如果你手中的模型是不同种类的模型，需要进行模型转换才可以运行。
### 1. 直接使用Paddle Fluid训练
该方式最为可靠，推荐方式
### 2. caffe转为Paddle Fluid模型
[链接](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/caffe2fluid)
### 3. ONNX
ONNX全称为“Open Neural Network Exchange”，即“开放的神经网络切换”。该项目的目的是让不同的神经网络开发框架做到互通互用。

除直接使用PaddlePaddle训练fluid版本的模型外，还可以通过onnx转换得到个别Paddle fluid模型。

目前，百度也在做onnx支持工作。相关转换项目在这里：[paddle-onnx](https://github.com/PaddlePaddle/paddle-onnx)。

![](http://7xop3k.com1.z0.glb.clouddn.com/15311951836000.jpg)

### 4. 部分测试模型和测试图片下载
[下载链接](http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip)

## 问题解决

欢迎提出或解决我们的问题，有疑问可以发issue. [Github Issues](https://github.com/PaddlePaddle/paddle-mobile/issues).

## Copyright and License
Paddle-Mobile 提供相对宽松的Apache-2.0开源协议 [Apache-2.0 license](LICENSE).


## 旧版 Mobile-Deep-Learning
原MDL(Mobile-Deep-Learning)工程被迁移到了这里 [Mobile-Deep-Learning](https://github.com/allonli/mobile-deep-learning) 


