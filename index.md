---
layout: post
title: Paddle-Lite 文档
---

## 总体概述

Paddle-Lite 框架是 PaddleMobile 新一代架构，重点支持移动端推理预测，特点**高性能、多硬件、轻量级** 。支持PaddleFluid/TensorFlow/Caffe/ONNX模型的推理部署，目前已经支持 ARM CPU, Mali GPU, Adreno GPU, Huawei NPU 等多种硬件，正在逐步增加 X86 CPU, Nvidia GPU 等多款硬件，相关硬件性能业内领先。


## 简介

- [技术特点]({{ site.baseurl }}{% post_url 2019-09-16-tech_highlights %})
- [架构设计]({{ site.baseurl }}{% post_url 2019-09-16-architecture %})
- [Road Map]({{ site.baseurl }}{% post_url 2019-09-16-roadmap %})

## Benchmark

- [最新性能]({{ site.baseurl }}{% post_url 2019-09-16-benchmark %})
- [测试方法]({{ site.baseurl }}{% post_url 2019-09-16-benchmark_tools %})

## 安装

- [源码编译]({{ site.baseurl }}{% post_url 2019-09-16-source_compile %})

## 使用

- [使用流程]({{ site.baseurl }}{% post_url 2019-09-16-tutorial %})
- [C++实例]({{ site.baseurl }}{% post_url 2019-09-16-cpp_demo %})
- [Java实例]({{ site.baseurl }}{% post_url 2019-09-16-java_demo %})
- [Android/IOS APP demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo %})
- [模型转化方法]({{ site.baseurl }}{% post_url 2019-09-16-model_optimize_tool %})

## 进阶

- [通过 X2Paddle 支持 Caffe, TensorFlow 模型]({{ site.baseurl }}{% post_url 2019-09-16-x2paddle %})
- [模型量化]({{ site.baseurl }}{% post_url 2019-09-16-model_quantization %})
- [支持Op列表]({{ site.baseurl }}{% post_url 2019-09-16-support_operation_list %})
- [新增Op方法]({{ site.baseurl }}{% post_url 2019-09-16-add_new_operation %})
- [测试工具]({{ site.baseurl }}{% post_url 2019-09-16-debug_tools %})
- [调试方法]({{ site.baseurl }}{% post_url 2019-09-16-debug_tools %})
- [使用华为NPU]({{ site.baseurl }}{% post_url 2019-09-16-npu %})
- [使用Android GPU]({{ site.baseurl }}{% post_url 2019-09-16-opencl %})
- [使用FPGA]({{ site.baseurl }}{% post_url 2019-09-16-fpga %})

## 开发者文档

- [开发基础须知](./for-developer)
- [架构详解](./architecture-intro)

## FAQ

- 问题或建议可以[发Issue](https://github.com/PaddlePaddle/Paddle-Lite/issues)，为加快问题解决效率，可先检索是否有类似问题，我们也会及时解答！
- 欢迎加入Paddle-Lite百度官方QQ群：696965088

## paddle-mobile

- [paddle-mobile 编译](./mobile)
