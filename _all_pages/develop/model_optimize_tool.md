---
layout: post
title: 模型转化方法
---

Lite架构在预测过程中表现出来的高性能得益于其丰富的优化组件，其中包括量化、子图融合、混合调度、Kernel优选等等策略。为了使优化过程更加方便易用，我们提供了**Model Optimize Tool**来自动完成优化步骤，输出一个轻量的、最优的可执行模型。具体使用方法介绍如下：

## 准备model_optimize_tool

可以选择下载或者手动编译model_optimize_tool模型优化工具。

### 下载model_optimize_tool

从 [Paddle-Lite Release](https://github.com/PaddlePaddle/Paddle-Lite/releases/)官网下载最新版本的`model_optimize_tool`

![model_optimize_tool](https://user-images.githubusercontent.com/45189361/65481346-8d2e7100-dec7-11e9-848b-b237a2f4a3ff.png)

注意：运行前需解压model_optimize_tool并添加可执行权限 

```
gunzip ./model_optimize_tool.gz
chmod +x model_optimize_tool
```

### 编译model_optimize_tool

1、参照 [编译安装](../source_compile) 进行环境配置和编译

2、进入docker中PaddleLite根目录，```git checkout [release-version-tag]```切换到release分支

3、执行如下命令编译model_optimize_tool
```bash
./lite/tools/build.sh build_optimize_tool 
```
4、编译完成，优化工具在```Paddle-Lite/build.model_optimize_tool/lite/api/model_optimize_tool```

## 使用方法

1、准备需要优化的fluid模型

fluid模型有两种形式，combined形式（权重保存为一个param文件）和非combined形式（权重保存为一个一个单独的文件），model_optimize_tool支持对这两种形式的fluid模型进行直接优化。

2、使用model_optimize_tool对模型进行优化(**需要在 x86 PC 端执行**)

```shell
./model_optimize_tool \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=(protobuf|naive_buffer) \
    --optimize_out=<output_optimize_model_dir> \
    --valid_targets=(arm|opencl|x86) \
    --prefer_int8_kernel=(true|false) \
    --record_tailoring_info =(true|false)
```

| 选项         | 说明 |
| ------------------- | ------------------------------------------------------------ |
| --model_dir         | 待优化的fluid模型（非combined形式）的路径，其中包括网络结构文件和一个一个单独保存的权重文件。|
| --model_file        | 待优化的fluid模型（combined形式）的网络结构路径。 |
| --param_file        | 待优化的fluid模型（combined形式）的权重文件路径。 |
| --optimize_out_type | 输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf。 |
| --optimize_out      | 优化模型的输出路径。                                         |
| --valid_targets     | 指定模型可执行的backend，目前可支持x86、arm、opencl，您可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。默认为arm。 |
| --prefer_int8_kernel | 若待优化模型为int8量化模型（如量化训练得到的量化模型），则设置该选项为true以使用int8内核函数进行推理加速，默认为false。                          |
| --record_tailoring_info | 当使用**根据模型裁剪库文件**功能时，则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false。       |
* 如果待优化的fluid模型是非combined形式，请设置`--model_dir`，忽略`--model_file`和`--param_file`。
* 如果待优化的fluid模型是combined形式，请设置`--model_file`和`--param_file`，忽略`--model_dir`。
* 优化后的模型包括__model__.nb和param.nb文件。
