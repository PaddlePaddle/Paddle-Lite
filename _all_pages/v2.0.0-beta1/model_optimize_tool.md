---
layout: post
title: 模型转化方法
---

Lite架构在预测过程中表现出来的高性能得益于其丰富的优化组件，其中包括量化、子图融合、混合调度、Kernel优选等等策略。为了使优化过程更加方便易用，我们提供了**Model Optimize Tool**来自动完成优化步骤，输出一个轻量的、最优的可执行模型。具体使用方法介绍如下：

## 准备model_optimize_tool

可以选择下载或者手动编译model_optimize_tool可执行文件。

### 下载model_optimize_tool

```sh
wget https://paddle-inference-dist.bj.bcebos.com/PaddleLite/model_optimize_tool
chmod 777 model_optimize_tool
```

### 编译model_optimize_tool

1、参照 [编译安装](../source_compile) 进行环境配置和编译

2、进入docker中PaddleLite根目录，```git checkout develop```切换到develop分支

3、使用cmake构建目标，执行如下命令编译model_optimize_tool
```bash
./lite/tools/build.sh --arm_os=android --arm_abi=armv8 --arm_lang=gcc --android_stl=c++_static full_publish
```
4、编译完成，优化工具在```Paddle-Lite/build.lite.android.armv8.gcc/lite/api/model_optimize_tool```

## 使用方法

1、准备需要优化的fluid模型

fluid模型有两种形式，combined形式（权重保存为一个param文件）和非combined形式（权重保存为一个一个单独的文件），model_optimize_tool支持对这两种形式的fluid模型进行直接优化。

2、将model_optimize_tool和需要优化的模型文件push到手机端

3、使用model_optimize_tool对模型进行优化

```shell
./model_optimize_tool \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=(protobuf|naive_buffer) \
    --optimize_out=<output_optimize_model_dir> \
    --valid_targets=(arm|opencl|x86) \
    --prefer_int8_kernel=(true|false)
```

| 选项         | 说明 |
| ------------------- | ------------------------------------------------------------ |
| --model_dir         | 待优化的fluid模型（非combined形式）的路径，其中包括网络结构文件和一个一个单独保存的权重文件。|
| --model_file        | 待优化的fluid模型（combined形式）的网络结构路径。 |
| --param_file        | 待优化的fluid模型（combined形式）的权重文件路径。 |
| --optimize_out_type | 输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf。 |
| --optimize_out      | 优化模型的输出路径。                                         |
| --valid_targets     | 指定模型可执行的backend，目前可支持x86、arm、opencl，您可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。默认为arm。 |
| --prefer_int8_kernel | 是否启用int8量化模型，默认为false。                          |

* 如果待优化的fluid模型是非combined形式，请设置`--model_dir`，忽略`--model_file`和`--param_file`。
* 如果待优化的fluid模型是combined形式，请设置`--model_file`和`--param_file`，忽略`--model_dir`。
* 优化后的模型包括__model__.nb和param.nb文件。
