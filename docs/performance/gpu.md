# 使用 GPU 获取最佳性能

Paddle Lite 支持在 Android 和 iOS 等设备上使用 GPU 后端获取最佳的性能。

GPU 可以用来运行较大运算强度的负载任务，将模型中计算任务切分为更小的工作负载，利用 GPU 提供的大规模线程来并行工作，从而获取高吞吐和较低延迟。与 CPU 不同，GPU 支持运行在 32 位浮点模式或者 16 位浮点模式而不用通过量化来获得最佳的性能。

Paddle Lite 支持多种 GPU 后端，包括 OpenCL、[Metal](https://developer.apple.com/metal/)，支持包括 ARM Mali、Qualcomm Adreno、Apple A Series 等系列 GPU 设备。

## Android 设备使用 OpenCL 获取最佳性能
详细见 [OpenCL 部署示例](../demo_guides/opencl)。

## iOS 设备使用 Metal 获取最佳性能
这里介绍在苹果 iOS 设备上，通过使用 Metal 后端利用 GPU 设备获取最佳性能。

### 1、编译获取支持 Metal 后端的 Paddle Lite 预测库
根据[源码编译](../source_compile/compile_env)中 [macOS 环境下编译适用于 iOS 的库](../source_compile/macos_compile_ios)准备编译环境, 拉取 Paddle Lite 代码，切换到特定分支，然后在 Paddle Lite 根目录下执行编译命令。
```
./lite/tools/build_ios.sh --with_metal=ON --with_extra=ON
```

### 2、使用 opt 工具进行模型优化
```opt``` 工具可以提供包括量化、子图融合、混合调度、Kernel优选等优化方法，自动完成优化步骤生成一个轻量级的、最优的可执行模型，详细使用可以参见[模型优化工具 opt](../user_guides/model_optimize_tool) 和[使用可执行文件 opt](../user_guides/opt/opt_bin)。Metal 后端支持与 ARM 后端算子混合调度执行，模型优化方式如下：
```
./opt --model_dir=./mobilenet_v1 --valid_targets=metal,arm --optimize_out=mobilenet_v1_opt
```
以上命令可以将```mobilenet_v1```模型转化为在 iOS GPU 平台执行的 naive_buffer 格式的 Paddle Lite 支持模型，优化后文件名为```mobilenet_v1_opt.nb```。
### 3、使用 Metal 加速的 API 使用示例
Paddle Lite 提供了使用 Metal 进行加速的 API 接口，详细开发文档见 [C++ API](../api_reference/cxx_api_doc)。以下简单提供 Predictor 创建示例。
```
#include "paddle_api.h"

// 1. Set MobileConfig，model_file is configured to .nb model path and metal_lib is configured to .metallib path
MobileConfig config;
config.set_model_from_file(model_file);
config.set_metal_use_mps(true);
config.set_metal_lib_path(metal_lib);

// 2. Create PaddlePredictor by MobileConfig
std::shared_ptr<PaddlePredictor> predictor =
  CreatePaddlePredictor<MobileConfig>(config);
return predictor;
```
### 4、XCode 集成开发
iOS 开发配置见 [iOS 工程示例](../demo_guides/ios_app_demo)。配置完成后，手动对 ```include``` 和 ```lib``` 目录进行替换，编译生成的 ```.metallib 文件```也可以同时放置在 ```lib``` 目录下。

另外使用 Metal 加速会依赖 [MetalPerformaceShaders](https://developer.apple.com/documentation/metalperformanceshaders?language=objc)，需要进行如下图配置，在```Project navigator ->  Your project -> PROJECT -> Your target -> General -> Frameworks, Libraries and Embedded Content``` 中添加 ```libpaddle_api_light_bundled.a``` 和 ```MetalPerformanceShaders.framework```.
<p align="center"><img width="900" height="400"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/xcode-metal.png"/>

完成这一步骤之后，你应该已经可以运行所开发的应用程序了。

## 支持的模型与 Ops
GPU 支持的模型列表见[支持模型](../quick_start/support_model_list)，详细的 OP 支持列表见[支持算子](../quick_start/support_operation_list)。

## 优化建议
* 减少低计算量、高访存算子的使用，如 concat、slice 等。
* 尽量减少只改变形状而没有计算量的算子使用，如 reshape、transpose、permute 等。
* 不宜使用太大尺寸的卷积核，尽量使用最常用的尺寸为 1x1 和 3x3 的卷积核。当模型需要较大卷积核时，可以考虑使用小卷积核进行代替。
