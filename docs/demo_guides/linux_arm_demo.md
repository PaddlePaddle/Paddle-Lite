# Linux(ARM) 工程示例

## 多种应用场景

我们提供 Paddle Lite 示例工程[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)，其中包含[Android](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo)、[iOS](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-ios-demo)和[Armlinux](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-armlinux-demo)平台的示例工程。

Linux(ARM) demo 涵盖[图像分类](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-armlinux-demo/image_classification_demo)、[目标检测](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-armlinux-demo/object_detection_demo) 2 个应用场景。

### 1. 图像分类

Paddle Lite 提供的图像分类 demo ，在移动端上提供了实时的物体识别能力，可以应用到生产线自动分拣或质检、识别医疗图像、辅助医生肉眼诊断等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat2.jpg"/></p>

### 2. 物体检测

Paddle Lite 提供的物体检测 demo ，在移动端上提供了检测多个物体的位置、名称、位置及数量的能力。可以应用到视频监控（是否有违规物体或行为）、工业质检（微小瑕疵的数量和位置）、医疗诊断（细胞计数、中药识别）等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog2.jpg"/></p>

## Linux(ARM) demo 部署方法

下面我们以**目标检测( object_detection_demo )** 为例讲解如何部署 Linux(ARM) 工程。

**目的**：将基于 Paddle Lite 的预测库部署到 Linux(ARM) 设备，实现物体检测的目标。

**需要的环境**：Linux(ARM) 设备、下载到本地的[ Paddle-Lite-Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo)工程

**部署步骤**：

1、 目标检测的 Linux(ARM) 示例位于 `Paddle-Lite-Demo\PaddleLite-armlinux-demo\object_detection_demo`

2、终端中执行 `download_models_and_libs.sh` 脚本自动下载模型和 Paddle Lite 预测库

```shell
cd PaddleLite-armlinux-demo          # 1. 终端中进入 Paddle-Lite-Demo\PaddleLite-armlinux-demo
sh download_models_and_libs.sh       # 2. 执行脚本下载依赖项 （需要联网）
```

下载完成后会出现提示： `Download successful!`

3、执行用例(保证 linux_arm 环境准备完成，参考[ Paddle-Lite-Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo) 要求-ARMLinux 小节)
```shell
cd object_detection_demo    # 1. 终端中进入
sh run.sh                   # 2. 执行脚本编译并执行物体检测 demo，输出预测数据和运行时间
```
demo 结果如下:
<img width="836" alt="image" src="https://user-images.githubusercontent.com/50474132/82852558-da228580-9f35-11ea-837c-e4d71066da57.png">

## 使用C++接口预测
Linux(ARM) demo 示例基于 `C++ API` 开发，调用 Paddle Lite `C++ API` 包括以下五步。更详细的 `API` 描述参考：[ Paddle Lite C++ API ](../api_reference/cxx_api_doc)。

```c++
#include <iostream>
// 引入 C++ API
#include "paddle_lite/paddle_api.h"
#include "paddle_lite/paddle_use_ops.h"
#include "paddle_lite/paddle_use_kernels.h"

// 1. 设置 MobileConfig
MobileConfig config;
config.set_model_from_file(<modelPath>); // 设置 NaiveBuffer 格式模型路径
config.set_power_mode(LITE_POWER_NO_BIND); // 设置 CPU 运行模式
config.set_threads(4); // 设置工作线程数

// 2. 创建 PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

// 3. 设置输入数据
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}

// 4. 执行预测
predictor->run();

// 5. 获取输出数据
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
std::cout << "Output shape " << output_tensor->shape()[1] << std::endl;
for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
  std::cout << "Output[" << i << "]: " << output_tensor->data<float>()[i]
            << std::endl;
}
```

## 使用 Python 接口预测

1. Python 预测库编译参考[编译 Linux](../source_compile/linux_x86_compile_arm_linux)，建议在开发版上编译。
2. [Paddle Lite Python API](../api_reference/python_api_doc)。
3. 代码参考，[Python 完整示例](../user_guides/python_demo)
