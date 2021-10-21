# IOS 工程示例

## 多种应用场景

我们提供 Paddle Lite 示例工程[ Paddle-Lite-Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo)，其中包含[ Android ](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo)、[ IOS ](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-ios-demo)和[ Armlinux ](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-armlinux-demo)平台的示例工程。

IOS demo涵盖[图像分类](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/image_classification_demo)、[目标检测](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/object_detection_demo)2个应用场景。

### 1. 图像分类

图像分类是 Paddle Lite 提供的图像处理 demo ，在移动端上提供了实时的物体识别能力，可以应用到生产线自动分拣或质检、识别医疗图像、辅助医生肉眼诊断等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat2.jpg"/></p>

### 2. 物体检测

物体检测是 Paddle Lite 提供的图像识别 demo ，在移动端上提供了检测多个物体的位置、名称、位置及数量的能力。可以应用到视频监控（是否有违规物体或行为）、工业质检（微小瑕疵的数量和位置）、医疗诊断（细胞计数、中药识别）等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog2.jpg"/></p>

## IOS demo 部署方法

下面我们以**目标检测( object_detection_demo )** 为例讲解如何部署 IOS 工程。

**目的**：将基于 Paddle Lite 预测库的 IOS APP部署到苹果手机，实现物体检测。

**需要的环境**：Mac 电脑上安装 Xcode、苹果手机、下载到本地的[ Paddle-Lite-Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo)工程

**部署步骤**：

1、 目标检测的 IOS 示例位于 `Paddle-Lite-Demo\PaddleLite-ios-demo\object_detection_demo`

2、终端中执行 `download_dependencies.sh` 脚本自动下载模型和 Paddle Lite 预测库

```shell
cd PaddleLite-ios-demo          # 1. 终端中进入 Paddle-Lite-Demo\PaddleLite-ios-demo
sh download_dependencies.sh     # 2. 执行脚本下载依赖项 （需要联网）
```

下载完成后会出现提示： `Extract done `

3、用 Xcode 打开 `ios-detection_demo/detection_demo.xcodeproj` 文件，修改工程配置。
依次修改 `General/Identity` 和 `Signing&Capabilities` 属性，替换为自己的工程代号和团队名称。（必须修改，不然无法通过编译）

![Xcode1](https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode1.png)



![Xcode2](https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode2.png)

4、 IPhone 手机连接电脑，在 Xcode 中连接自己的手机 （第一次连接 IPhone 到电脑时，需要在 IPhone 的`设置->通用->设备管理`中选择本电脑并信任）

<p align="center"><img width="600" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode-phone.jpg"/>

5、按下左上角的 Run 按钮，自动编译 APP 并安装到手机。在苹果手机中设置信任该 APP（进入`设置->通用->设备管理`，选中新安装的 APP 并`验证该应用`）

成功后效果如下，图一：APP安装到手机        图二： APP打开后的效果，会自动识别图片中的物体并标记

<p align="center"><img width="300" height="450"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/IOS2.jpeg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="300" height="450"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/IOS3.jpeg"/></p>

## IOS demo 结构讲解

IOS 示例的代码结构如下图所示：

<p align="center"><img width="600" height="450"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/IOS-struct.png"/>

   1、 `mobilenetv1-ssd`： 模型文件( opt 工具转化后 Paddle Lite 模型)

```shell
# 位置：
ios-detection_demo/detection_demo/models/mobilenetv1-ssd
```

  2、`libpaddle_api_light_bundled.a`、`paddle_api.h`：Paddle-Lite C++ 预测库和头文件

```shell
# 位置：
# IOS 预测库
ios-detection_demo/detection_demo/lib/libpaddle_api_light_bundled.a
# 预测库头文件
ios-detection_demo/detection_demo/include/paddle_api.h
ios-detection_demo/detection_demo/include/paddle_use_kernels.h
ios-detection_demo/detection_demo/include/paddle_use_ops.h
```

  3、 `ViewController.mm`：主要预测代码

```shell
# 位置
ios-detection_demo/detection_demo/ViewController.mm
```

## 代码讲解 （如何使用 Paddle Lite `C++ API` 执行预测）

IOS 示例基于 `C++ API` 开发，调用 Paddle Lite `C++ API` 包括以下五步。更详细的 `API` 描述参考： [ Paddle-Lite C++ API ](../api_reference/cxx_api_doc)。

```c++
#include <iostream>
// 引入 C++ API
#include "include/paddle_api.h"
#include "include/paddle_use_ops.h"
#include "include/paddle_use_kernels.h"

// 1. 设置 MobileConfig
MobileConfig config;
config.set_model_from_file(<modelPath>); // 设置 NaiveBuffer 格式模型路径
config.set_power_mode(LITE_POWER_NO_BIND); // 设置 CPU 运行模式
config.set_threads(4); // 设置工作线程数
// 如果需要使用 Metal 在 GPU 上加速预测，需要额外进行以下配置, 需将编译生成的 lite.metallib 拷贝到<metal_lib>路径下
config.set_metal_lib_path(<metal_lib>);
config.set_metal_use_mps(true);

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

### Q&A:
问题：
- 提示 `feed kernel not found`:
  - 在包含 Paddle Lite 头文件的时候加上以下两行即可
  ```
   #include "include/paddle_use_ops.h"
   #include "include/paddle_use_kernels.h"
  ```
