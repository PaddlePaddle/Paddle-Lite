# Paddle Lite 预测流程

## 概述

Paddle Lite 是一种轻量级、灵活性强、易于扩展的高性能的深度学习预测框架，它可以支持诸如 ARM、OpenCL 、NPU 等等多种终端，同时拥有强大的图优化及预测加速能力。如果您希望将 Paddle Lite 框架集成到自己的项目中，那么只需要如下几步简单操作即可。


![workflow](https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite/develop/docs/images/workflow.png)

## 预测流程

**一. 准备模型**

Paddle Lite 框架直接支持模型结构为[ PaddlePaddle ](https://www.paddlepaddle.org.cn/)深度学习框架产出的模型格式。在 PaddlePaddle 静态图模式下，使用`save_inference_model`这个 API 保存预测模型，Paddle Lite 对此类预测模型已经做了充分支持；在 PaddlePaddle 动态图模式下，使用`paddle.jit.save`这个 API 保存预测模型，Paddle Lite 可以支持绝大部分此类预测模型了。

如果您手中的模型是由诸如 Caffe 、Tensorflow 、PyTorch 等框架产出的，那么您可以使用 [ X2Paddle ](https://github.com/PaddlePaddle/X2Paddle) 工具将模型转换为 PadddlePaddle 格式。

**二. 模型优化**

Paddle Lite 框架拥有优秀的加速、优化策略及实现，包含量化、子图融合、Kernel 优选等优化手段。优化后的模型更轻量级，耗费资源更少，并且执行速度也更快。
这些优化通过 Paddle Lite 提供的[ opt 工具](../user_guides/model_optimize_tool) 实现。opt 工具还可以统计并打印出模型中的算子信息，并判断不同硬件平台下 Paddle Lite 的支持情况。您获取 PaddlePaddle 格式的模型之后，一般需要通过 opt 工具做模型优化。opt 工具的下载和使用，请参考 [模型优化方法](../user_guides/model_optimize_tool)。

>> **注意**: 为了减少第三方库的依赖、提高 Paddle Lite 预测框架的通用性，在移动端使用 Paddle Lite API 您需要准备 Naive Buffer 存储格式的模型。

**三. 下载或编译**

Paddle Lite 提供了 `Android/IOS/ArmLinux/Windows/MacOS/Ubuntu` 平台的官方 Release 预测库下载，我们优先推荐您直接下载 [Paddle Lite 预编译库](../quick_start/release_lib)。您也可以根据目标平台选择对应的 [源码编译方法](../source_compile/compile_env)。Paddle Lite 提供了源码编译脚本，位于 `lite/tools/` 文件夹下，只需要 [准备环境](../source_compile/docker_env) 和 `lite/tools/` 文件夹 [脚本](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/tools) 两个步骤即可一键编译得到目标平台的 Paddle Lite 预测库。

**四. 开发应用程序**

Paddle Lite提供了 `C++` 、`Java` 、`Python` 三种 `API` ，只需简单五步即可完成预测（以 `C++ API` 为例）：

1. 声明 `MobileConfig` ，设置第二步优化后的模型文件路径，或选择从内存中加载模型
2. 创建 `Predictor` ，调用 `CreatePaddlePredictor` 接口，一行代码即可完成引擎初始化
3. 准备输入，通过 `predictor->GetInput(i)` 获取输入变量，并为其指定输入大小和输入值
4. 执行预测，只需要运行 `predictor->Run()` 一行代码，即可使用 Paddle Lite 框架完成预测执行
5. 获得输出，使用 `predictor->GetOutput(i)` 获取输出变量，并通过 `data<T>` 取得输出值

Paddle Lite 提供了 `C++` 、`Java` 、`Python` 三种 `API` 的完整使用示例和开发说明文档，您可以参考示例中的说明文档进行快速学习，并集成到您自己的项目中去。

- [ C++ 完整示例](../user_guides/cpp_demo)
- [ Java 完整示例](../user_guides/java_demo)
- [ Python 完整示例](../user_guides/python_demo)

此外，针对不同的硬件平台，Paddle Lite 提供了各个平台的完整示例：

- [ Android 示例](../demo_guides/android_app_demo)
- [ IOS 示例](../demo_guides/ios_app_demo)
- [ ARMLinux 示例](../demo_guides/linux_arm_demo)
- [ X86 示例](../demo_guides/x86)
- [ OpenCL 示例](../demo_guides/opencl)
- [ FPGA 示例](../demo_guides/fpga)
- [华为 NPU 示例](../demo_guides/huawei_kirin_npu)
- [昆仑芯 XPU 示例](../demo_guides/kunlunxin_xpu)
- [瑞芯微 NPU 示例](../demo_guides/rockchip_npu)
- [晶晨 NPU 示例](../demo_guides/amlogic_npu)
- [联发科 APU 示例](../demo_guides/mediatek_apu)

您也可以下载以下基于 Paddle Lite 开发的预测 APK 程序，安装到 Andriod 平台上，先睹为快：

- [图像分类](https://paddlelite-demo.bj.bcebos.com/apps/android/mobilenet_classification_demo.apk)
- [目标检测](https://paddlelite-demo.bj.bcebos.com/apps/android/yolo_detection_demo.apk)
- [口罩检测](https://paddlelite-demo.bj.bcebos.com/apps/android/mask_detection_demo.apk)
- [人脸关键点](https://paddlelite-demo.bj.bcebos.com/apps/android/face_keypoints_detection_demo.apk)
- [人像分割](https://paddlelite-demo.bj.bcebos.com/apps/android/human_segmentation_demo.apk)

## 更多测试工具

为了使您更好的了解并使用 Paddle Lite 框架，我们向有进一步使用需求的用户开放了 [Profiler 工具](../user_guides/profiler)。该工具具体分为性能 Profiler 和精度 Profiler：
- 性能 Profiler 工具可以帮助您了解每个 `Op` 的执行时间消耗，其会自动统计 `Op` 执行的次数，最长、最短、平均执行时间等等信息，为性能调优做一个基础参考；
- 精度 Profiler 工具用于模型逐层精度统计，可以获取到模型中每个 `Op` 的输出 tensor 精度信息，能够快速定位计算精度出现问题的 `Op`。
