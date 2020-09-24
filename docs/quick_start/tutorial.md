# Lite 预测流程

Lite是一种轻量级、灵活性强、易于扩展的高性能的深度学习预测框架，它可以支持诸如ARM、OpenCL、NPU等等多种终端，同时拥有强大的图优化及预测加速能力。如果您希望将Lite框架集成到自己的项目中，那么只需要如下几步简单操作即可。


![workflow](https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite/develop/docs/images/workflow.png)

**一. 准备模型**

Paddle Lite框架直接支持模型结构为[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)深度学习框架产出的模型格式。目前PaddlePaddle用于推理的模型是通过[save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/save_inference_model_cn.html#save-inference-model)这个API保存下来的。
如果您手中的模型是由诸如Caffe、Tensorflow、PyTorch等框架产出的，那么您可以使用 [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) 工具将模型转换为PadddlePaddle格式。

**二. 模型优化**

Paddle Lite框架拥有优秀的加速、优化策略及实现，包含量化、子图融合、Kernel优选等优化手段。优化后的模型更轻量级，耗费资源更少，并且执行速度也更快。
这些优化通过Paddle Lite提供的opt工具实现。opt工具还可以统计并打印出模型中的算子信息，并判断不同硬件平台下Paddle Lite的支持情况。您获取PaddlePaddle格式的模型之后，一般需要通该opt工具做模型优化。opt工具的下载和使用，请参考 [模型优化方法](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)。

**注意**: 为了减少第三方库的依赖、提高Lite预测框架的通用性，在移动端使用Lite API您需要准备Naive Buffer存储格式的模型。

**三. 下载或编译**

Paddle Lite提供了Android/iOS/X86平台的官方Release预测库下载，我们优先推荐您直接下载 [Paddle Lite预编译库](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html)。
您也可以根据目标平台选择对应的[源码编译方法](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html#id2)。Paddle Lite 提供了源码编译脚本，位于 `lite/tools/`文件夹下，只需要 [准备环境](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html) 和 [调用编译脚本](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html#id2) 两个步骤即可一键编译得到目标平台的Paddle Lite预测库。

**四. 开发应用程序**

Paddle Lite提供了C++、Java、Python三种API，只需简单五步即可完成预测（以C++ API为例）：

1. 声明`MobileConfig`，设置第二步优化后的模型文件路径，或选择从内存中加载模型
2. 创建`Predictor`，调用`CreatePaddlePredictor`接口，一行代码即可完成引擎初始化
3. 准备输入，通过`predictor->GetInput(i)`获取输入变量，并为其指定输入大小和输入值
4. 执行预测，只需要运行`predictor->Run()`一行代码，即可使用Lite框架完成预测执行
5. 获得输出，使用`predictor->GetOutput(i)`获取输出变量，并通过`data<T>`取得输出值

Paddle Lite提供了C++、Java、Python三种API的完整使用示例和开发说明文档，您可以参考示例中的说明快速了解使用方法，并集成到您自己的项目中去。

- [C++完整示例](cpp_demo.html)
- [Java完整示例](java_demo.html)
- [Python完整示例](python_demo.html)

针对不同的硬件平台，Paddle Lite提供了各个平台的完整示例：

- [Android示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/android_app_demo.html)
- [iOS示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/ios_app_demo.html)
- [ARMLinux示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/linux_arm_demo.html)
- [X86示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/x86.html)
- [OpenCL示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/opencl.html)
- [FPGA示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/fpga.html)
- [华为NPU示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/huawei_kirin_npu.html)
- [百度XPU示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/baidu_xpu.html)
- [瑞芯微NPU示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/rockchip_npu.html)
- [联发科APU示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/mediatek_apu.html)

您也可以下载以下基于Paddle-Lite开发的预测APK程序，安装到Andriod平台上，先睹为快：

- [图像分类](https://paddlelite-demo.bj.bcebos.com/apps/android/mobilenet_classification_demo.apk)  
- [目标检测](https://paddlelite-demo.bj.bcebos.com/apps/android/yolo_detection_demo.apk) 
- [口罩检测](https://paddlelite-demo.bj.bcebos.com/apps/android/mask_detection_demo.apk)  
- [人脸关键点](https://paddlelite-demo.bj.bcebos.com/apps/android/face_keypoints_detection_demo.apk) 
- [人像分割](https://paddlelite-demo.bj.bcebos.com/apps/android/human_segmentation_demo.apk)

## 更多测试工具

为了使您更好的了解并使用Lite框架，我们向有进一步使用需求的用户开放了 [Debug工具](../user_guides/debug) 和 [Profile工具](../user_guides/debug)。Lite Model Debug Tool可以用来查找Lite框架与PaddlePaddle框架在执行预测时模型中的对应变量值是否有差异，进一步快速定位问题Op，方便复现与排查问题。Profile Monitor Tool可以帮助您了解每个Op的执行时间消耗，其会自动统计Op执行的次数，最长、最短、平均执行时间等等信息，为性能调优做一个基础参考。您可以通过 [相关专题](../user_guides/debug) 了解更多内容。
