# Paddle Lite 介绍

Paddle Lite 是一组工具，可帮助开发者在移动设备、嵌入式设备和 loT 设备上运行模型，以便实现设备端机器学习。

### 主要特性

- 支持多平台：涵盖 Android、iOS、嵌入式 Linux 设备、Windows、macOS 和 Linux 主机
- 支持多种语言：包括[Java](../api_reference/java_api_doc)、[Python](../api_reference/python_api_doc)、[C++](../api_reference/cxx_api_doc)
- 轻量化和高性能：针对移动端设备的机器学习进行优化，压缩模型和二进制文件体积，高效推理，降低内存消耗

## 开发工作流程
以下介绍了该工作流程的每一个步骤，并提供了进一步说明的链接：

### 1. 创建 Paddle Lite 模型

您可以通过以下方式生成 Paddle Lite 模型：

- 将 Paddle 模型转换为 Paddle Lite 模型：使用 [Paddle Lite opt 工具](../user_guides/model_optimize_tool) 将 Paddle 模型转换为 Paddle Lite 模型。在转换过程中，您可以应用量化等优化措施，以缩减模型大小和缩短延时，并最大限度降低或完全避免准确率损失。

### 2. 运行推断

推断是指在设备上执行 Paddle Lite 模型，以便根据输入数据进行预测的过程。您可以通过以下方式运行推断：

- 使用 Paddle Lite API，在多个平台和语言中均受支持（如 [Java](../user_guides/java_demo)、[C++](../user_guides/cpp_demo)、[Python](../user_guides/python_demo)）
  - 配置参数（`MobileConfig`），设置模型来源等
  - 创建推理器（Predictor），调用 `CreatePaddlePredictor` 接口即可创建
  - 设置模型输入，通过 `predictor->GetInput(i)` 获取输入变量，并为其指定大小和数值
  - 执行预测，只需要调用 `predictor->Run()`
  - 获得输出，使用 `predictor->GetOutput(i)` 获取输出变量，并通过 `data<T>` 取得输出值

在有 GPU 的设备上，您可以使用 [Paddle Lite 的 OpenCL 后端](../demo_guides/opencl)加速来提升性能。

## 开始使用

根据目标设备，您可以参阅以下指南：

- Android：请浏览 [Android apps](../demo_guides/android_app_demo)
- iOS：请浏览 [iOS apps](../demo_guides/ios_app_demo)
- 嵌入式 Linux：请浏览 [Linux apps](../demo_guides/linux_arm_demo)
- windows、macOS、Linux 等 x86 架构的 CPU 主机： 请浏览 [Paddle Lite 使用 X86 预测部署](../demo_guides/x86)
- AI 加速芯片
  - 华为麒麟 NPU ：请浏览 [Paddle Lite 使用华为麒麟 NPU 预测部署](../demo_guides/huawei_kirin_npu)
  - 华为昇腾 NPU ：请浏览 [Paddle Lite 使用华为昇腾 NPU 预测部署](../demo_guides/huawei_ascend_npu)
  - 昆仑芯 XPU ：请浏览 [Paddle Lite 使用昆仑芯 XPU 预测部署](../demo_guides/kunlunxin_xpu)
  - 昆仑芯 XTCL ：请浏览 [Paddle Lite 使用昆仑芯 XTCL 预测部署](../demo_guides/kunlunxin_xtcl)
  - 联发科 APU ：请浏览 [Paddle Lite 使用联发科 APU 预测部署](../demo_guides/mediatek_apu)
  - 颖脉 NNA ：请浏览 [Paddle Lite 使用颖脉 NNA 预测部署](../demo_guides/imagination_nna)
  - 高通 QNN ：请浏览 [Paddle Lite 使用高通 QNN 预测部署](../demo_guides/qualcomm_qnn)
  - 寒武纪 MLU ：请浏览 [Paddle Lite 使用寒武纪 MLU 预测部署](../demo_guides/cambricon_mlu)
  - 亿智 NPU ：请浏览 [Paddle Lite 使用亿智 NPU 预测部署](../demo_guides/eeasytech_npu)
  - Intel OpenVINO ：请浏览 [Paddle Lite 使用Intel OpenVINO 预测部署](../demo_guides/intel_openvino)
  - 安卓 NNAPI ：请浏览 [Paddle Lite 使用 Android NNAPI 预测部署](../demo_guides/android_nnapi)
  - 芯原 NPU：请浏览 [Paddle Lite 使用 芯原 TIM-VX 预测部署](../demo_guides/verisilicon_timvx)

## 技术路线

- [Paddle Lite RoadMap](roadmap)
