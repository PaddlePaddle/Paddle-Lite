# 编译选项说明


| 编译选项 |  说明  | 默认值 |
| :-- |  :-- |--: |
| LITE_WITH_LOG |  是否输出日志信息 | ON |
| LITE_WITH_EXCEPTION | 是否在错误发生时抛出异常 | OFF |
| LITE_WITH_TRAIN |  打开[模型训练功能](../demo_guides/cpp_train_demo.html)，支持移动端模型训练 | OFF |
| LITE_BUILD_EXTRA |  编译[全量预测库](library.html)，包含更多算子和模型支持 | OFF |
| LITE_BUILD_TAILOR | 编译时[根据模型裁剪预测库](library_tailoring.html)，缩小预测库大小 | OFF |
| WITH_SYSTEM_BLAS |  编译时强制使用 reference BLAS |  OFF |

### 轻量级编译选项

适用于移动端编译，或者对预测库大小有要求的运行环境：

| 编译选项 |  说明  | 默认值 |
| :-- |  :-- | --: |
| LITE_WITH_LIGHT_WEIGHT_FRAMEWORK | 编译移动端轻量级预测框架 | OFF |
| LITE_ON_TINY_PUBLISH |  编译移动端部署库，无第三方库依赖 | OFF |

### 全功能编译选项

适用于服务端编译，或者对预测库大小没有要求的运行环境：

| 编译选项 |  说明  | 默认值 |
| :-- |  :-- | --: |
| LITE_WITH_PROFILE |  编译 [Profiler 工具](../user_guides/debug.html)，用于 CPU 上 kernel 耗时统计 | OFF |
| LITE_WITH_PRECISION_PROFILE |  开启 Profiler 工具的模型精度分析功能 | OFF |
| WITH_TESTING |  编译 Lite 单测模块 | OFF |

## 部分平台相关编译选项

| 编译选项 |  说明  | 适用平台 | 默认值 |
| :-- |  :-- | --: | --: |
| LITE_WITH_ARM |  编译支持 Andriod 或 ARMLinux 平台预测库 | Andriod / ARMLinux | OFF |
| LITE_WITH_JAVA |  编译支持 [Java API](../api_reference/java_api_doc.html)的预测库 | Andriod / ARMLinux | OFF |
| LITE_WITH_ARM_CLANG | 使用 clang 编译 ARM 平台预测库 | Andriod / ARMLinux |OFF |
| WITH_ARM_DOTPROD |  编译 ARM 点积指令优化的预测库 | Andriod / ARMLinux |ON |
| LITE_WITH_CV |  编译 [CV 图像加速库](../api_reference/cv.html) | Andirod / ARMLinux |OFF |
| ANDROID_API_LEVEL | 设置安卓 API LEVEL | Android | Default，即 ARMv7 下为16，ARMv8 下为21 |
| LITE_WITH_OPENMP |  编译时打开 OpenMP | ARMLinux / X86 | ON |
| LITE_WITH_X86 |  编译[ X86 平台](../demo_guides/x86.html)预测库 | X86 | ON |
| WITH_AVX |  编译有 AVX 指令优化的预测库 | X86 |ON IF ${AVX_FOUND} |
| WITH_MKL | 编译有 Intel MKL 支持的预测库 | X86 |ON IF ${AVX_FOUND} |
| LITE_ON_MODEL_OPTIMIZE_TOOL |  编译[模型优化工具 opt](../user_guides/model_optimize_tool.html) | X86 |OFF|
| LITE_WITH_PYTHON |  编译支持 [Python API](../api_reference/python_api_doc.html)的预测库 | X86 / CUDA |OFF |
| LITE_WITH_OPENCL |  编译 [OpenCL 平台](../demo_guides/opencl.html)预测库 | OpenCL | OFF |
| LITE_WITH_NPU |  编译 [华为 NPU 平台](../demo_guides/huawei_kirin_npu.html)预测库 | NPU | OFF |
| LITE_WITH_RKNPU |  编译[瑞芯微 NPU 平台](../demo_guides/rockchip_npu.html)预测库 | RKNPU | OFF |
| LITE_WITH_XPU |  编译[百度 XPU 平台](../demo_guides/baidu_xpu.html)预测库 | XPU |OFF |
| LITE_WITH_XTCL | 通过 XTCL 方式支持百度 XPU，默认 Kernel 方式 | XPU |OFF IF LITE_WITH_XPU |
| LITE_WITH_APU | 编译[联发科 APU 平台](../demo_guides/mediatek_apu.html)预测库 | APU |OFF |

## 具体平台编译选项

- [使用 MacOS 环境构建 / 目标硬件 OS 为 iOS](./compile_ios.md)
- [使用 MacOS 环境构建 / 目标硬件 OS 为 Android](./compile_macos_android.rst)
- [使用 x86 Windows 环境构建 / 目标硬件 OS 为 x86 Windows](./compile_windows.rst)
- [使用 x86 Linux 环境构建 / 目标终端为 OS 为 Android](./compile_android.rst)
- [使用 ARM Linux 环境构建 / 目标终端为 OS 为 ARM Linux](./arm_host_compile_arm_linux.rst)
- [使用 x86 Linux 环境构建 / 目标硬件 OS 为 ARM Linux](./x86_host_compile_arm_linux.rst)
- [使用 x86 Linux 环境构建 / 目标硬件 OS 为 x86 Linux](./x86_host_compile_x86_linux.rst)
- [使用 MacOS x86 芯片环境构建 / 目标硬件 OS 为 MacOS](./compile_x86macos.rst)
- [使用 MacOS M1 芯片环境构建 / 目标硬件 OS 为 MacOS](./compile_armmacos.md)
