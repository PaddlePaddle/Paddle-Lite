# 编译选项说明


| 编译选项 |  说明  | 默认值 |
| :-- |  :-- |--: |
| LITE_WITH_LOG |  是否输出日志信息 | ON |
| LITE_WITH_EXCEPTION | 是否在错误发生时抛出异常 | OFF |
| LITE_WITH_TRAIN |  打开[模型训练功能](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/cpp_train_demo.html)，支持移动端模型训练 | OFF |
| LITE_BUILD_EXTRA |  编译[全量预测库](https://paddle-lite.readthedocs.io/zh/develop/source_compile/library.html)，包含更多算子和模型支持 | OFF |
| LITE_BUILD_TAILOR | 编译时[根据模型裁剪预测库](https://paddle-lite.readthedocs.io/zh/develop/source_compile/library_tailoring.html)，缩小预测库大小 | OFF |
| WITH_SYSTEM_BLAS |  编译时强制使用 reference BLAS |  OFF |

### 轻量级编译选项

适用于移动端编译，或者对预测库大小有要求的运行环境：

| 编译选项 |  说明  | 默认值 |
| :-- |  :-- | --: |
| LITE_ON_TINY_PUBLISH |  编译移动端部署库，无第三方库依赖 | OFF |

### 全功能编译选项

适用于服务端编译，或者对预测库大小没有要求的运行环境：

| 编译选项 |  说明  | 默认值 |
| :-- |  :-- | --: |
| LITE_WITH_PROFILE |  编译[性能 Profiler 工具](https://paddle-lite.readthedocs.io/zh/develop/user_guides/profiler.html)，用于 kernel 耗时统计 | OFF |
| LITE_WITH_PRECISION_PROFILE |  编译[精度 Profiler 工具](https://paddle-lite.readthedocs.io/zh/develop/user_guides/profiler.html)，用于 kernel 精度分析 | OFF |
| WITH_TESTING |  编译 Lite 单测模块 | OFF |

## 部分平台相关编译选项

| 编译选项 |  说明  | 适用平台 | 默认值 |
| :-- |  :-- | --: | --: |
| LITE_WITH_ARM |  编译支持 Andriod 或 ARMLinux 平台预测库 | Andriod / ARMLinux | OFF |
| LITE_WITH_JAVA |  编译支持 [Java API](https://paddle-lite.readthedocs.io/zh/develop/api_reference/java_api_doc.html) 的预测库 | Andriod / ARMLinux | OFF |
| LITE_WITH_ARM_CLANG | 使用 clang 编译 ARM 平台预测库 | Andriod / ARMLinux |OFF |
| WITH_ARM_DOTPROD |  编译 ARM 点积指令优化的预测库 | Andriod / ARMLinux |ON |
| LITE_WITH_CV |  编译 [CV 图像加速库](https://paddle-lite.readthedocs.io/zh/develop/api_reference/cv.html) | Andirod / ARMLinux |OFF |
| ANDROID_API_LEVEL | 设置安卓 API LEVEL | Android | Default，即 ARMv7 下为16，ARMv8 下为21 |
| LITE_WITH_OPENMP |  编译时打开 OpenMP | ARMLinux / X86 | ON |
| LITE_WITH_X86 |  编译[ X86 平台](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/x86.html)预测库 | X86 | ON |
| WITH_AVX |  编译有 AVX 指令优化的预测库 | X86 |ON IF ${AVX_FOUND} |
| WITH_MKL | 编译有 Intel MKL 支持的预测库 | X86 |ON IF ${AVX_FOUND} |
| LITE_ON_MODEL_OPTIMIZE_TOOL |  编译[模型优化工具 opt](https://paddle-lite.readthedocs.io/zh/develop/user_guides/model_optimize_tool.html) | X86 |OFF|
| LITE_WITH_PYTHON |  编译支持 [Python API](https://paddle-lite.readthedocs.io/zh/develop/api_reference/python_api_doc.html) 的预测库 | X86 / CUDA |OFF |
| LITE_WITH_OPENCL |  编译 [OpenCL 平台](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/opencl.html)预测库 | OpenCL | OFF |
| LITE_WITH_XPU |  编译[昆仑芯 XPU 平台](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/kunlunxin_xpu.html)预测库 | XPU |OFF |
