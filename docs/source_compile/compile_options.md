# 编译选项说明


| 编译选项 |  说明  | 默认值 |
| :-- |  :-- |--: |
| LITE_WITH_LOG |  是否输出日志信息 | ON |
| LITE_WITH_EXCEPTION | 是否在错误发生时抛出异常 | OFF |
| LITE_WITH_TRAIN |  打开[模型训练功能](../demo_guides/cpp_train_demo.html)，支持移动端模型训练 | OFF |
| LITE_BUILD_EXTRA |  编译[全量预测库](library.html)，包含更多算子和模型支持 | OFF |
| LITE_BUILD_TAILOR | 编译时[根据模型裁剪预测库](library_tailoring.html)，缩小预测库大小 | OFF |
| WITH_SYSTEM_BLAS |  编译时强制使用reference BLAS |  OFF |

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
| LITE_WITH_PROFILE |  编译[Profiler工具](../user_guides/debug.html)，用于CPU上kernel耗时统计 | OFF |
| LITE_WITH_PRECISION_PROFILE |  开启Profiler工具的模型精度分析功能 | OFF |
| WITH_TESTING |  编译Lite单测模块 | OFF |

## 平台相关编译选项

| 编译选项 |  说明  | 适用平台 | 默认值 |
| :-- |  :-- | --: | --: |
| LITE_WITH_ARM |  编译支持Andriod或ARMLinux平台预测库 | Andriod / ARMLinux | OFF |
| LITE_WITH_JAVA |  编译支持[Java API](../api_reference/java_api_doc.html)的预测库 | Andriod / ARMLinux | OFF |
| LITE_WITH_ARM_CLANG | 使用clang编译ARM平台预测库 | Andriod / ARMLinux |OFF |
| WITH_ARM_DOTPROD |  编译ARM点积指令优化的预测库 | Andriod / ARMLinux |ON |
| LITE_WITH_CV |  编译[CV图像加速库](../api_reference/cv.html) | Andirod / ARMLinux |OFF |
| LITE_WITH_OPENMP |  编译时打开OpenMP | ARMLinux / X86 | ON |
| LITE_WITH_X86 |  编译[X86平台](../demo_guides/x86.html)预测库 | X86 | ON |
| WITH_AVX |  编译有AVX指令优化的预测库 | X86 |ON IF ${AVX_FOUND} |
| WITH_MKL | 编译有Intel MKL支持的预测库 | X86 |ON IF ${AVX_FOUND} |
| LITE_ON_MODEL_OPTIMIZE_TOOL |  编译[模型优化工具opt](../user_guides/model_optimize_tool.html) | X86 |OFF|
| LITE_WITH_CUDA |  编译[CUDA平台](../demo_guides/cuda.html)预测库 | CUDA | OFF |
| WITH_DSO |  编译动态CUDA库 | CUDA | ON |
| LITE_WITH_STATIC_CUDA |   编译静态CUDA库 | CUDA |OFF |
| LITE_WITH_NVTX | 是否打开NVIDIA Tools Extension (NVTX) | CUDA |OFF |
| CUDA_WITH_FP16 |  编译CUDA FP16支持| CUDA |OFF |
| LITE_WITH_PYTHON |  编译支持[Python API](../api_reference/python_api_doc.html)的预测库 | X86 / CUDA |OFF |
| LITE_WITH_OPENCL |  编译[OpenCL平台](../demo_guides/opencl.html)预测库 | OpenCL | OFF |
| LITE_WITH_FPGA |  编译[FPGA平台](../demo_guides/fpga.html)预测库 | FPGA | OFF |
| LITE_WITH_NPU |  编译[华为NPU平台](../demo_guides/huawei_kirin_npu.html)预测库 | NPU | OFF |
| LITE_WITH_RKNPU |  编译[瑞芯微NPU平台](../demo_guides/rockchip_npu.html)预测库 | RKNPU | OFF |
| LITE_WITH_XPU |  编译[百度XPU平台](../demo_guides/baidu_xpu.html)预测库 | XPU |OFF |
| LITE_WITH_XTCL | 通过XTCL方式支持百度XPU，默认Kernel方式 | XPU |OFF IF LITE_WITH_XPU |
| LITE_WITH_APU | 编译[联发科APU平台](../demo_guides/mediatek_apu.html)预测库 | APU |OFF |
