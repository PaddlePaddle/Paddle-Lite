
# 概述

Paddle Lite 提供了Android/iOS/X86/MacOS平台的官方Release预测库下载，如果您使用的是这四个平台，我们优先推荐您直接下载 [Paddle Lite 预编译库](../quick_start/release_lib)。

您也可以根据目标平台选择对应的源码编译方法，Paddle Lite提供了源码编译脚本，位于`lite/tools/`文件夹下，只需要“编译准备环境”和“执行编译脚本”两个步骤即可一键编译得到目标平台的Paddle Lite预测库。

# 编译环境准备
Paddle Lite 已支持多种交叉编译。**Paddle Lite 官方建议使用 [Docker 开发环境]() 进行编译，以避免复杂繁琐的环境搭建过程。** 用户也可以遵循 Paddle Lite 官方提供的环境搭建指南，自行在宿主机器上搭建编译环境。
|No|宿主机器（侧体系结构/操作系统）|目标机器（体系结构/操作系统）| 编译环境搭建及编译指南 |
|---|---|---|---|
|1.|x86 Linux|x86 Linux| [点击进入]() |
|2.|x86 linux|Arm Linux| [点击进入]() |
|3.|x86 Linux|Arm Android|[点击进入]() |
|4.|arm Linux|Arm Linux| [点击进入]() |
|5.|x86 MacOS|x86 MacOS|[点击进入]() |
|6.|x86 MacOS|Arm Android|[点击进入]() |
|7.|x86 MacOS|Arm iOS|[点击进入]() |

# 执行编译脚本
Paddle Lite 针对不同的交叉编译任务，分别提供了完善的一键式编译脚本，位于`lite/tools/`文件夹下。执行编译脚本即可启动 Paddle Lite 的编译工程。
|No|目标机器|编译脚本|
|---|---|---|
|1.|Android设备|build_android.sh|
|2.|Linux设备|build_linux.sh|
|3.|iOS设备|build_ios.sh|
|4.|MacOS设备|build_macos.sh|
|5.|Windows设备|build_windows.bat|

编译脚本具体的参数配置，请参考`编译环境准备`中的`编译环境搭建及编译指南`。

# 源码编译已支持的后端平台
Paddle Lite 已支持包括 CPU、GPU 和 AI硬件 在内的多种设备进行预测部署，且均可进行源码编译：
|No|已支持的硬件/后端|指南|
|---|---|---|
|1.|Android 设备|[点击进入]()|
|2.|iOS 设备|[点击进入]()|
|3.|Arm Linux 设备|[点击进入]()|
|4.|x86 设备|[点击进入]()|
|5.|OpenCL|[点击进入]()|
|6.|FPGA|[点击进入]()|
|7.|Baidu XPU|[点击进入]()|
|8.|Huawei Kirin NPU|[点击进入]()|
|9.|Huawei Ascend NPU|[点击进入]()|
|10.|Rockchip NPU|[点击进入]()|
|11.|Mediatek APU|[点击进入]()|
|12.|Imagination NNA|[点击进入]()|
|13.|Amlogic NPU|[点击进入]()|
|14.|Bitmain Sophon|[点击进入]()|
