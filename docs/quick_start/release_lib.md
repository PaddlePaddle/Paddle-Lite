
# Paddle Lite 预编译库下载

## 编译版本介绍

- arch=`armv7/armv7hf/armv8/x86`        目标设备的 CPU 架构，可选择包括 `armv7`、`armv7hf`、`armv8` 和 `x86` 等
- os=`Android/IOS/Linux/MacOS/Windows`  目标设备的操作系统，可选择包括 `Android`、`IOS`、`Linux`、`MacOS` 和 `Windows` 等
- toolchain=`gcc/clang`                 源码编译时的编译器，可选择包括 `gcc` 和 `clang` 等
- android_stl=`c++_static/c++_shared`   预测库采用的 Android STL 库的种类，可选择包括 `c++_static` (静态链接)和 `c++_shared` (动态链接)
- with_extra=`ON/OFF`                   是否编译全量 OP，OFF 时只编译 CV 相关基础 OP，[参数详情](../source_compile/compile_options)
- with_cv=`ON/OFF`                      是否编译 CV 相关 API
- with_log=`ON/OFF`                     预编译库是否带有日志打印
- python_version=`2.7/3.5/3.6/3.7`      python 版本，可选择包括 `2.7`、`3.5`、`3.6` 和 `3.7` 等


## Android

|Arch |toolchain |android_stl |with_extra |with_cv |下载链接 |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|armv7 |clang |c++_shared |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.clang.c++_shared.tar.gz)                      |
|armv7 |clang |c++_shared |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.clang.c++_shared.with_cv.tar.gz)              |
|armv7 |clang |c++_shared |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.clang.c++_shared.with_extra.tar.gz)           |
|armv7 |clang |c++_shared |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.clang.c++_shared.with_extra.with_cv.tar.gz)   |
|armv7 |clang |c++_static |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.clang.c++_static.tar.gz)                      |
|armv7 |clang |c++_static |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.clang.c++_static.with_cv.tar.gz)              |
|armv7 |clang |c++_static |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.clang.c++_static.with_extra.tar.gz)           |
|armv7 |clang |c++_static |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.clang.c++_static.with_extra.with_cv.tar.gz)   |
|armv7 |gcc   |c++_shared |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.gcc.c++_shared.tar.gz)                        |
|armv7 |gcc   |c++_shared |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.gcc.c++_shared.with_cv.tar.gz)                |
|armv7 |gcc   |c++_shared |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.tar.gz)             |
|armv7 |gcc   |c++_shared |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.with_cv.tar.gz)     |
|armv7 |gcc   |c++_static |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.gcc.c++_static.tar.gz)                        |
|armv7 |gcc   |c++_static |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.gcc.c++_static.with_cv.tar.gz)                |
|armv7 |gcc   |c++_static |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.tar.gz)             |
|armv7 |gcc   |c++_static |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.with_cv.tar.gz)     |
|armv8 |clang |c++_shared |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.clang.c++_shared.tar.gz)                      |
|armv8 |clang |c++_shared |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.clang.c++_shared.with_cv.tar.gz)              |
|armv8 |clang |c++_shared |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.clang.c++_shared.with_extra.tar.gz)           |
|armv8 |clang |c++_shared |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.clang.c++_shared.with_extra.with_cv.tar.gz)   |
|armv8 |clang |c++_static |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.clang.c++_static.tar.gz)                      |
|armv8 |clang |c++_static |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.clang.c++_static.with_cv.tar.gz)              |
|armv8 |clang |c++_static |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.clang.c++_static.with_extra.tar.gz)           |
|armv8 |clang |c++_static |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv.tar.gz)   |
|armv8 |gcc   |c++_shared |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.gcc.c++_shared.tar.gz)                        |
|armv8 |gcc   |c++_shared |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.gcc.c++_shared.with_cv.tar.gz)                |
|armv8 |gcc   |c++_shared |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.tar.gz)             |
|armv8 |gcc   |c++_shared |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.with_cv.tar.gz)     |
|armv8 |gcc   |c++_static |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.gcc.c++_static.tar.gz)                        |
|armv8 |gcc   |c++_static |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.gcc.c++_static.with_cv.tar.gz)                |
|armv8 |gcc   |c++_static |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.tar.gz)             |
|armv8 |gcc   |c++_static |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.with_cv.tar.gz)     |


## IOS

|Arch |with_cv |with_extra |with_log |下载链接 |
|:-----:|:-----:|:-----:|:-----:|:-----:|
|armv7 |OFF |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.tiny_publish.tar.gz)                                |
|armv7 |OFF |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.with_log.tiny_publish.tar.gz)                       |
|armv7 |OFF |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.with_extra.tiny_publish.tar.gz)                     |
|armv7 |OFF |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.with_extra.with_log.tiny_publish.tar.gz)            |
|armv7 |ON  |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.with_cv.tiny_publish.tar.gz)                        |
|armv7 |ON  |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.with_cv.with_log.tiny_publish.tar.gz)               |
|armv7 |ON  |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.with_cv.with_extra.tiny_publish.tar.gz)             |
|armv7 |ON  |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.with_cv.with_extra.with_log.tiny_publish.tar.gz)    |
|armv8 |OFF |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv7.tiny_publish.tar.gz)                                |
|armv8 |OFF |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv8.with_extra.tiny_publish.tar.gz)                     |
|armv8 |OFF |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv8.with_extra.tiny_publish.tar.gz)                     |
|armv8 |OFF |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv8.with_extra.with_log.tiny_publish.tar.gz)            |
|armv8 |ON  |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv8.with_cv.tiny_publish.tar.gz)                        |
|armv8 |ON  |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv8.with_cv.with_log.tiny_publish.tar.gz)               |
|armv8 |ON  |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv8.with_cv.with_extra.tiny_publish.tar.gz)             |
|armv8 |ON  |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.ios.armv8.with_cv.with_extra.with_log.tiny_publish.tar.gz)    |


## Linux (ARM)

|Arch |with_extra |with_cv |下载链接 |适用的设备 |适用的操作系统 |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|armv7hf |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armlinux.armv7hf.gcc.tar.gz)                     |Raspberry Pi 3 Model B |Raspbian OS     |
|armv7hf |OFF |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armlinux.armv7hf.gcc.with_cv.tar.gz)             |Raspberry Pi 3 Model B |Raspbian OS     |
|armv7hf |ON  |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armlinux.armv7hf.gcc.with_extra.tar.gz)          |Raspberry Pi 3 Model B |Raspbian OS     |
|armv7hf |ON  |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armlinux.armv7hf.gcc.with_extra.with_cv.tar.gz)  |Raspberry Pi 3 Model B |Raspbian OS     |
|armv8   |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armlinux.armv8.gcc.tar.gz)                       |RK3399                 |firefly (Linux) |
|armv8   |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armlinux.armv8.gcc.with_cv.tar.gz)               |RK3399                 |firefly (Linux) |
|armv8   |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armlinux.armv8.gcc.with_extra.tar.gz)            |RK3399                 |firefly (Linux) |
|armv8   |OFF |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz)    |RK3399                 |firefly (Linux) |


## Linux (X86)

|with_log |下载链接 |适用的操作系统 |
|:-----:|:-----:|:-----:|
|OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.x86.linux.tar.gz)             |Ubuntu (Linux) |
|ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.x86.linux.with_log.tar.gz)    |Ubuntu (Linux) |


## MacOS

|Arch |with_log |下载链接 |
|:-----:|:-----:|:-----:|
|x86 |OFF |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.x86.macOS.tar.gz)          |
|x86 |ON  |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.x86.macOS.with_log.tar.gz) |


## Windows

|Arch |python_version |下载链接 |
|:-----:|:-----:|:-----:|
|x86 |2.7 |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.win.x86.MSVC.C++_static.py27.full_publish.zip) |
|x86 |3.5 |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.win.x86.MSVC.C++_static.py35.full_publish.zip) |
|x86 |3.6 |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.win.x86.MSVC.C++_static.py36.full_publish.zip) |
|x86 |3.7 |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.win.x86.MSVC.C++_static.py37.full_publish.zip) |
|x86 |3.9 |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.win.x86.MSVC.C++_static.py39.full_publish.zip) |


## Opencl

|Arch |下载链接 |
|:-----:|:-----:|
|armv7 |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armv7.clang.with_exception.with_extra.with_cv.opencl.tar.gz) |
|armv8 |[v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/inference_lite_lib.armv8.clang.with_exception.with_extra.with_cv.opencl.tar.gz) |


## 昆仑芯 XPU

|Arch |下载链接 |适用的操作系统 |
|:-----:|:-----:|:-----:|
|x86   |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.centos.x86.kunlunxin_xpu.tar.gz) |CentOS 6.3 |
|x86   |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.ubuntu.x86.kunlunxin_xpu.tar.gz) |Ubuntu     |
|armv8 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.ky10.armv8.kunlunxin_xpu.tar.gz) |银河麒麟v10 |


## 华为昇腾 NPU

|Arch |下载链接 |适用的操作系统 |
|:-----:|:-----:|:-----:|
|x86   |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.ubuntu.x86.huawei_ascend_npu.tar.gz) |Ubuntu    |
|armv8 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.ky10.armv8.huawei_ascend_npu.tar.gz) |银河麒麟v10 |


## 华为麒麟 NPU

|Arch |下载链接 |
|:-----:|:-----:|
|armv7 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.android.armv7.huawei_kirin_npu.with_cv.with_extra.with_log.tiny_publish.tar.gz) |
|armv8 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.android.armv8.huawei_kirin_npu.with_cv.with_extra.with_log.tiny_publish.tar.gz) |


## 瑞芯微 NPU

|Arch |下载链接 |
|:-----:|:-----:|
|armv7hf |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.armlinux.armv7hf.rockchip_npu.with_extra.with_log.tiny_publish.tar.gz) |
|armv8   |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.armlinux.armv8.rockchip_npu.with_extra.with_log.tiny_publish.tar.gz)   |


## 晶晨 NPU

|Arch |下载链接 |适用的操作系统 |
|:-----:|:-----:|:-----:|
|armv7 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.android.armv7.amlogic_npu.with_extra.with_log.tiny_publish.tar.gz)  |Android |
|armv8 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.armlinux.armv8.amlogic_npu.with_extra.with_log.tiny_publish.tar.gz) |Linux   |


## 联发科 APU

|Arch |下载链接 |
|:-----:|:-----:|
|armv7 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.android.armv7.mediatek_apu.with_extra.with_log.tiny_publish.tar.gz) |
|armv8 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.android.armv8.mediatek_apu.with_extra.with_log.tiny_publish.tar.gz) |


## 颖脉 NNA

|Arch |下载链接 |
|:-----:|:-----:|
|armv8 |[v2.11-rc](https://paddlelite-data.bj.bcebos.com/release/v2.11-rc/inference_lite_lib.armlinux.armv8.imagination_nna.with_extra.with_log.tiny_publish.tar.gz) |


## opt 工具

|适用的操作系统 |下载链接 |
|:-----:|:-----:|
|Linux | [v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/opt_linux) |
|MacOS | [v2.11-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11-rc/opt_mac)   |


## 安装 Paddle Lite python 库方法

- 支持平台：Windows10、Ubuntu、Mac
- python version: 2.7、3.5、3.6、3.7
```
# 当前最新版本是 2.11-rc
python -m pip install paddlelite==2.11-rc
```
