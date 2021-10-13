
# Paddle Lite 预编译库下载

## 编译版本介绍

- arch=`armv7/armv7hf/armv8/x86`        Host端的CPU架构，可选择包括`armv7`、`armv7hf`、`armv8`和`x86`等
- os=`Android/iOS/Linux/macOS/Windows`  操作系统，可选择包括`Android`、`iOS`、`linux`、`macOS`和`Windows`等
- toolchain=`gcc/clang`                 源码编译时的编译器，可选择包括`gcc`和`clang`等
- android_stl=`c++_static/c++_shared`   PaddleLite预测库链接STL库的方式，可选择包括`c++_static`(静态链接)和`c++_shared`(动态链接)
- with_extra=`ON/OFF`                   是否编译全量OP，OFF时只编译CV相关基础OP，[参数详情](../source_compile/library)
- with_cv=`ON/OFF`                      是否编译CV相关API
- with_log=`ON/OFF`                     预编译库是否带有日志打印
- python_version=`2.7/3.5/3.6/3.7`      python版本，可选择包括`2.7`、`3.5`、`3.6`和`3.7`等


## Android

|arch |toolchain |arm_stl |with_extra |with_cv |dowload |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|armv7 |clang |c++_shared |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_shared.tar.gz)                      |
|armv7 |clang |c++_shared |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_shared.with_cv.tar.gz)              |
|armv7 |clang |c++_shared |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_shared.with_extra.tar.gz)           |
|armv7 |clang |c++_shared |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_shared.with_extra.with_cv.tar.gz)   |
|armv7 |clang |c++_static |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_static.tar.gz)                      |
|armv7 |clang |c++_static |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_static.with_cv.tar.gz)              |
|armv7 |clang |c++_static |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_static.with_extra.tar.gz)           |
|armv7 |clang |c++_static |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_static.with_extra.with_cv.tar.gz)   |
|armv7 |gcc   |c++_shared |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.gcc.c++_shared.tar.gz)                        |
|armv7 |gcc   |c++_shared |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.gcc.c++_shared.with_cv.tar.gz)                |
|armv7 |gcc   |c++_shared |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.tar.gz)             |
|armv7 |gcc   |c++_shared |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.with_cv.tar.gz)     |
|armv7 |gcc   |c++_static |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.gcc.c++_static.tar.gz)                        |
|armv7 |gcc   |c++_static |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.gcc.c++_static.with_cv.tar.gz)                |
|armv7 |gcc   |c++_static |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.tar.gz)             |
|armv7 |gcc   |c++_static |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.with_cv.tar.gz)     |
|armv8 |clang |c++_shared |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_shared.tar.gz)                      |
|armv8 |clang |c++_shared |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_shared.with_cv.tar.gz)              |
|armv8 |clang |c++_shared |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_shared.with_extra.tar.gz)           |
|armv8 |clang |c++_shared |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_shared.with_extra.with_cv.tar.gz)   |
|armv8 |clang |c++_static |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_static.tar.gz)                      |
|armv8 |clang |c++_static |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_static.with_cv.tar.gz)              |
|armv8 |clang |c++_static |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_static.with_extra.tar.gz)           |
|armv8 |clang |c++_static |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv.tar.gz)   |
|armv8 |gcc   |c++_shared |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.gcc.c++_shared.tar.gz)                        |
|armv8 |gcc   |c++_shared |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.gcc.c++_shared.with_cv.tar.gz)                |
|armv8 |gcc   |c++_shared |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.tar.gz)             |
|armv8 |gcc   |c++_shared |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.with_cv.tar.gz)     |
|armv8 |gcc   |c++_static |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.gcc.c++_static.tar.gz)                        |
|armv8 |gcc   |c++_static |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.gcc.c++_static.with_cv.tar.gz)                |
|armv8 |gcc   |c++_static |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.tar.gz)             |
|armv8 |gcc   |c++_static |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.with_cv.tar.gz)     |


## iOS

|arch |with_cv |with_extra |with_log |dowload |
|:-----:|:-----:|:-----:|:-----:|:-----:|
|armv7 |OFF |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.tiny_publish.tar.gz)                                |
|armv7 |OFF |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_log.tiny_publish.tar.gz)                       |
|armv7 |OFF |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_extra.tiny_publish.tar.gz)                     |
|armv7 |OFF |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_extra.with_log.tiny_publish.tar.gz)            |
|armv7 |ON  |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_cv.tiny_publish.tar.gz)                        |
|armv7 |ON  |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_cv.with_log.tiny_publish.tar.gz)               |
|armv7 |ON  |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_cv.with_extra.tiny_publish.tar.gz)             |
|armv7 |ON  |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_cv.with_extra.with_log.tiny_publish.tar.gz)    |
|armv8 |OFF |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.tiny_publish.tar.gz)                                |
|armv8 |OFF |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_extra.tiny_publish.tar.gz)                     |
|armv8 |OFF |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_extra.tiny_publish.tar.gz)                     |
|armv8 |OFF |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_extra.with_log.tiny_publish.tar.gz)            |
|armv8 |ON  |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_cv.tiny_publish.tar.gz)                        |
|armv8 |ON  |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_cv.with_log.tiny_publish.tar.gz)               |
|armv8 |ON  |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_cv.with_extra.tiny_publish.tar.gz)             |
|armv8 |ON  |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_cv.with_extra.with_log.tiny_publish.tar.gz)    |


## Linux

|Arch |with_extra |with_cv |dowload |device |os |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|armv7hf |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armlinux.armv7hf.gcc.tar.gz)                     |Raspberry Pi 3 Model B |Raspbian OS     |
|armv7hf |OFF |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armlinux.armv7hf.gcc.with_cv.tar.gz)             |Raspberry Pi 3 Model B |Raspbian OS     |
|armv7hf |ON  |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armlinux.armv7hf.gcc.with_extra.tar.gz)          |Raspberry Pi 3 Model B |Raspbian OS     |
|armv7hf |ON  |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armlinux.armv7hf.gcc.with_extra.with_cv.tar.gz)  |Raspberry Pi 3 Model B |Raspbian OS     |
|armv8   |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armlinux.armv8.gcc.tar.gz)                       |RK3399                 |firefly (Linux) |
|armv8   |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armlinux.armv8.gcc.with_cv.tar.gz)               |RK3399                 |firefly (Linux) |
|armv8   |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armlinux.armv8.gcc.with_extra.tar.gz)            |RK3399                 |firefly (Linux) |
|armv8   |OFF |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz)    |RK3399                 |firefly (Linux) |


|Arch |with_log |dowload |os |
|:-----:|:-----:|:-----:|:-----:|
|x86 |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.x86.linux.tar.gz)             |Ubuntu (Linux) |
|x86 |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.x86.linux.with_log.tar.gz)    |Ubuntu (Linux) |


## macOS

|Arch |with_log |dowload |
|:-----:|:-----:|:-----:|
|x86 |OFF |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.x86.macOS.tar.gz)          |
|x86 |ON  |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.x86.macOS.with_log.tar.gz) |


## windows

|Arch |python_version |dowload |
|:-----:|:-----:|:-----:|
|x86 |2.7 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.win.x86.MSVC.C++_static.py27.full_publish.zip) |
|x86 |3.5 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.win.x86.MSVC.C++_static.py35.full_publish.zip) |
|x86 |3.6 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.win.x86.MSVC.C++_static.py36.full_publish.zip) |
|x86 |3.7 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.win.x86.MSVC.C++_static.py37.full_publish.zip) |


## Opencl

|Arch |dowload |
|:-----:|:-----:|
|armv7 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armv7.clang.with_exception.with_extra.with_cv.opencl.tar.gz) |
|armv8 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.armv8.clang.with_exception.with_extra.with_cv.opencl.tar.gz) |


## 百度XPU

|Arch |dowload |os |
|:-----:|:-----:|:-----:|
|x86   |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.baidu_xpu.x86.centos.tar.gz) |CentOS 6.3 |
|x86   |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.baidu_xpu.x86.ubuntu.tar.gz) |Ubuntu     |
|armv8 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.baidu_xpu.armv8.ky10.tar.gz) |银河麒麟v10 |


## 华为昇腾NPU

|Arch |dowload |os |
|:-----:|:-----:|:-----:|
|x86   |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.huawei_ascend_npu.x86.ubuntu.tar.gz) |Ubuntu    |
|armv8 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.huawei_ascend_npu.armv8.ky10.tar.gz) |银河麒麟v10 |


## 华为麒麟NPU

|Arch |dowload |
|:-----:|:-----:|
|armv7 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.huawei_kirin_npu.with_cv.with_extra.with_log.tiny_publish.tar.gz) |
|armv8 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.huawei_kirin_npu.with_cv.with_extra.with_log.tiny_publish.tar.gz) |


## 瑞芯微NPU

|Arch |dowload |
|:-----:|:-----:|
|armv7hf |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7hf.rockchip_npu.with_extra.with_log.tiny_publish.tar.gz) |
|armv8   |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.rockchip_npu.with_extra.with_log.tiny_publish.tar.gz)   |

## 联发科APU

|Arch |dowload |
|:-----:|:-----:|
|armv7 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.mediatek_apu.with_extra.with_log.tiny_publish.tar.gz) |
|armv8 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.mediatek_apu.with_extra.with_log.tiny_publish.tar.gz) |

## 颖脉NNA

|Arch |dowload |
|:-----:|:-----:|
|armv8 |[v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.imagination_nna.with_extra.with_log.tiny_publish.tar.gz) |


## opt 工具

|os |dowload |
|:-----:|:-----:|
|Linux | [v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/opt_linux) |
|macOS | [v2.10](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/opt_mac)   |


## 安装Paddle-Lite python 库方法

- 支持平台： windows10、Ubuntu、Mac
- python version: 2.7、3.5、3.6、 3.7
```
# 当前最新版本是 2.10
python -m pip install paddlelite==2.10
```

## 对应源码编译方法

- [opt源码编译](../user_guides/model_optimize_tool.html#opt)
- [Android源码编译](../source_compile/compile_andriod)
- [iOS源码编译](../source_compile/compile_ios)
- [ArmLinux源码编译](../source_compile/compile_linux)
- [x86源码编译](../demo_guides/x86)
- [opencl源码编译](../demo_guides/opencl)
- [FPGA源码编译](../demo_guides/fpga)
- [华为NPU源码编译](../demo_guides/huawei_kirin_npu)
- [百度XPU源码编译](../demo_guides/baidu_xpu)
- [瑞芯微NPU源码编译](../demo_guides/rockchip_npu)
- [晶晨NPU源码编译](../demo_guides/amlogic_npu)
- [联发科APU源码编译](../demo_guides/mediatek_apu)
