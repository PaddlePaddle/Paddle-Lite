
# Lite 预编译库下载

## 编译版本介绍

- arch=`armv7/armv8`                                       arm版本，可选择armv7或者armv8
- arm_os=`android\ios\armlinux`    安装平台，支持的arm端移动平台包括 `ios`、`armlinux`和`android`
- toolchain=`gcc/clang`                                 源码编译时的编译器，默认为`gcc`编译器
- android_stl=`c++_static/c++_shared`     Lite预测库链接STL库的方式，支持静态或动态链接
- with_extra=`ON/OFF`                                     是否编译全量OP，OFF时只编译CV相关基础OP，[参数详情](../source_compile/library)
- with_cv=`ON/OFF`                                          是否编译编译Paddle-Lite CV 相关API


## Android（toolchain=gcc）

| Arch  |with_extra|arm_stl|with_cv|下载|
|:-------:|:-----:|:-----:|:-----:|:-------:|
|armv7|OFF|c++_shared|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.gcc.c++_shared.tar.gz)|
|armv7|OFF|c++_shared|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.gcc.c++_shared.with_cv.tar.gz)|
|armv7|ON|c++_shared|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.tar.gz)|
|armv7|ON|c++_shared|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.with_cv.tar.gz)|
|armv7|OFF|c++_static|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.gcc.c++_static.tar.gz)|
|armv7|OFF|c++_static|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.gcc.c++_static.with_cv.tar.gz)|
|armv7|ON|c++_static|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.tar.gz)|
|armv7|ON|c++_static|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.with_cv.tar.gz)|
|armv8|OFF|c++_shared|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_shared.tar.gz)|
|armv8|OFF|c++_shared|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_shared.with_cv.tar.gz)|
|armv8|ON|c++_shared|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.tar.gz)|
|armv8|ON|c++_shared|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.with_cv.tar.gz)|
|armv8|OFF|c++_static|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_static.tar.gz)|
|armv8|OFF|c++_static|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_static.with_cv.tar.gz)|
|armv8|ON|c++_static|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.tar.gz)|
|armv8|ON|c++_static|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.with_cv.tar.gz)|

## Android（toolchain=clang）

| Arch  |with_extra|arm_stl|with_cv|下载|
|:-------:|:-----:|:-----:|:-----:|:-------:|
|armv7|OFF|c++_shared|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.clang.c++_shared.tar.gz)|
|armv7|OFF|c++_shared|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.clang.c++_shared.with_cv.tar.gz)|
|armv7|ON|c++_shared|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.clang.c++_shared.with_extra.tar.gz)|
|armv7|ON|c++_shared|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.clang.c++_shared.with_extra.with_cv.tar.gz)|
|armv7|OFF|c++_static|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.clang.c++_static.tar.gz)|
|armv7|OFF|c++_static|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.clang.c++_static.with_cv.tar.gz)|
|armv7|ON|c++_static|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.clang.c++_static.with_extra.tar.gz)|
|armv7|ON|c++_static|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv7.clang.c++_static.with_extra.with_cv.tar.gz)|
|armv8|OFF|c++_shared|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.clang.c++_shared.tar.gz)|
|armv8|OFF|c++_shared|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.clang.c++_shared.with_cv.tar.gz)|
|armv8|ON|c++_shared|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.clang.c++_shared.with_extra.tar.gz)|
|armv8|ON|c++_shared|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.clang.c++_shared.with_extra.with_cv.tar.gz)|
|armv8|OFF|c++_static|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.clang.c++_static.tar.gz)|
|armv8|OFF|c++_static|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.clang.c++_static.with_cv.tar.gz)|
|armv8|ON|c++_static|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.clang.c++_static.with_extra.tar.gz)|
|armv8|ON|c++_static|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv.tar.gz)|

## iOS

|ARM Version|with_cv|with_extra|下载|
|:-------:|:-----:|:-----:|:-----:|
|armv7|OFF|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.ios.armv7.tiny_publish.tar.gz)|
|armv7|OFF|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.ios.armv7.with_cv.tiny_publish.tar.gz)|
|armv7|ON|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.ios.armv7.with_extra.tiny_publish.tar.gz)|
|armv7|ON|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.ios.armv7.with_cv.with_extra.tiny_publish.tar.gz)|
|armv8|OFF|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.ios.armv8.tiny_publish.tar.gz)|
|armv8|OFF|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.ios.armv8.with_cv.tiny_publish.tar.gz)|
|armv8|ON|OFF|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.ios.armv8.with_extra.tiny_publish.tar.gz)|
|armv8|ON|ON|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.ios.armv8.with_cv.with_extra.tiny_publish.tar.gz)|


## x86 Linux

|Operating System|下载|
|:-------:|:-----:|
|Ubuntu (Linux)|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.x86.linux.tar.gz)|

## macOS

|Operating System|下载|
|:-------:|:-----:|
|macOS|[v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.x86.macOS.tar.gz)|


## opt 工具

| 运行系统 |      下载       |
| :---------: |  :--------------: |
|    Linux    | [v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/opt_linux) |
|    macOS   | [v2.9](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/opt_mac) |

## 安装Paddle-Lite python 库方法

- 支持平台： windows10、Ubuntu、Mac
- python version: 2.7、3.5、3.6、 3.7
```
# 当前最新版本是 2.9
python -m pip install paddlelite==2.9
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
- [联发科APU源码编译](../demo_guides/mediatek_apu)
