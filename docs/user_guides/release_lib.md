
# 预编译库

## 编译版本介绍

- ARM_Version=`armv7/armv8`                        arm版本，可选择armv7或者armv8

- arm_os=`android\ios\ios64\armlinux`   安装平台，支持的arm端移动平台包括 `ios\ios64`、`armlinux`和`android`

- arm_lang=`gcc/clang`                                  源码编译时的编译器，默认为`gcc`编译器

- arm_stl=`c++_static/c++_shared`             Lite预测库链接STL库的方式，支持静态或动态链接

- build_extra=`ON/OFF`                                     是否编译全量OP，OFF时只编译CV相关基础OP，[参数详情](library)

-  `tiny_publish/full_publish`                   编译模式，`tiny_publish`编译移动端部署库、`full_publish`编译部署库的同时编译第三方依赖库


## Android

|ARM Version|build_extra|arm_stl|target|下载|
|:-------:|:-----:|:-----:|:-----:|:-------:|
|armv7|OFF|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_static.tiny_publish.tar.gz)|
|armv7|OFF|c++_static|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_static.full_publish.tar.gz)|
|armv7|OFF|c++_shared|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_shared.tiny_publish.tar.gz)|
|armv7|OFF|c++_shared|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_shared.full_publish.tar.gz)|
|armv7|ON|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.tiny_publish.tar.gz)|
|armv7|ON|c++_static|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.full_publish.tar.gz)|
|armv7|ON|c++_shared|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.tiny_publish.tar.gz)|
|armv7|ON|c++_shared|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.full_publish.tar.gz)|
|armv8|OFF|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.tiny_publish.tar.gz)|
|armv8|OFF|c++_static|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.full_publish.tar.gz)|
|armv8|OFF|c++_shared|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_shared.tiny_publish.tar.gz)|
|armv8|OFF|c++_shared|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_shared.full_publish.tar.gz)|
|armv8|ON|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.tiny_publish.tar.gz)|
|armv8|ON|c++_static|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.full_publish.tar.gz)|
|armv8|ON|c++_shared|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.tiny_publish.tar.gz)|
|armv8|ON|c++_shared|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.full_publish.tar.gz)|


## iOS

|ARM Version|arm_os|with_extra|下载|
|:-------:|:-----:|:-----:|:-----:|
|armv7|ios|OFF|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios.armv7.tar.gz)|
|armv7|ios|ON|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios.armv7.with_extra.tar.gz)|
|armv8|ios64|OFF|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios64.armv8.tar.gz)|
|armv8|ios64|ON|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios64.armv8.with_extra.tar.gz)|


## opt 工具

| 运行系统 |      下载       |
| :---------: |  :--------------: |
|    Linux    | [release/v2.3](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/opt) |
|    MacOs   | [release/v2.3](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/opt_mac) |



## 对应源码编译方法

- [opt源码编译](../user_guides/model_optimize_tool.html#opt)
- [Android源码编译](./source_compile.html#paddlelite)
- [iOS源码编译](./source_compile.html#paddlelite)
- [ArmLinux源码编译](./source_compile.html#paddlelite)
- [x86源码编译](../demo_guides/x86)
- [opencl源码编译](../demo_guides/opencl)
- [CUDA源码编译](../demo_guides/cuda)
- [FPGA源码编译](../demo_guides/fpga)
