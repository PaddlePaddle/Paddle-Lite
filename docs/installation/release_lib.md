
# 官方 release 预编译库

## 编译版本介绍

- ARM_Version=`armv7/armv8`               编译时选择的arm版本，v7或者v8
- build_extra=`ON/OFF`                                是否编译全量算子，OFF时只编译CV相关基础算子
- arm_stl=`c++_static/c++_shared`       选择动态或者静态链接NDK STL库
- target:  `tiny_publish/full_publish` 是否编译full_api预测库，`tiny_publish`时只编译light_api预测库


## Android

|ARM Version|build_extra|arm_stl|target|下载|编译命令|
|:-------:|:-----:|:-----:|:-----:|:-------:|:---------:|
|armv7|OFF|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_static.tiny_publish.tar.gz)|[编译命令](compile_command/android1.html)|
|armv7|OFF|c++_static|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_static.full_publish.tar.gz)|[编译命令](compile_command/android2.html)|
|armv7|OFF|c++_shared|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_shared.tiny_publish.tar.gz)|[编译命令](compile_command/android3.html)|
|armv7|OFF|c++_shared|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_shared.full_publish.tar.gz)|[编译命令](compile_command/android4.html)|
|armv7|ON|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.tiny_publish.tar.gz)|[编译命令](compile_command/android5.html)|
|armv7|ON|c++_static|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.full_publish.tar.gz)|[编译命令](compile_command/android6.html)|
|armv7|ON|c++_shared|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.tiny_publish.tar.gz)|[编译命令](compile_command/android7.html)|
|armv7|ON|c++_shared|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.full_publish.tar.gz)|[编译命令](compile_command/android8.html)|
|armv8|OFF|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.tiny_publish.tar.gz)|[编译命令](compile_command/android9.html)|
|armv8|OFF|c++_static|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.full_publish.tar.gz)|[编译命令](compile_command/android10.html)|
|armv8|OFF|c++_shared|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_shared.tiny_publish.tar.gz)|[编译命令](compile_command/android11.html)|
|armv8|OFF|c++_shared|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_shared.full_publish.tar.gz)|[编译命令](compile_command/android12.html)|
|armv8|ON|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.tiny_publish.tar.gz)|[编译命令](compile_command/android13.html)|
|armv8|ON|c++_static|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.full_publish.tar.gz)|[编译命令](compile_command/android14.html)|
|armv8|ON|c++_shared|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.tiny_publish.tar.gz)|[编译命令](compile_command/android15.html)|
|armv8|ON|c++_shared|full_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.full_publish.tar.gz)|[编译命令](compile_command/android16.html)|
## iOS

|ARM Version|arm_os|with_extra|下载|编译命令|
|:-------:|:-----:|:-----:|:-----:|:-------:|
|armv7|Ios|OFF|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios.armv7.tar.gz)|[编译命令](compile_command/ios1.html)|
|armv7|ios|ON|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios.armv7.with_extra.tar.gz)|[编译命令](compile_command/ios2.html)|
|armv8|Ios64|OFF|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios64.armv8.tar.gz)|[编译命令](compile_command/ios3.html)|
|armv8|Ios64|ON|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios64.armv8.with_extra.tar.gz)|[编译命令](compile_command/ios4.html)|

opt 工具

| 运行系统 |      下载       |编译命令|
| :---------: |  :--------------: |  :--------------: |
|    Linux    |  [release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/opt) |[编译命令](compile_command/opt.html)|
|    MacOs   |  [release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/opt_mac) |[编译命令](compile_command/opt.html)|



## 对应源码编译方法

- [opt源码编译](../user_guides/model_optimize_tool.html#opt)
- [Android源码编译](./source_compile.html#paddlelite)
- [iOS源码编译](./source_compile.html#paddlelite)
- [ArmLinux源码编译](./source_compile.html#paddlelite)
- [x86源码编译](../advanced_user_guides/x86)
- [opencl源码编译](../advanced_user_guides/opencl)
- [CUDA源码编译](../advanced_user_guides/cuda)
- [FPGA源码编译](../advanced_user_guides/fpga)
