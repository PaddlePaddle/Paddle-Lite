
# 官方 release 预编译库

## 编译版本介绍

- ARM_Version=`armv7/armv8`               编译时选择的arm版本，v7或者v8
- build_extra=`ON/OFF`                                是否编译全量算子，OFF时只编译CV相关基础算子
- arm_stl=`c++_static/c++_shared`       选择动态或者静态链接NDK STL库
- target:  `tiny_publish/full_publish` 是否编译full_api预测库，`tiny_publish`时只编译light_api预测库


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
|armv7|Ios|OFF|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios.armv7.tar.gz)|
|armv7|ios|ON|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios.armv7.with_extra.tar.gz)|
|armv8|Ios64|OFF|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios64.armv8.tar.gz)|
|armv8|Ios64|ON|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.ios64.armv8.with_extra.tar.gz)|


## opt 工具

| 运行系统 |      下载       |
| :---------: |  :--------------: | 
|    Linux    |  [release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/opt) |
|    MacOs   |  [release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/opt_mac) |



## 对应源码编译方法

- [opt源码编译](../user_guides/model_optimize_tool.html#opt)
- [Android源码编译](./source_compile.html#paddlelite)
- [iOS源码编译](./source_compile.html#paddlelite)
- [ArmLinux源码编译](./source_compile.html#paddlelite)
- [x86源码编译](../advanced_user_guides/x86)
- [opencl源码编译](../advanced_user_guides/opencl)
- [CUDA源码编译](../advanced_user_guides/cuda)
- [FPGA源码编译](../advanced_user_guides/fpga)
