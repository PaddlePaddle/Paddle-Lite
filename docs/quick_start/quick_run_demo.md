# 试用 Paddle Lite

## 概述

本教程在模型已完成转换和预测库已完成编译情况下，告诉大家如何快速使用 Paddle Lite 推理，以获取最终的推理性能和精度数据。

本文将以安卓端 CPU 为例，介绍口罩检测 Mask Detection 示例。

## 环境准备

此处环境准备包含两个方面：预测库下载和安卓手机环境准备。

### 安卓手机环境

准备一台安卓手机，并在电脑上安装 adb 工具 ，以确保电脑和手机可以通过 adb 连接。

>> 备注：手机通过 USB 连接电脑，打开`设置 -> 开发者模式 -> USB调试 -> 允许（授权）当前电脑调试手机`。保证当前电脑已经安装[ adb 工具](https://developer.android.com/studio/command-line/adb)，运行以下命令，确认当前手机设备已被识别

``` shell
adb devices
# 如果手机设备已经被正确识别，将输出如下信息
List of devices attached
017QXM19C1000664	device
```

### 预测库下载
在预测库[ Lite 预编译库](release_lib)下载界面，可根据您的手机型号和运行需求选择合适版本。

以**Android-ARMv8架构**为例，可以下载以下版本：

| Arch  | with_extra | arm_stl | with_cv | 下载 |
|:-------:|:-----:|:-----:|:-----:|:-------:|
| armv8 | ON | c++_static | OFF |[ 2.10-rc ](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10-rc/inference_lite_lib.android.armv8.clang.c++_static.with_extra.tar.gz)|

**解压后内容结构如下：**

```shell
inference_lite_lib.android.armv8          Paddle Lite 预测库
├── cxx                                       C++ 预测库
│   ├── include                                   C++ 预测库头文件
│   └── lib                                       C++ 预测库文件
│       ├── libpaddle_api_light_bundled.a             静态预测库
│       └── libpaddle_light_api_shared.so             动态预测库
├── demo                                      示例 Demo
│   ├── cxx                                       C++ 示例 Demo
│       ├── mask_detection                           mask_detection Demo 文件夹
│           ├── MakeFile                              MakeFile 文件，用于编译可执行文件
│           └── mask_detection.cc                     C++ 接口的推理源文件
│           └── prepare.sh                            下载模型和预测图片、运行环境准备脚本
│           └── run.sh                                运行 mask_detection 可执行文件脚本
│   └── java                                      Java 示例 Demo
└── java                                      Java 预测库
```

## 运行
在环境准备好，按照下述步骤完成口罩检测 Mask Detection 推理，获取模型的性能和精度数据

```shell
cd inference_lite_lib.android.armv8/demo/cxx/mask_detection

# 设置 NDK_ROOT 路径
export NDK_ROOT=/opt/android-ndk-r20b

# 准备预测部署文件
bash prepare.sh

# 执行预测
cd mask_demo && bash run.sh

# 运行成功后，将在控制台输出如下内容，可以打开 test_img_result.jpg 图片查看预测结果
../mask_demo/: 9 files pushed, 0 skipped. 141.6 MB/s (28652282 bytes in 0.193s)
Load detecion model succeed.

======= benchmark summary =======
model_dir: pyramidbox_lite_v2_10_opt2.nb
repeats: 100
*** time info(ms) ***
1st_duration: 124.481
max_duration: 123.179
min_duration: 40.093
avg_duration: 41.2289
detection pre_process time: 4.924
Detecting face succeed.

Load classification model succeed.
detect face, location: x=237, y=107, width=194, height=255, wear mask: 1, prob: 0.987625
detect face, location: x=61, y=238, width=166, height=213, wear mask: 1, prob: 0.925679
detect face, location: x=566, y=176, width=245, height=294, wear mask: 1, prob: 0.550348
write result to file: test_img_result.jpg, success.
/data/local/tmp/mask_demo/test_img_result.jpg: 1 file pulled, 0 skipped. 28.0 MB/s (279080 bytes in 0.010s)
```
