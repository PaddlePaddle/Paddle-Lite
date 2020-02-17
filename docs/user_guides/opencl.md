# Lite基于OpenCL的ARM GPU预测

Lite支持在Android系统上运行基于OpenCL的程序，目前支持Ubuntu环境下armv8、armv7的交叉编译。

## 编译

### 编译环境

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

详见 **源码编译指南-环境准备** 章节。

### 编译选项

|参数|介绍|值|
|--------|--------|--------|
|--arm_os|代表目标操作系统|目前仅支持且默认为`android`|
|--arm_abi|代表体系结构类型，支持armv8和armv7|默认为`armv8`即arm64-v8a；`armv7`即armeabi-v7a|
|--arm_lang|代表编译目标文件所使用的编译器|默认为gcc，支持 gcc和clang两种|

### 编译Paddle-Lite OpenCL库范例

注：以android-armv8-opencl的目标、Docker容器的编译开发环境为例，CMake3.10，android-ndk-r17c位于`/opt/`目录下。

```bash
# 假设当前位于处于Lite源码根目录下

# 导入NDK_ROOT变量，注意检查您的安装目录若与本示例不同
export NDK_ROOT=/opt/android-ndk-r17c

# 删除上一次CMake自动生成的.h文件
rm ./lite/api/paddle_use_kernels.h
rm ./lite/api/paddle_use_ops.h

# 根据指定编译参数编译
./lite/tools/ci_build.sh \
  --arm_os=android \
  --arm_abi=armv8 \
  --arm_lang=gcc \
  build_test_arm_opencl
```

编译产物位于`build.lite.android.armv8.gcc.opencl`下的`inference_lite_lib.android.armv8.opencl`文件夹内，这里仅罗列关键产物：

- `cxx`:该目录是编译目标的C++的头文件和库文件;
- `demo`:该目录包含了两个demo，用来调用使用`libpaddle_api_full_bundled.a`和`libpaddle_api_light_bundled.a`，分别对应`mobile_full`和`mobile_light`文件夹。编译对应的demo仅需在`mobile_full`或`mobile_light`文
  - `mobile_full`:使用cxx config，可直接加载fluid模型，若使用OpenCL需要在`mobilenetv1_full_api.cc`代码里开启`DEMO_USE_OPENCL`的宏，详细见代码注释;
  - `mobile_light`:使用mobile config，只能加载`model_optimize_tool`优化过的模型;
- `opencl`:该目录存放opencl实现的相关kernel。

```bash
.
|-- cxx
|   |-- include
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib
|       |-- libpaddle_api_full_bundled.a
|       |-- libpaddle_api_light_bundled.a
|       |-- libpaddle_full_api_shared.so
|       `-- libpaddle_light_api_shared.so
|-- demo
|   `-- cxx
|       |-- Makefile.def
|       |-- README.md
|       |-- include
|       |   |-- paddle_api.h
|       |   |-- paddle_lite_factory_helper.h
|       |   |-- paddle_place.h
|       |   |-- paddle_use_kernels.h
|       |   |-- paddle_use_ops.h
|       |   `-- paddle_use_passes.h
|       |-- mobile_full
|       |   |-- Makefile
|       |   `-- mobilenetv1_full_api.cc
|       `-- mobile_light
|           |-- Makefile
|           `-- mobilenetv1_light_api.cc
`-- opencl
    `-- cl_kernel
        |-- buffer
        |   |-- depthwise_conv2d_kernel.cl
        |   |-- elementwise_add_kernel.cl
        |   |-- fc_kernel.cl
        |   |-- im2col_kernel.cl
        |   |-- layout_kernel.cl
        |   |-- mat_mul_kernel.cl
        |   |-- pool_kernel.cl
        |   `-- relu_kernel.cl
        |-- cl_common.h
        `-- image
            |-- channel_add_kernel.cl
            |-- elementwise_add_kernel.cl
            |-- pool_kernel.cl
            `-- relu_kernel.cl
```

调用`libpaddle_api_full_bundled.a`和`libpaddle_api_light_bundled.a`见下一部分运行示例。



## 运行示例

下面以android、ARMv8、gcc的环境为例，介绍3个示例，分别如何在手机上执行基于OpenCL的ARM GPU推理过程。


**注意：** 以下命令均在Lite源码根目录下运行。在3个示例前，下面这段命令都先要执行用来准备环境:

```bash
# 在/data/local/tmp目录下创建OpenCL文件目录
adb shell mkdir -p /data/local/tmp/opencl
adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/buffer
adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/image

# 将OpenCL的kernels文件推送到/data/local/tmp/opencl目录下
adb push lite/backends/opencl/cl_kernel/cl_common.h /data/local/tmp/opencl/cl_kernel/
adb push lite/backends/opencl/cl_kernel/buffer/* /data/local/tmp/opencl/cl_kernel/buffer/
adb push lite/backends/opencl/cl_kernel/image/* /data/local/tmp/opencl/cl_kernel/image/
```

### 运行示例1: 编译产物demo示例

```bash
######################################################################
# 编译mobile_full的demo                                              #
######################################################################
# 步骤:                                                              #
#   0.确保编译Paddle-Lite时编译了OpenCL;                             #
#   1.编辑`mobilenetv1_full_api.cc`代码, 开启`DEMO_USE_OPENCL`的宏;  #
#   2.在产物目录`demo/cxx/mobile_full`下编译`mobile_full`的demo;     #
#   3.上传demo, 模型, opencl kernel文件到手机;                       #
#   4.运行demo得到预期结果.                                          #
######################################################################
adb shell mkdir /data/local/tmp/opencl/mobilenet_v1
chmod +x ./build.lite.android.armv8.gcc.opencl/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_full/mobilenetv1_full_api
adb push ./build.lite.android.armv8.gcc.opencl/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_full/mobilenetv1_full_api /data/local/tmp/opencl/
adb push ./build.lite.android.armv8.gcc.opencl/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1

# use mobile_full run mobilenet_v1
# `GLOG_v` is log level
adb shell "export GLOG_v=0; \
    /data/local/tmp/opencl/mobilenetv1_full_api \
    --model_dir=/data/local/tmp/opencl/mobilenet_v1 \
    --optimized_model_dir=/data/local/tmp/opencl/full_api_opt_model"



######################################################################
# 编译mobile_light的demo                                             #
######################################################################
# 步骤:                                                              #
#   0.确保编译Paddle-Lite时编译了OpenCL;                             #
#   1.编译model_optimize_tool并对模型优化, `targets`参数为`opencl`;  #
#   2.在产物目录`demo/cxx/mobile_light`下编译`mobile_light`的demo;   #
#   3.上传demo, 模型, opencl kernel文件到手机;                       #
#   4.运行demo得到预期结果.                                          #
######################################################################

# use model_optimize_tool to optimize model
./build.model_optimize_tool/lite/api/model_optimize_tool \
  --model_dir=./build.lite.android.armv8.gcc.opencl/install/mobilenet_v1/ \
  --optimize_out_type=naive_buffer \
  --optimize_out=./build.lite.android.armv8.gcc.opencl/install/mobilenet_v1/ \
  --valid_targets=opencl

adb shell mkdir /data/local/tmp/opencl/mobilenet_v1
chmod +x ./build.lite.android.armv8.gcc.opencl/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_light/mobilenetv1_light_api
adb push ./build.lite.android.armv8.gcc.opencl/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_light/mobilenetv1_light_api /data/local/tmp/opencl/
adb push ./build.lite.android.armv8.gcc.opencl/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1

# use mobile_light run mobilenet_v1
adb shell "export GLOG_v=5; \
  /data/local/tmp/opencl/mobilenetv1_light_api \
  --model_dir=/data/local/tmp/opencl/"
```

### 运行示例2: test_mobilenetv1单元测试

- **运行文件准备**

```bash
# 将mobilenet_v1的模型文件推送到/data/local/tmp/opencl目录下
adb shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
adb push build.lite.android.armv8.gcc.opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/

# 将OpenCL单元测试程序test_mobilenetv1，推送到/data/local/tmp/opencl目录下
adb push build.lite.android.armv8.gcc.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl
```

- **执行OpenCL推理过程**

使用如下命令运行OpenCL程序。其中：

- `--cl_path`指定了OpenCL的kernels文件即cl\_kernel所在目录；
- `--modle_dir`指定了模型文件所在目录。

```bash
adb shell chmod +x /data/local/tmp/opencl/test_mobilenetv1

adb shell /data/local/tmp/opencl/test_mobilenetv1 \
  --cl_path=/data/local/tmp/opencl \
  --model_dir=/data/local/tmp/opencl/mobilenet_v1 \
  --warmup=1 \
  --repeats=1
```

**注意：** 因为权重参数均会在Op Kernel第一次运行时进行加载，所以第一次的执行时间会略长。一般将warmup的值设为1，repeats值设为多次。

### 运行示例3: test_layout_opencl单元测试

- **运行文件准备**

```bash
# 将OpenCL单元测试程序test_layout_opencl，推送到/data/local/tmp/opencl目录下
adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_layout_opencl /data/local/tmp/opencl/
```


OpenCL推理过程**

```bash
adb shell chmod +x /data/local/tmp/opencl/test_layout_opencl
adb shell /data/local/tmp/opencl/test_layout_opencl
```


# 如何在Code中使用

见运行示例1的demo代码:

1. [./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc);
2. [./lite/demo/cxx/mobile_full/mobilenetv1_full_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_full/mobilenetv1_full_api.cc).

注：这里给出的链接会跳转到线上最新develop分支的代码，很可能与您本地的代码存在差异，建议参考自己本地位于`lite/demo/cxx/`目录的代码，查看如何使用。

**NOTE：** 对OpenCL的支持还在持续开发中。
