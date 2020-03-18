# PaddleLite使用OpenCL预测部署

Lite支持在Android系统上运行基于OpenCL的程序，目前支持Ubuntu环境下armv8、armv7的交叉编译。

## 编译

### 编译环境

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

详见 **源码编译指南-环境准备** 章节。

### 编译Paddle-Lite OpenCL库范例

注：以android-armv8-opencl的目标、Docker容器的编译开发环境为例，CMake3.10，android-ndk-r17c位于`/opt/`目录下。

#### 针对 Lite 用户的编译命令(无单元测试,有编译产物)

- `arm_os`: `[android]`，目前不支持linux；
- `arm_abi`: `[armv7 | armv8]`；
- `arm_lang`: `[gcc]`，目前不支持clang；
- `build_extra`: `[OFF | ON]`，编译全量op和kernel，体积会大，编译时间长；
- `build_cv`: `[OFF | ON]`，编译arm cpu neon实现的的cv预处理模块；
- `android_stl`: `[c++_shared | c++_static]`，paddlelite的库以何种方式链接`android_stl`，选择`c++_shared`得到的动态库体积更小，但使用时候记得上传paddlelite所编译版本（armv7或armv8）一致的`libc++_shared.so`（来自Android-NDK）；
注：调用`./lite/tools/build.sh`执行编译。

```bash
# 假设当前位于处于Lite源码根目录下

# 导入NDK_ROOT变量，注意检查您的安装目录若与本示例不同
export NDK_ROOT=/opt/android-ndk-r17c

# 删除上一次CMake自动生成的.h文件
rm ./lite/api/paddle_use_kernels.h
rm ./lite/api/paddle_use_ops.h

# 根据指定编译参数编译
./lite/tools/build.sh \
  --arm_os=android \
  --arm_abi=armv8 \
  --arm_lang=gcc \
  --build_extra=OFF \
  --build_cv=OFF \
  --android_stl=c++_shared \
  opencl
```

#### 针对 Lite 开发者的编译命令(有单元测试,编译产物)

注：调用`./lite/tools/ci_build.sh`执行编译，该命令会编译armv7和armv8的opencl库。虽然有编译产物，但因编译单元测试，编译产物包体积可能较大，不推荐使用。

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
  build_opencl
```

注：如果要调试cl kernel，假设已经完成上述脚本编译(已生成cmake文件)。调试只需要修改`./lite/backends/opencl/cl_kernel/`下对应的kernel文件，保存后在项目根目录执行`python ./lite/tools/cmake_tools/gen_opencl_code.py ./lite/backends/opencl/cl_kernel ./lite/backends/opencl/opencl_kernels_source.cc`，该命令会自动将修改后，再切到build目录下执行`make publish_inference`或者你要编译的单测的可执行文件名，cl kernel文件的内容会随着编译自动打包到产物包如 .so 中或者对应单测可执行文件中。

### 编译产物说明

编译产物位于`build.lite.android.armv8.gcc.opencl`下的`inference_lite_lib.android.armv8.opencl`文件夹内，这里仅罗列关键产物：

- `cxx`:该目录是编译目标的C++的头文件和库文件;
- `demo`:该目录包含了两个demo，用来调用使用`libpaddle_api_full_bundled.a`和`libpaddle_api_light_bundled.a`，分别对应`mobile_full`和`mobile_light`文件夹。编译对应的demo仅需在`mobile_full`或`mobile_light`文
  - `mobile_full`:使用cxx config，可直接加载fluid模型，若使用OpenCL需要在`mobilenetv1_full_api.cc`代码里开启`DEMO_USE_OPENCL`的宏，详细见代码注释;
  - `mobile_light`:使用mobile config，只能加载`model_optimize_tool`优化过的模型。
注：`opencl`实现的相关kernel已经打包到动态库中。

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
`-- demo
    `-- cxx
        |-- Makefile.def
        |-- README.md
        |-- include
        |   |-- paddle_api.h
        |   |-- paddle_lite_factory_helper.h
        |   |-- paddle_place.h
        |   |-- paddle_use_kernels.h
        |   |-- paddle_use_ops.h
        |   `-- paddle_use_passes.h
        |-- mobile_full
        |   |-- Makefile
        |   `-- mobilenetv1_full_api.cc
        `-- mobile_light
            |-- Makefile
            `-- mobilenetv1_light_api.cc
```

调用`libpaddle_api_full_bundled.a`和`libpaddle_api_light_bundled.a`见下一部分运行示例。



## 运行示例

下面以android、ARMv8、gcc的环境为例，介绍3个示例，分别如何在手机上执行基于OpenCL的ARM GPU推理过程。

### 运行示例1: 编译产物demo示例

```bash
######################################################################
# 编译mobile_light的demo                                             #
######################################################################
# 步骤:                                                              #
#   0.确保编译Paddle-Lite时编译了OpenCL;                             #
#   1.编译model_optimize_tool并对模型优化, `targets`参数为`opencl`;  #
#   2.在产物目录`demo/cxx/mobile_light`下编译`mobile_light`的demo;   #
#   3.上传demo, 模型文件到手机;                                      #
#   4.运行demo得到预期结果.                                          #
######################################################################
# 在/data/local/tmp目录下创建OpenCL文件目录
adb shell mkdir -p /data/local/tmp/opencl

# use model_optimize_tool to optimize model
./build.model_optimize_tool/lite/api/model_optimize_tool \
  --model_dir=./build.lite.android.armv8.gcc.opencl/install/mobilenet_v1/ \
  --optimize_out_type=naive_buffer \
  --optimize_out=./build.lite.android.armv8.gcc.opencl/install/mobilenet_v1/mobilenetv1_opt \
  --valid_targets=opencl

adb shell mkdir /data/local/tmp/opencl/mobilenet_v1/
chmod +x ./build.lite.android.armv8.gcc.opencl/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_light/mobilenetv1_light_api
adb push ./build.lite.android.armv8.gcc.opencl/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_light/mobilenetv1_light_api /data/local/tmp/opencl/
adb push ./build.lite.android.armv8.gcc.opencl/install/mobilenet_v1/mobilenetv1_opt.nb /data/local/tmp/opencl/

# use mobile_light run mobilenet_v1
adb shell "export GLOG_v=1; \
  /data/local/tmp/opencl/mobilenetv1_light_api \
  /data/local/tmp/opencl/mobilenetv1_opt.nb"
```

**注：** `GLOG_v`是指定需要显示VLOG的日志级别，默认为0。权重参数会在第一次运行时加载，所以第一次执行时间略长。一般将warmup的值设为10，repeats值设为多次。

### 运行示例2: test_mobilenetv1单元测试

- **运行文件准备**

```bash
# 在/data/local/tmp目录下创建OpenCL文件目录
adb shell mkdir -p /data/local/tmp/opencl

# 将mobilenet_v1的模型文件推送到/data/local/tmp/opencl目录下
adb shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
adb push build.lite.android.armv8.gcc.opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/

# 将OpenCL单元测试程序test_mobilenetv1，推送到/data/local/tmp/opencl目录下
adb push build.lite.android.armv8.gcc.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl
```

- **执行OpenCL推理过程**

```bash
adb shell chmod +x /data/local/tmp/opencl/test_mobilenetv1

adb shell "export GLOG_v=1; \
   /data/local/tmp/opencl-image/test_mobilenetv1 \
  --model_dir=/data/local/tmp/opencl-image/mobilenetv1_fluid/ \
  --warmup=10 \
  --repeats=100"
```

### 运行示例3: test_layout_opencl单元测试

```bash
adb shell mkdir -p /data/local/tmp/opencl
adb shell chmod +x /data/local/tmp/opencl/test_layout_opencl
adb shell "export GLOG_v=4; \
  /data/local/tmp/opencl/test_layout_opencl"
```

### 如何在Code中使用

见运行示例1的demo代码:

1. [./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc);
2. [./lite/demo/cxx/mobile_full/mobilenetv1_full_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_full/mobilenetv1_full_api.cc).

注：这里给出的链接会跳转到线上最新develop分支的代码，很可能与您本地的代码存在差异，建议参考自己本地位于`lite/demo/cxx/`目录的代码，查看如何使用。

**NOTE：** 对OpenCL的支持还在持续开发中。
