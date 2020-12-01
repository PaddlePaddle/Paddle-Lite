# PaddleLite使用OpenCL预测部署

Lite支持在Android系统上运行基于OpenCL的程序，目前支持Ubuntu环境下armv8、armv7的交叉编译。

## 1. 编译

### 1.1 编译环境

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

详见 **源码编译指南-环境准备** 章节。

### 1.2 编译Paddle-Lite OpenCL库范例

注：以android/armv7/opencl的目标、Docker容器的编译开发环境为例，CMake3.10，android-ndk-r17c位于`/opt/`目录下。

#### 针对 Lite 用户的编译命令(无单元测试,有编译产物,适用于benchmark)

- `with_opencl`: `[ON | OFF]`，编译OpenCL必选；
- `arm_abi`: `[armv7 | armv8]`；
- `toolchain`: `[gcc | clang]`；
- `build_extra`: `[OFF | ON]`，编译全量op和kernel，包含控制流NLP相关的op和kernel体积会大，编译时间长；
- `build_cv`: `[OFF | ON]`，编译arm cpu neon实现的的cv预处理模块；
- `android_stl`: `[c++_shared | c++_static | gnu_static | gnu_shared]`，paddlelite的库以何种方式链接`android_stl`，选择`c++_shared`得到的动态库体积更小，但使用时候记得上传paddlelite所编译版本（armv7或armv8）一致的`libc++_shared.so`。默认使用`c++_static`。

```bash
######################################
# 假设当前位于处于Lite源码根目录下   #
######################################

# 导入NDK_ROOT变量，注意检查NDK安装目录若与本示例是否不同
export NDK_ROOT=/opt/android-ndk-r17c

# 删除上一次CMake自动生成的.h文件
rm ./lite/api/paddle_use_kernels.h
rm ./lite/api/paddle_use_ops.h

# 设置编译参数并开始编译
# android-armv7:cpu+gpu+cv+extra
./lite/tools/build_android.sh \
  --arch=armv7 \
  --toolchain=clang \
  --with_log=OFF \
  --with_extra=ON \
  --with_cv=ON \
  --with_opencl=ON

# android-armv8:cpu+gpu+cv+extra
./lite/tools/build_android.sh \
  --arch=armv8 \
  --toolchain=clang \
  --with_log=OFF \
  --with_extra=ON \
  --with_cv=ON \
  --with_opencl=ON


# 注：编译帮助请执行: ./lite/tools/build_android.sh help
```

注：该方式的编译产物中的`demo/cxx/mobile_light`适用于做benchmark，该过程不会打印开发中加入的log，注意需要提前转好模型。关于使用，详见下文**运行示例1: 编译产物demo示例**。

#### 针对 Lite 开发者的编译命令(有单元测试,编译产物)

注：调用`./lite/tools/ci_build.sh`执行编译，该命令会编译armv7和armv8的opencl库。虽然有编译产物，但因编译单元测试，编译产物包体积可能较大，生产环境不推荐使用。

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

### 1.3 编译产物说明

编译产物位于`build.lite.android.armv8.gcc.opencl`下的`inference_lite_lib.android.armv8.opencl`文件夹内，根据编译参数不同，文件夹名字会略有不同。这里仅罗列关键产物：

- `cxx`:该目录是编译目标的C++的头文件和库文件;
- `demo`:该目录包含了两个demo，用来调用使用`libpaddle_api_full_bundled.a`和`libpaddle_api_light_bundled.a`，分别对应`mobile_full`和`mobile_light`文件夹。编译对应的demo仅需在`mobile_full`或`mobile_light`文件夹下执行`make`
  - `mobile_full`:使用cxx config，可直接加载fluid模型，若使用OpenCL需要在`mobilenetv1_full_api.cc`代码里开启`DEMO_USE_OPENCL`的宏，详细见该文件的代码注释;
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



## 2. 运行示例

下面以android的环境为例，介绍3个示例，分别如何在手机上执行基于OpenCL的ARM GPU推理过程。

### 2.1 运行示例1: 编译产物demo示例和benchmark

需要提前用模型优化工具opt转好模型(下面假设已经转换好模型，且模型名为`mobilenetv1_opencl_fp32_opt_releasev2.6_b8234efb_20200423.nb`)。编译脚本为前文**针对 Lite 用户的编译命令(无单元测试,有编译产物,适用于benchmark)**。

```bash
#################################
# 假设当前位于build.xxx目录下   #
#################################

# prepare enviroment on phone
adb shell mkdir -p /data/local/tmp/opencl/

# build demo
cd inference_lite_lib.android.armv7.opencl/demo/cxx/mobile_light/
make
cd -

# push executable binary, library to device
adb push inference_lite_lib.android.armv7.opencl/demo/cxx/mobile_light/mobilenetv1_light_api /data/local/tmp/opencl/
adb shell chmod +x /data/local/tmp/opencl/mobilenetv1_light_api
adb push inference_lite_lib.android.armv7.opencl/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/opencl/

# push model with optimized(opt) to device
adb push ./mobilenetv1_opencl_fp32_opt_releasev2.6_b8234efb_20200423.nb /data/local/tmp/opencl/

# run demo on device
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/opencl/; \
           /data/local/tmp/opencl/mobilenetv1_light_api \
           /data/local/tmp/opencl/mobilenetv1_opencl_fp32_opt_releasev2.6_b8234efb_20200423.nb \
           1,3,224,224 \
           100 10 0" # round=100, warmup=10, print_output_tensor=0
```

**注：** 权重参数会在第一次运行时加载，且`.cl`文件也会在第一次运行时在线编译，所以第一次执行时间略长。一般将warmup的值设为10，repeats值设为多次。

### 2.2 运行示例2: test_mobilenetv1单元测试

编译脚本为前文**针对 Lite 开发者的编译命令(有单元测试,编译产物)**。

- **运行文件准备**

```bash
# 在/data/local/tmp目录下创建OpenCL文件目录
adb shell mkdir -p /data/local/tmp/opencl

# 将mobilenet_v1的fluid格式模型文件推送到/data/local/tmp/opencl/mobilenet_v1目录下
adb push build.lite.android.armv8.gcc.opencl/third_party/install/mobilenet_v1/ /data/local/tmp/opencl/mobilenet_v1

# 将OpenCL单元测试程序test_mobilenetv1，推送到/data/local/tmp/opencl目录下
adb push build.lite.android.armv8.gcc.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl
```

- **执行OpenCL推理过程**

```bash
adb shell chmod +x /data/local/tmp/opencl/test_mobilenetv1

adb shell "export GLOG_v=1; \
   /data/local/tmp/opencl/test_mobilenetv1 \
  --model_dir=/data/local/tmp/opencl/mobilenet_v1/ \
  --warmup=10 \
  --repeats=100"
```

### 2.3 运行示例3: test_layout_opencl单元测试

编译脚本为前文**针对 Lite 开发者的编译命令(有单元测试,编译产物)**。

```bash
adb shell mkdir -p /data/local/tmp/opencl
adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_layout_opencl /data/local/tmp/opencl/
adb shell chmod +x /data/local/tmp/opencl/test_layout_opencl
adb shell "export GLOG_v=4; \
  /data/local/tmp/opencl/test_layout_opencl"
```

## 3. 如何在Code中使用

即编译产物`demo/cxx/mobile_light`目录下的代码，在线版参考GitHub仓库[./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)，其中也包括判断当前设备是否支持OpenCL的方法;

注：这里给出的链接会跳转到线上最新develop分支的代码，很可能与您本地的代码存在差异，建议参考自己本地位于`lite/demo/cxx/`目录的代码，查看如何使用。

**NOTE：** 对OpenCL的支持还在持续开发中。

## 4. 常见问题

1. opencl计算过程中大多以`cl::Image2D`的数据排布进行计算，不同gpu支持的最大`cl::Image2D`的宽度和高度有限制，模型输入的数据格式是buffer形式的`NCHW`数据排布方式。要计算你的模型是否超出最大支持（大部分手机支持的`cl::Image2D`最大宽度和高度均为16384），可以通过公式`image_h = tensor_n * tensor_h, image_w=tensor_w * (tensor_c + 3) / 4`计算当前层NCHW排布的Tensor所需的`cl::Image2D`的宽度和高度；
2. 部署时需考虑不支持opencl的情况，可预先使用API`bool ::IsOpenCLBackendValid()`判断，对于不支持的情况加载CPU模型，详见[./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
