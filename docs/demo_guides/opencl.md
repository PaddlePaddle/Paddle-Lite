# OpenCL 部署示例

Paddle Lite 支持在 Android 系统上运行基于 OpenCL 的程序，目前支持 Ubuntu 环境下 armv8、armv7 的交叉编译。

## 1. 编译

### 1.1 编译环境

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

详见 **源码编译指南-环境准备** 章节。

### 1.2 编译 Paddle Lite OpenCL 库范例

注：以 `android/armv7/opencl` 的目标、Docker 容器的编译开发环境为例，CMake3.10，android-ndk-r17c 位于 `/opt/` 目录下。

#### 针对 Paddle Lite 用户的编译命令(无单元测试,有编译产物,适用于 benchmark)

- `with_opencl`: `[ON | OFF]`，编译 OpenCL 必选；
- `arm_abi`: `[armv7 | armv8]`；
- `toolchain`: `[gcc | clang]`；
- `build_extra`: `[OFF | ON]`，编译全量 op 和 kernel，包含控制流 NLP 相关的 op 和 kernel 体积会大，编译时间长；
- `build_cv`: `[OFF | ON]`，编译 ARM CPU Neon 实现的的 cv 预处理模块；
- `android_stl`: `[c++_shared | c++_static | gnu_static | gnu_shared]`，Paddle Lite 的库以何种方式链接 `android_stl`，选择 `c++_shared` 得到的动态库体积更小，但使用时候记得上传 Paddle Lite 所编译版本（ armv7 或 armv8 ）一致的 `libc++_shared.so`，默认使用 `c++_static`。

```bash
######################################
# 假设当前位于处于 Paddle Lite 源码根目录下   #
######################################

# 导入 NDK_ROOT 变量，注意检查 NDK 安装目录若与本示例是否不同
export NDK_ROOT=/opt/android-ndk-r17c

# 删除上一次 CMake 自动生成的 .h 文件
rm ./lite/api/paddle_use_kernels.h
rm ./lite/api/paddle_use_ops.h

# 设置编译参数并开始编译
# android-armv7: cpu+gpu+cv+extra
./lite/tools/build_android.sh \
  --arch=armv7 \
  --toolchain=clang \
  --with_log=OFF \
  --with_extra=ON \
  --with_cv=ON \
  --with_opencl=ON
或
./lite/tools/build.sh \
  --arm_os=android \
  --arm_abi=armv7 \
  --arm_lang=clang \
  --android_stl=c++_shared \
  --with_log=OFF \
  --build_extra=ON \
  opencl

# android-armv8: cpu+gpu+cv+extra
./lite/tools/build_android.sh \
  --arch=armv8 \
  --toolchain=clang \
  --with_log=OFF \
  --with_extra=ON \
  --with_cv=ON \
  --with_opencl=ON
或
./lite/tools/build.sh \
  --arm_os=android \
  --arm_abi=armv8 \
  --arm_lang=clang \
  --android_stl=c++_shared \
  --with_log=OFF \
  --build_extra=ON \
  opencl

# 注：编译帮助请执行: ./lite/tools/build_android.sh help
```

注：该方式的编译产物中的 `demo/cxx/mobile_light` 适用于做 benchmark，该过程不会打印开发中加入的 log，注意需要提前转好模型。关于使用，详见下文**运行示例1: 编译产物 demo 示例**。

#### 针对 Paddle Lite 开发者的编译命令(有单元测试,编译产物)

注：调用 `./lite/tools/ci_build.sh` 执行编译，该命令会编译 armv7 和 armv8 的 opencl 库。虽然有编译产物，但因编译单元测试，编译产物包体积可能较大，生产环境不推荐使用。

```bash
# 假设当前位于处于 Paddle Lite 源码根目录下

# 导入 NDK_ROOT 变量，注意检查您的安装目录若与本示例不同
export NDK_ROOT=/opt/android-ndk-r17c

# 删除上一次 CMake 自动生成的 .h 文件
rm ./lite/api/paddle_use_kernels.h
rm ./lite/api/paddle_use_ops.h

# 根据指定编译参数编译
./lite/tools/ci_build.sh \
  --arm_os=android \
  --arm_abi=armv8 \
  --arm_lang=gcc \
  build_opencl
```

注：如果要调试 cl kernel，假设已经完成上述脚本编译(已生成 cmake 文件)。调试只需要修改 `./lite/backends/opencl/cl_kernel/` 下对应的 kernel 文件，保存后在项目根目录执行 `python ./lite/tools/cmake_tools/gen_opencl_code.py ./lite/backends/opencl/cl_kernel ./lite/backends/opencl/opencl_kernels_source.cc`，该命令会自动更新 `opencl_kernels_source.cc`，然后进入 build 目录（如 `build.lite.android.armv8.gcc` ）下执行 `make publish_inference` 或者待编译的单测的可执行文件名（如 `make test_fc_image_opencl`），cl kernel 文件的内容会随着编译自动打包到产物包如 `.so` 中或者对应单测可执行文件中。

### 1.3 编译产物说明

编译产物位于 `build.lite.android.armv8.gcc.opencl` 下的 `inference_lite_lib.android.armv8.opencl` 文件夹内，根据编译参数不同，文件夹名字会略有不同。这里仅罗列关键产物：

- `cxx`: 该目录是编译目标的C++的头文件和库文件;
- `demo`: 该目录包含了两个demo，用来调用使用 `libpaddle_api_full_bundled.a` 和 `libpaddle_api_light_bundled.a`，分别对应 `mobile_full` 和 `mobile_light` 文件夹。编译对应的 demo 仅需在 `mobile_full` 或 `mobile_light` 文件夹下执行 `make`
  - `mobile_full`: 使用 `cxx config`，可直接加载 fluid 模型，若使用 OpenCL 需要在运行时加入 `--use_gpu=true` 选项;
  - `mobile_light`: 使用 `mobile config`，只能加载 `model_optimize_tool` 优化过的模型。
注：`opencl` 实现的相关 kernel 已经打包到动态库中。

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

调用 `libpaddle_api_full_bundled.a` 和 `libpaddle_api_light_bundled.a` 见下一部分运行示例。



## 2. 运行示例

下面以 android 的环境为例，介绍 3 个示例，分别如何在手机上执行基于 OpenCL 的 ARM GPU 推理过程。

### 2.1 运行示例1: 编译产物 demo 示例和 benchmark

需要提前用模型优化工具 opt 转好模型(下面假设已经转换好模型，且模型名为 `mobilenetv1_opencl_fp32_opt_releasev2.6_b8234efb_20200423.nb`)。编译脚本为前文**针对 Paddle Lite 用户的编译命令(无单元测试,有编译产物,适用于 benchmark)**。

```bash
#################################
# 假设当前位于 build.xxx 目录下   #
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
           100 10 0 1 1 0"
           # repeats=100, warmup=10
           # power_mode=0 绑定大核, thread_num=1
           # accelerate_opencl=1 开启 opencl kernel cache & tuning，仅当模型运行在 opencl 后端时该选项才会生效
           # print_output=0 不打印模型输出 tensors 详细数据
```


### 2.2 运行示例2: test_mobilenetv1 单元测试

编译脚本为前文**针对 Paddle  Lite 开发者的编译命令(有单元测试,编译产物)**。

- **运行文件准备**

```bash
# 在 /data/local/tmp 目录下创建 OpenCL 文件目录
adb shell mkdir -p /data/local/tmp/opencl

# 将 mobilenet_v1 的 fluid 格式模型文件推送到 /data/local/tmp/opencl/mobilenet_v1 目录下
adb push build.lite.android.armv8.gcc.opencl/third_party/install/mobilenet_v1/ /data/local/tmp/opencl/mobilenet_v1

# 将 OpenCL 单元测试程序 test_mobilenetv1，推送到 /data/local/tmp/opencl 目录下
adb push build.lite.android.armv8.gcc.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl
```

- **执行 OpenCL 推理过程**

```bash
adb shell chmod +x /data/local/tmp/opencl/test_mobilenetv1

adb shell "export GLOG_v=1; \
   /data/local/tmp/opencl/test_mobilenetv1 \
  --model_dir=/data/local/tmp/opencl/mobilenet_v1/ \
  --warmup=10 \
  --repeats=100"
```

### 2.3 运行示例3: test_layout_opencl 单元测试

编译脚本为前文**针对 Paddle Lite 开发者的编译命令(有单元测试,编译产物)**。

```bash
adb shell mkdir -p /data/local/tmp/opencl
adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_layout_opencl /data/local/tmp/opencl/
adb shell chmod +x /data/local/tmp/opencl/test_layout_opencl
adb shell "export GLOG_v=4; \
  /data/local/tmp/opencl/test_layout_opencl"
```

## 3. 如何在 Code 中使用

即编译产物 `demo/cxx/mobile_light` 目录下的代码，在线版参考 GitHub 仓库[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)，其中也包括判断当前设备是否支持 OpenCL 的方法;

注：这里给出的链接会跳转到线上最新 develop 分支的代码，很可能与您本地的代码存在差异，建议参考自己本地位于 `lite/demo/cxx/` 目录的代码，查看如何使用。

**NOTE：** 对 OpenCL 的支持还在持续开发中。

## 4. 性能分析和精度分析

Android 平台下分析：
```
# 开启性能分析，会打印出每个 op 耗时信息和汇总信息
./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_opencl=ON --with_extra=ON --with_profile=ON full_publish
# 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_opencl=ON --with_extra=ON --with_precision_profile=ON full_publish
```

macOS x86 平台下分析：
```
# 开启性能分析，会打印出每个 op 耗时信息和汇总信息
./lite/tools/build.sh --build_opencl=ON --build_extra=ON --with_profile=ON x86
# 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
./lite/tools/build.sh --build_opencl=ON --build_extra=ON --with_precision_profile=ON x86
```

Windows x86 平台下分析：
```
# 开启性能分析，会打印出每个 op 耗时信息和汇总信息
.\lite\tools\build_windows.bat with_opencl with_extra with_profile
# 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
.\lite\tools\build_windows.bat with_opencl with_extra with_precision_profile
```
详细输出信息的说明可查阅 [Profiler 工具](../user_guides/profiler)。

## 5. 常见问题

1. opencl 计算过程中大多以 `cl::Image2D` 的数据排布进行计算，不同 gpu 支持的最大 `cl::Image2D` 的宽度和高度有限制，模型输入的数据格式是 buffer 形式的 `NCHW` 数据排布方式。要计算你的模型是否超出最大支持（大部分手机支持的 `cl::Image2D` 最大宽度和高度均为 16384），可以通过公式 `image_h = tensor_n * tensor_h, image_w=tensor_w * (tensor_c + 3) / 4` 计算当前层 `NCHW` 排布的 Tensor 所需的 `cl::Image2D` 的宽度和高度；
2. 部署时需考虑不支持 opencl 的情况，可预先使用 API `bool ::IsOpenCLBackendValid()` 判断，对于不支持的情况加载 CPU 模型，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)；
3. 对性能不满足需求的场景，可以考虑使用调优 API `config.set_opencl_tune(CL_TUNE_NORMAL)`，首次会有一定的初始化耗时，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)；
4. 对精度要求较高的场景，可以考虑通过 API `config.set_opencl_precision(CL_PRECISION_FP32)` 强制使用 `FP32` 精度，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)；
5. 对首次加载耗时慢的问题，可以考虑使用 API `config.set_opencl_binary_path_name(bin_path, bin_name)`，提高首次推理时，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
