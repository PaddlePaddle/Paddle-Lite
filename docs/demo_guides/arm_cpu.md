# ARM CPU 部署示例

Lite 支持在 Android/IOS/ARMLinux 等移动端设备上运行高性能的 CPU 预测库，目前支持 Ubuntu 环境下 armv8、armv7 的交叉编译。

## 1. 编译

### 1.1 编译环境

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

详见 **源码编译指南-环境准备** 章节。

### 1.2 编译 Paddle Lite ARM CPU 库范例

注：以 `android/armv8` 目标、Docker 容器的编译开发环境为例，CMake3.10，android-ndk-r17c 位于 `/opt/` 目录下。

#### 针对 Paddle Lite 用户的编译命令(无单元测试,有编译产物,适用于 benchmark)

- `arm_abi`: `[armv7 | armv8]`；
- `toolchain`: `[gcc | clang]`；
- `build_extra`: `[OFF | ON]`，编译全量 op 和 kernel，包含控制流 NLP 相关的 op 和 kernel 体积会大，编译时间长；
- `build_cv`: `[OFF | ON]`，编译 ARM CPU Neon 实现的的 cv 预处理模块；
- `android_stl`: `[c++_shared | c++_static | gnu_static | gnu_shared]`，Paddle Lite 的库以何种方式链接 `android_stl` ，选择 `c++_shared` 得到的动态库体积更小，但使用时候记得上传 Paddle Lite 所编译版本（armv7 或 armv8 ）一致的 `libc++_shared.so`, 默认使用 `c++_static`。

```bash
######################################
# 假设当前位于处于 Paddle Lite 源码根目录下   #
######################################

# 导入 NDK_ROOT 变量，注意检查 NDK 安装目录若与本示例是否不同
export NDK_ROOT=/opt/android-ndk-r17c

# 删除上一次 CMake 自动生成的 `.h` 文件
rm ./lite/api/paddle_use_kernels.h
rm ./lite/api/paddle_use_ops.h

# 设置编译参数并开始编译
# android-armv8: cpu+cv+extra
./lite/tools/build_android.sh \
  --arch=armv8 \
  --toolchain=clang \
  --with_log=OFF \
  --with_extra=ON \
  --with_cv=ON

# android-armv7: cpu+cv+extra
./lite/tools/build_android.sh \
  --arch=armv7 \
  --toolchain=clang \
  --with_log=OFF \
  --with_extra=ON \
  --with_cv=ON

# android-armv8-(v8.2+FP16): cpu+FP16+cv+extra
# update NDK version > 19
export NDK_ROOT=/opt/android-ndk-r20b
./lite/tools/build_android.sh \
  --arch=armv8 \
  --toolchain=clang \
  --with_log=OFF \
  --with_extra=ON \
  --with_arm82_fp16=ON \
  --with_cv=ON

# 注：编译帮助请执行: `./lite/tools/build_android.sh` help
```
>> 注意：
- 该方式的编译产物中的 `demo/cxx/mobile_light` 适用于做 benchmark，该过程不会打印开发中加入的 log，注意需要提前转好模型。关于使用，详见下文**运行示例1: 编译产物 demo 示例**
- 如果运行 FP16 预测库，模型在 OPT 转换的时候需要加上 `--enable_fp16=1` 选项，这样转换的模型会选择 **FP16 kernel** 实现。并且，FP16 预测库和 FP16 模型**只在支持 ARMv8.2 架构的手机**上运行，如小米 9，华为 Meta30 等。
- 当前 Paddle Lite只支持 **ARMv8 架构**的 FP16 运算。

#### 针对 Lite 开发者的编译命令(有单元测试,编译产物)

>> 注：调用`./lite/tools/ci_build.sh` 执行编译，该命令会编译 armv7 和 armv8 的预测库。虽然有编译产物，但因编译单元测试，编译产物包体积可能较大，生产环境不推荐使用。

```bash
# 假设当前位于处于 Paddle Lite 源码根目录下

# 导入 NDK_ROOT 变量，注意检查您的安装目录若与本示例不同
export NDK_ROOT=/opt/android-ndk-r17c

# 删除上一次 CMake 自动生成的 `.h` 文件
rm ./lite/api/paddle_use_kernels.h
rm ./lite/api/paddle_use_ops.h

# 根据指定编译参数编译
./lite/tools/build.sh \
  --arm_os=android \
  --arm_abi=armv8 \
  --arm_lang=clang \
  --build_extra=on \
  --build_cv=on \
  test
```

### 1.3 编译产物说明

编译产物位于 `build.lite.android.armv8.clang` 下的 `lite` 文件夹内。这里仅罗列关键产物：

- `api`: 包含了基于 API 接口和模型的各种可执行的单测文件
- `tests`:该目录包含了多个层面的可执行的单测文件
   - `kernels`: 包含已支持 OP 的各种可执行的单测文件，如 `activation` OP 单测；
   - `benchmark`: 提供便利化脚本用于 convolution/pooling 等算子性能的批量测试
   - `math`: 包含各类卷积算子如 `GEMM`、`GEMV` 等可执行的单测文件
```bash
.
|-- api
|   |-- *.a
|   |-- *.so
|   |-- test_model_bin
|   |-- test_mobilenetv1
|   |-- test_mobilenetv1_int8
|   |....
|-- kernel
|   |-- apu
|   |-- arm
|   |   |-- *.a(example:libconv_compute_arm.a,  libmul_compute_arm.a etc.)
|   |-- bm
|   |-- cuda
|   |-- host
|   |....
|-- tests
|   |-- api
|   |   |-- test_inception_v4_fp32_arm
|   |   |-- test_mobilenet_v1_int8_dygraph_arm
|   |   |-- test_nlp_lstm_int8_arm
|   |   |...
|   |-- benchmark
|   |   |-- get_activation_latency
|   |   |-- get_batchnorm_latency
|   |   |-- get_conv_latency
|   |   |...
|   |-- cv
|   |   |-- image_convert_test
|   |   |-- image_profiler_test
|   |-- kernels
|   |   |-- test_kernel_activation_compute
|   |   |-- test_kernel_expand_as_compute
|   |   |-- test_kernel_group_norm_compute
|   |   |...
|   |-- math
|   |   |-- conv_compute_test
|   |   |-- sgemm_compute_test
|   |   |-- sgemv_compute_test
|   |   |...
....
```

## 2. 运行示例

下面以 android 的环境为例，介绍 3 个示例，分别如何在手机上执行 ARM CPU 推理过程。

### 2.1 运行示例1: 编译产物 demo 示例和 benchmark

需要提前用模型优化工具 opt 转好模型(下面假设已经转换好模型，且模型名为 `mobilenetv1_fp32.nb`)。

编译脚本为前文**针对 Paddle Lite 用户的编译命令(无单元测试,有编译产物,适用于 benchmark)**。
注：产物 demo 需要用 `tiny_publish` 或 `full_publish` 方式编译才能获取。

```bash
#################################
# 假设当前位于 build.xxx 目录下   #
#################################

# prepare enviroment on phone
adb shell mkdir -p /data/local/tmp/arm_cpu/

# build demo
cd inference_lite_lib.android.armv8/demo/cxx/mobile_light/
make
cd -

# push executable binary, library to device
adb push inference_lite_lib.android.armv8/demo/cxx/mobile_light/mobilenetv1_light_api /data/local/tmp/arm_cpu/
adb shell chmod +x /data/local/tmp/arm_cpu/mobilenetv1_light_api
adb push inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/arm_cpu/

# push model with optimized(opt) to device
adb push ./mobilenetv1_fp32.nb /data/local/tmp/arm_cpu/

# run demo on device
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/mobilenetv1_fp32/; \
           /data/local/tmp/mobilenetv1_fp32/mobilenetv1_light_api \
           /data/local/tmp/mobilenetv1_fp32/mobilenetv1_fp32.nb \
           1,3,224,224 \
           100 10 0 1 1 0" 
           # repeats=100, warmup=10
           # power_mode=0 绑定大核, thread_num=1
           # print_output=0 不打印模型输出 tensors 详细数据
```
注：如果要运行 FP16 模型，需要提前完成以下操作：

 - 在编译预测库时，需要添加 `with_arm82_fp16=ON` 选项进行编译；
 - OPT 模型转换时，需要添加 `--enable_fp16=1` 选项，完成 FP16 模型转换
 - 只能在**V8.2 架构以上的手机**执行，即高端手机，如小米9，华为 P30 等
 - 推理执行过程同上

### 2.2 运行示例2: `test_model_bin` 单元测试

编译脚本为前文**针对 Paddle Lite 开发者的编译命令(有单元测试,编译产物)**。

- **运行文件准备**

```bash
# 在 `/data/local/tmp` 目录下创建 `arm_cpu` 文件目录
adb shell mkdir -p /data/local/tmp/arm_cpu

# 将单元测试程序 test_model_bin，推送到 `/data/local/tmp/arm_cpu` 目录下
adb push build.lite.android.armv8.clang/lite/api/test_model_bin /data/local/tmp/arm_cpu
```

- **执行推理过程**

```bash
# 将转换好的模型文件推送到 `/data/local/tmp/arm_cpu` 目录下
adb push caffe_mv1_fp32.nb /data/local/tmp/arm_cpu/
adb shell chmod +x /data/local/tmp/arm_cpu/test_mobilenetv1

adb shell "export GLOG_v=1; \
   /data/local/tmp/arm_cpu/test_mobilenetv1 \
  --use_optimize_nb=1 \
  --model_dir=/data/local/tmp/arm_cpu/caffe_mv1_fp32 \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=100"
```

- **FP16 模型推理过程**

1. 单测编译的时候，需要添加 `--build_arm82_fp16=ON` 选项，即：

```bash
export NDK_ROOT=/disk/android-ndk-r20b #ndk_version > 19
./lite/tools/build.sh \
--arm_os=android \
--arm_abi=armv8 \
--build_extra=on \
--build_cv=on \
--arm_lang=clang \
--build_arm82_fp16=ON \
test
```

2. 模型在 OPT 转换的时候，需要添加 `--enable_fp16=1` 选项，完成 FP16 模型转换，即：

```bash
./build.opt/lite/api/opt \
--optimize_out_type=naive_buffer \
--enable_fp16=1 \
--optimize_out caffe_mv1_fp16 \
--model_dir ./caffe_mv1
```

3. 执行

1) 推送 OPT 转换后的模型至手机, 运行时请将 `use_optimize_nb` 设置为1

```bash
# 将转换好的模型文件推送到 `/data/local/tmp/arm_cpu` 目录下
adb push caffe_mv1_fp16.nb /data/local/tmp/arm_cpu/
adb shell chmod +x /data/local/tmp/arm_cpu/test_mobilenetv1

adb shell "\
   /data/local/tmp/arm_cpu/test_mobilenetv1 \
  --use_optimize_nb=1 \
  --model_dir=/data/local/tmp/arm_cpu/caffe_mv1_fp16 \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=100"
```

2) 推送原始模型至手机, 运行时请将 `use_optimize_nb` 设置为0， `use_fp16` 设置为1；（`use_fp16` 默认为0）

```bash
# 将 fluid 原始模型文件推送到 `/data/local/tmp/arm_cpu` 目录下
adb push caffe_mv1 /data/local/tmp/arm_cpu/
adb shell chmod +x /data/local/tmp/arm_cpu/test_mobilenetv1

adb shell "export GLOG_v=1; \
   /data/local/tmp/arm_cpu/test_mobilenetv1 \
  --use_optimize_nb=0 \
  --use_fp16=1 \
  --model_dir=/data/local/tmp/arm_cpu/caffe_mv1 \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=100"
```

注：如果想输入真实数据，请将预处理好的输入数据用文本格式保存。在执行的时候加上 `--in_txt=./*.txt` 选项即可

### 2.3 运行示例3: conv_compute_test 单元测试

编译脚本为前文**针对 Paddle Lite 开发者的编译命令(有单元测试,编译产物)**。

```bash
adb shell mkdir -p /data/local/tmp/arm_cpus
adb push build.lite.android.armv8.clang/lite/test/math/conv_compute_test /data/local/tmp/arm_cpu
adb shell chmod +x /data/local/tmp/arm_cpu/conv_compute_test
adb shell "export GLOG_v=4; \
  /data/local/tmp/arm_cpu/conv_compute_test --basic_test=0" # basic_test 表示是否跑所有单测案例
# 如果想跑某个 case 的 convolution 单测：
adb shell "export GLOG_v=4; \
  /data/local/tmp/arm_cpu/conv_compute_test --basic_test=0 --in_channel=3 \
  --out_channel=32 --in_height=224 --in_width=224 --group=1 --kernel_h=3 --kernel_w=3 \
  --stride_w=2 --stride_h=2 --pad_h0=1 --pad_h1=1 --pad_w0=1 --pad_w1=1 --flag_act=1 \
  --flag_bias=0 --warmup=10 --repeats=100 --threads=1"
# 如果想跑 GEMM 单测：
adb shell "export GLOG_v=4; \
  /data/local/tmp/arm_cpu/sgemm_compute_test --basic_test=0 --M=32 --N=128 --K=1024 \
  --warmup=10 --repeats=100 --threads=1"
```

## 3. 性能分析和精度分析

Android 平台下分析：

### 1. 开启性能分析，会打印出每个 op 耗时信息和汇总信息

```bash
./lite/tools/build.sh \
--arm_os=android \
--arm_abi=armv8 \
--build_extra=on \
--build_cv=on \
--arm_lang=clang \
--with_profile=ON \
test
```

### 2. 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息

```bash
# 开启性能分析，会打印出每个 op 耗时信息和汇总信息
./lite/tools/build.sh \
--arm_os=android \
--arm_abi=armv8 \
--build_extra=on \
--build_cv=on \
--arm_lang=clang \
--with_profile=ON \
--with_precision_profile=ON \
test
```

详细输出信息的说明可查阅 [Profiler 工具](../user_guides/profiler)。
