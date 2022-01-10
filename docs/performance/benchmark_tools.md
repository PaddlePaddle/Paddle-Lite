# 性能测试

当我们已经有一个 Paddle 格式的模型后，我们可以使用 Benchmark 工具对该模型进行性能测试。Benchmark 工具可以输出的性能指标包括但不限于：
- 初始化耗时
- 首帧耗时
- 平均耗时

Benchmark 工具的详细功能包括但不限于：
- 支持 Paddle combined / uncombined 格式模型作为输入模型
- 支持 Paddle Lite .nb 格式模型作为输入模型
- 支持单输入和多输入模型
- 支持从文本读取输入数据
- 支持设置不同的运行时精度
- 支持时间 profile 和精度 profile

## 适用场景
Benchmark 工具可方便快捷地评测给定模型在如下硬件上运行时的性能：
- 安卓系统下的 ARM CPU / GPU / NNAdapter
- Linux 系统下的 x86 CPU / ARM CPU / ARM GPU / NNAdapter
- macOS 系统下的 x86 CPU / ARM CPU / GPU

## 在 Android 上运行性能测试
### 编译
根据[源码编译](../source_compile/compile_env)准备编译环境，建议使用 Docker 配置交叉编译环境。
拉取 [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) 代码，切换到特定分支，然后在 Paddle Lite 根目录下执行编译命令：
```shell
./lite/tools/build_android.sh --toolchain=clang --with_benchmark=ON full_publish
```
可选参数：

| 参数 | 说明 | 可选值 | 默认值 |
| :-- | :-- | :-- | :-- |
| arch                  | 目标 ARM 架构    |  armv7hf / armv7 / armv8   |  armv8   |
| toolchain             | 工具链           |  gcc / clang     |  gcc     |
| with_profile          | 逐层时间 profile |  ON / OFF        |  OFF     |
| with_precision_profile| 逐层精度 profile |  ON / OFF        |  OFF     |

编译完成后，会生成`build.lite.*./lite/api/tools/benchmark/benchmark_bin`二进制文件。

### 运行
需要将如下文件通过`adb`上传至手机：
- Paddle 模型（combined 或 uncombined 格式均可）或已经`opt`工具离线优化后的`.nb`文件
- 二进制文件`benchmark_bin`

在 Host 端机器上操作例子如下：
```shell
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 上传文件
adb shell mkdir /data/local/tmp/benchmark
adb push MobileNetV1 /data/local/tmp/benchmark
adb push build.lite.android.armv8.clang/lite/api/tools/benchmark/benchmark_bin /data/local/tmp/benchmark

# 执行性能测试
adb shell "cd /data/local/tmp/benchmark;
  ./benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=arm"
```

会输出如下信息：
```shell
======= Opt Info =======
Load paddle model from inference.pdmodel and inference.pdiparams
Save optimized model to .nb

======= Device Info =======
Brand: Xiaomi
Device: cepheus
Model: MI 9
Android Version: 10
Android API Level: 29

======= Model Info =======
optimized_model_file: .nb
input_data_path:
input_shape: 1,3,224,224
output tensor num: 1
--- output tensor 0 ---
output shape(NCHW): 1 1000
output tensor 0 elem num: 1000
output tensor 0 mean value: 0.001
output tensor 0 standard deviation: 0.00219647

======= Runtime Info =======
benchmark_bin version: acf6614
threads: 1
power_mode: 0
warmup: 10
repeats: 20
result_path:

======= Backend Info =======
backend: arm
cpu precision: fp32

======= Perf Info =======
Time(unit: ms):
init  = 15.305
first = 43.670
min   = 32.577
max   = 32.895
avg   = 32.723
```

## 在 ARM Linux 上运行性能测试
### 编译
根据[源码编译](../source_compile/compile_env)准备编译环境，建议使用 Docker 配置交叉编译环境。
拉取 [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) 代码，切换到特定分支，然后在 Paddle Lite 根目录下执行编译命令：
```shell
./lite/tools/build_linux.sh --arch=armv8 --with_benchmark=ON full_publish
```
可选参数：

| 参数 | 说明 | 可选值 | 默认值 |
| :-- | :-- | :-- | :-- |
| arch                  | 目标 ARM 架构    |  armv7 / armv8   |  armv8   |
| toolchain             | 工具链           |  gcc / clang     |  gcc     |
| with_profile          | 逐层时间 profile |  ON / OFF        |  OFF     |
| with_precision_profile| 逐层精度 profile |  ON / OFF        |  OFF     |

编译完成后，会生成`build.lite.*./lite/api/tools/benchmark/benchmark_bin`二进制文件。

### 运行
需要将如下文件通过`scp`或其他方式上传至 ARM Linux 设备：
- Paddle 文件（combined 或 uncombined 格式均可）或已经`opt`工具离线优化后的`.nb`文件
- 二进制文件`benchmark_bin`

在 Host 端机器上操作例子如下：
```shell
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 上传文件到 ARM Linux 设备

```

然后通过`ssh`登录到 ARM Linux 设备，执行：
```shell
# 性能测试
cd /path/to/benchmark_bin; \
./benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=arm
```
会输出如下信息：
```shell
======= Opt Info =======
Load paddle model from inference.pdmodel and inference.pdiparams
Save optimized model to .nb

======= Model Info =======
optimized_model_file: .nb
input_data_path:
input_shape: 1,3,224,224
output tensor num: 1
--- output tensor 0 ---
output shape(NCHW): 1 1000
output tensor 0 elem num: 1000
output tensor 0 mean value: 0.001
output tensor 0 standard deviation: 0.00219647

======= Runtime Info =======
benchmark_bin version: acf6614
threads: 1
power_mode: 0
warmup: 10
repeats: 20
result_path:

======= Backend Info =======
backend: arm
cpu precision: fp32

======= Perf Info =======
Time(unit: ms):
init  = 15.305
first = 43.670
min   = 32.577
max   = 32.895
avg   = 32.723
```

## 在 Linux 上运行性能测试
### 编译
根据[源码编译](../source_compile/compile_env)准备编译环境，建议使用 Docker 配置环境。
拉取 [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) 代码，切换到特定分支，然后在 Paddle Lite 根目录下执行编译命令：
```shell
./lite/tools/build_linux.sh --arch=x86 --with_benchmark=ON full_publish
```
可选参数：

| 参数 | 说明 | 可选值 | 默认值 |
| :-- | :-- | :-- | :-- |
| toolchain             | 工具链           |  gcc / clang     |  gcc     |
| with_profile          | 逐层时间 profile |  ON / OFF        |  OFF     |
| with_precision_profile| 逐层精度 profile |  ON / OFF        |  OFF     |

编译完成后，会生成`build.lite.*./lite/api/tools/benchmark/benchmark_bin`二进制文件。

### 运行
运行所需文件：
- Paddle 文件（combined 或 uncombined 格式均可）或已经`opt`工具离线优化后的`.nb`文件
- 二进制文件`benchmark_bin`
- `libmklml_intel.so`

在待测试的 Linux 机器上操作例子如下：
```shell
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 设置环境变量
export LD_LIBRARY_PATH=build.lite.x86.gcc/third_party/install/mklml/lib/:$LD_LIBRARY_PATH

# 执行性能测试
./build.lite.linux.x86.gcc/lite/api/tools/benchmark/benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=x86
```
会输出如下信息：
```shell
======= Opt Info =======
Load paddle model from MobileNetV1/inference.pdmodel and MobileNetV1/inference.pdiparams
Save optimized model to .nb


======= Model Info =======
optimized_model_file: .nb
input_data_path: All 1.f
input_shape: 1,3,224,224
output tensor num: 1
--- output tensor 0 ---
output shape(NCHW): 1 1000
output tensor 0 elem num: 1000
output tensor 0 mean value: 0.001
output tensor 0 standard deviation: 0.00219647

======= Runtime Info =======
benchmark_bin version: 380d8d0
threads: 1
power_mode: 0
warmup: 10
repeats: 20
result_path:

======= Backend Info =======
backend: x86
cpu precision: fp32

======= Perf Info =======
Time(unit: ms):
init  = 18.135
first = 80.052
min   = 31.982
max   = 38.947
avg   = 33.918
```

## 在 macOS 上运行性能测试
### 编译
根据[源码编译](../source_compile/compile_env)准备编译环境，可以使用 Docker 配置环境，也可以使用系统原生开发环境。
拉取 [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) 代码，切换到特定分支，然后在 Paddle Lite 根目录下执行编译命令：
```shell
# 芯片为 x86 架构时，执行：
./lite/tools/build_macos.sh --with_benchmark=ON x86

# 芯片为 ARM 架构时，执行：
./lite/tools/build_macos.sh --with_benchmark=ON arm64
```
可选参数：

| 参数 | 说明 | 可选值 | 默认值 |
| :-- | :-- | :-- | :-- |
| toolchain             | 工具链          |  gcc / clang     |  gcc     |
| with_profile          | 逐层时间 profile |  ON / OFF        |  OFF     |
| with_precision_profile| 逐层精度 profile |  ON / OFF        |  OFF     |

编译完成后，会生成`build.lite.*./lite/api/tools/benchmark/benchmark_bin`二进制文件。

### 运行
运行所需文件：
- Paddle 文件（combined 或 uncombined 格式均可）或已经`opt`工具离线优化后的`.nb`文件
- 二进制文件`benchmark_bin`
- `libmklml.dylib`

在 macOS 机器上操作例子如下：
```shell
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 设置环境变量
export LD_LIBRARY_PATH=build.lite.x86.opencl/third_party/install/mklml/lib/:$LD_LIBRARY_PATH

# 执行性能测试
./build.lite.x86.opencl/lite/api/tools/benchmark/benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=x86
```
会输出如下信息：
```shell
======= Opt Info =======
Load paddle model from MobileNetV1/inference.pdmodel and MobileNetV1/inference.pdiparams
Save optimized model to MobileNetV1/opt.nb


======= Model Info =======
optimized_model_file: MobileNetV1/opt.nb
input_data_path: All 1.f
input_shape: 1,3,224,224
output tensor num: 1
--- output tensor 0 ---
output shape(NCHW): 1 1000
output tensor 0 elem num: 1000
output tensor 0 mean value: 0.001
output tensor 0 standard deviation: 0.00219646

======= Runtime Info =======
benchmark_bin version: 380d8d004
threads: 1
power_mode: 0
warmup: 0
repeats: 1
result_path:

======= Backend Info =======
backend: x86
cpu precision: fp32

======= Perf Info =======
Time(unit: ms):
init  = 11.410
first = 53.964
min   = 53.964
max   = 53.964
avg   = 53.964
```

## 高阶用法
Benchnark 工具提供了丰富的运行时选项，来满足不同的运行时参数设置。用户可以通过在目标设备上执行`./benchmark_bin --help`获取所有选项介绍。

### 指定不同的 backend
#### 在 CPU 上运行模型
- 设备 OS 为 Android 或 ARM Linux 时，通过使用`--backend=arm`来实现
- 设备 OS 为 Linux 或 macOS(x86 芯片) 时，通过使用`--backend=x86`来实现

#### 在 GPU 上运行模型
- 设备 OS 为 Android 或 ARM Linux 时，通过使用`--backend=opencl,arm`来实现
- 设备 OS 为 macOS(arm 芯片，如 m1) 时，通过使用`--backend=opencl,arm`来实现, 只支持精度为fp32，需设置`--gpu_precision=fp32`
- 设备 OS 为 macOS(x86 芯片) 时，通过使用`--backend=opencl,x86`来实现

说明：
- 由于 Linux 上运行 OpenCL 必须提前预装 OpenCL 相关驱动库，因此暂不支持使用 Linux 系统上的 GPU 执行模型推理预测
- 当指定在 GPU 上运行模型时，有如下 4 个重要运行时参数，不同设置会对性能有较大影响：
  - `--opencl_cache_dir`：设置 opencl cache 文件的存放路径，当显式设置该选项后，会开启 opencl kernel 预编译 和 auto-tune 功能
  - `--opencl_kernel_cache_file`：设置 opencl kernel cache 文件名字
  - `--opencl_tuned_file`：设置 opencl auto-tune 文件名字
  - `--opencl_tune_mode`：设置 opencl auto-tune 模式

比如在 Android 设备上使用 GPU 运行模型时，推荐使用：
```shell
adb shell "cd /data/local/tmp/benchmark;
  ./benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=opencl,arm \
    --opencl_cache_dir=/data/local/tmp \
    --opencl_kernel_cache_file=MobileNetV1_kernel.bin \
    --opencl_tuned_file=MobileNetV1_tuned.bin"
```

### 在 NNAdapter 上运行模型
在 NNAdapter 上运行模型，需配置三个重要参数：
- `--backend`：设置模型运行时的后端，支持 NNAdapter 与 x86、ARM 组合进行异构计算
- `--nnadapter_device_names`：设置 NNAdapter 的实际新硬件后端
- `--nnadapter_context_properties`：设置新硬件硬件资源（目前仅在 Huawei Ascend NPU 上使用）

#### 运行前的数据准备
##### 步骤 1：编译 benchmark_bin
- Huawei Kirin NPU / Mediatek NPU / Amlogic NPU(S905D3 Android 版本) 请参考 『在 Android 上运行性能测试』进行编译。
- Huawei Ascend NPU（arm host） / Rockchip NPU / Imagination NNA / Amlogic NPU(C308X 或 A311D) 请参考 『在 ARM Linux 上运行性能测试』进行编译。
- Huawei Ascend NPU（x86 host）请参考『在 Linux 上运行性能测试』进行编译。

编译完成后，会生成`build.lite.*./lite/api/tools/benchmark/benchmark_bin`二进制文件。

##### 步骤 2：编译 NNAdapter 运行时库与 NNAdapter Device HAL 库
请参考下表编译指南，编译 NNAdapter 运行时库及 NNAdapter Device HAL 库

|No.| 新硬件名称 | Device HAL 库名称|编译指南 |
|---|---|---|---|
|1|Huawei Kirin NPU|libhuawei_kirin_npu.so| [点击进入](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/huawei_kirin_npu.html) |
|2|Huawei Ascend NPU|libhuawei_ascend_npu.so| [点击进入](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/huawei_ascend_npu.html) |
|3|Rockchip NPU|librockchip_npu.so| [点击进入](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/rockchip_npu.html) |
|4|Imagination NNA|libimagination_nna.so| [点击进入](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/imagination_nna.html) |
|5|Mediatek APU|libmediatek_apu.so| [点击进入](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/mediatek_apu.html) |
|6|Amlogic NPU|libamlogic_npu.so| [点击进入](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/amlogic_npu.html)|

编译完成后，NNAdapter 运行时库和 Device HAL 库将会生成在`build.lite*/inference_lite_lib*/cxx/lib/`目录下。

##### 步骤 3：获取新硬件 DDK
请下载 [Paddle Lite 通用示例程序](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，并参照下表路径，获取新硬件所需的 DDK。

|No.| 新硬件名称 | DDK 路径 |
|---|---|---|
|1|Huawei Kirin NPU| PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/huawei_kirin_npu<br>PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/huawei_kirin_npu |
|2|Huawei Ascend NPU| PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/huawei_ascend_npu<br>PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/huawei_ascend_npu |
|3|Rockchip NPU| PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/rockchip_npu<br>PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/rockchip_npu |
|4|Imagination NNA| PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/imagination_nna |
|5|Mediatek APU| PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/mediatek_apu |
|6|Amlogic NPU| PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/amlogic_npu<br>PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/amlogic_npu|

##### 步骤 4：拷贝数据到新硬件设备
将 `benchmark_bin` 及所需动态库全部拷入新硬件设备后，即可开始运行模型并获得性能数据。
- 对于 Android 设备，我们建议您将全部数据放在`/data/local/tmp/benchmark`目录下
- 对于 Linux 设备，我们建议您将全部数据放在`~/benchmark`目录下

为方便后续命令的表示，我们做以下约定：
- 用户已在构建机器的`~/benchmark`路径下归档好包含 `benchmark_bin`、`NNAdapter 运行时库`、`NNAdapter Device HAL 库`、`新硬件 DDK`、`Paddle 模型文件`在内的全部数据。

#### 在 Huawei Kirin NPU 上运行模型
```shell
# 拷贝 benchmark 文件夹到新硬件
adb shell "rm -rf /data/local/tmp/benchmark"
adb shell "mkdir /data/local/tmp/benchmark"
adb push ~/benchmark/* /data/local/tmp/benchmark
# 设置环境变量并运行模型
adb shell "cd /data/local/tmp/benchmark;
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH;
./benchmark_bin \
  --model_file=MobileNetV1/inference.pdmodel \
  --param_file=MobileNetV1/inference.pdiparams \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=20 \
  --backend=nnadapter,arm \
  --nnadapter_device_names=huawei_kirin_npu"
```

#### 在 Huawei Ascend NPU 上运行模型
```shell
# Host 侧为 x86 CPU 时
# 拷贝 benchmark 文件夹到新硬件
ssh name@ip "rm -rf ~/benchmark"
scp -r ~/benchmark name@ip:~
ssh name@ip
cd ~/benchmark
# 设置环境变量
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
# 运行模型
./benchmark_bin \
  --model_file=MobileNetV1/inference.pdmodel \
  --param_file=MobileNetV1/inference.pdiparams \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=20 \
  --backend=nnadapter,x86 \
  --nnadapter_device_names=huawei_ascend_npu \
  --nnadapter_context_properties="HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0"

# Host 侧为 ARM CPU 时
# 拷贝 benchmark 文件夹到新硬件
ssh name@ip "rm -rf ~/benchmark"
scp -r ~/benchmark name@ip:~
ssh name@ip
cd ~/benchmark
# 设置环境变量
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
# 运行模型
./benchmark_bin \
  --model_file=MobileNetV1/inference.pdmodel \
  --param_file=MobileNetV1/inference.pdiparams \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=20 \
  --backend=nnadapter,arm \
  --nnadapter_device_names=huawei_ascend_npu \
  --nnadapter_context_properties="HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0"
```

#### 在 Rockchip NPU 上运行模型
```shell
# 拷贝 benchmark 文件夹到新硬件
ssh name@ip "rm -rf ~/benchmark"
scp -r ~/benchmark name@ip:~
ssh name@ip
cd ~/benchmark
# 设置环境变量
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
# 运行模型
./benchmark_bin \
  --uncombined_model_dir=./mobilenet_v1_int8_224_per_layer \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=20 \
  --backend=nnadapter,arm \
  --nnadapter_device_names=rockchip_npu
```

#### 在 Imagination NNA 上运行模型
```shell
# 拷贝 benchmark 文件夹到新硬件
ssh name@ip "rm -rf ~/benchmark"
scp -r ~/benchmark name@ip:~
ssh name@ip
cd ~/benchmark
# 设置环境变量
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
# 运行模型
./benchmark_bin \
  --uncombined_model_dir=./mobilenet_v1_int8_224_per_layer \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=20 \
  --backend=nnadapter,arm \
  --nnadapter_device_names=imagination_nna
```

#### 在 Mediatek APU 上运行模型
```shell
# 拷贝 benchmark 文件夹到新硬件
adb shell "rm -rf /data/local/tmp/benchmark"
adb shell "mkdir /data/local/tmp/benchmark"
adb push ~/benchmark/* /data/local/tmp/benchmark
# 设置环境变量并运行模型
adb shell "cd /data/local/tmp/benchmark;
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH;
./benchmark_bin \
  --uncombined_model_dir=./mobilenet_v1_int8_224_per_layer \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=20 \
  --backend=nnadapter,arm \
  --nnadapter_device_names=mediatek_apu"
```

#### 在 Amlogic NPU 上运行模型
```shell
# 在 C308X 或 A311D 上运行模型
# 拷贝 benchmark 文件夹到新硬件
ssh name@ip "rm -rf ~/benchmark"
scp -r ~/benchmark name@ip:~
ssh name@ip
cd ~/benchmark
# 设置环境变量
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
# 运行模型
./benchmark_bin \
  --uncombined_model_dir=./mobilenet_v1_int8_224_per_layer \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=20 \
  --backend=nnadapter,arm \
  --nnadapter_device_names=amlogic_npu

# 在 S905D3 上运行模型
# 拷贝 benchmark 文件夹到新硬件
adb shell "rm -rf /data/local/tmp/benchmark"
adb shell "mkdir /data/local/tmp/benchmark"
adb push ~/benchmark/* /data/local/tmp/benchmark
# 设置环境变量并运行模型
adb shell "cd /data/local/tmp/benchmark;
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH;
./benchmark_bin \
  --uncombined_model_dir=./mobilenet_v1_int8_224_per_layer \
  --input_shape=1,3,224,224 \
  --warmup=10 \
  --repeats=20 \
  --backend=nnadapter,arm \
  --nnadapter_device_names=amlogic_npu"
```

### 逐层耗时和精度分析
当在编译时设置`--with_profile=ON`时，运行`benchmark_bin`时会输出模型每层的耗时信息；
当在编译时设置`--with_precision_profile=ON`时，运行`benchmark_bin`时会输出模型每层的精度信息。具体可以参见 [Profiler 工具](../user_guides/profiler)。
