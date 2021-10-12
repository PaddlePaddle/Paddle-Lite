# 概述
当我们已经有一个 Paddle 格式的模型后，我们可以使用 Benchmark 工具对该模型进行性能测试。Benchmark 工具可以输出的性能指标包括但不限于：
- 初始化耗时
- 首帧耗时
- 平均耗时

Benchmark 工具的详细功能包括但不限于：
- 同时支持 Paddle combined / uncombined 格式模型作为输入模型
- 支持单输入和多输入模型
- 支持从文本读取输入数据
- 支持设置不同的运行时精度
- 支持时间 profile 和精度 profile

# 适用场景
Benchmark 工具可方便快捷地评测给定模型在如下硬件上运行时的性能：
- 安卓系统下的 ARM CPU / GPU
- Linux 系统下的 X86 CPU / ARM CPU / ARM GPU
- OSX 系统下的 CPU / GPU

备注：本工具正在支持对运行在 M1 芯片上的模型进行性能测试

# 在 Android 上运行性能测试
## 编译
根据[源码编译](../source_compile/compile_env)准备编译环境，建议使用 Docker 配置交叉编译环境。
拉取 [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 代码，切换到特定分支，然后在 Paddle-Lite 根目录下执行编译命令：
```
./lite/tools/build_android.sh --toolchain=clang --with_benchmark=ON full_publish
```
可选参数：

| 参数 | 说明 | 可选值 | 默认值 |
| :-- | :-- | :-- | :-- |
| arch                  | 目标 ARM 架构    |  armv7 / armv8   |  armv8   |
| toolchain             | 工具链           |  gcc / clang     |  gcc     |
| with_profile          | 逐层时间 profile |  ON / OFF        |  OFF     |
| with_precision_profile| 逐层精度 profile |  ON / OFF        |  OFF     |

编译完成后，会生成`build.lite.*./lite/api/benchmark_bin`二进制文件。

## 运行
需要将如下文件通过`adb`上传至手机：
- Paddle 模型（combined 或 uncombined 格式均可）或已经`opt`工具离线优化后的`.nb`文件
- 二进制文件`benchmark_bin`

在 Host 端机器上操作例子如下：
```
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 上传文件
adb shell mkdir /data/local/tmp/benchmark
adb push MobileNetV1 /data/local/tmp/benchmark
adb push build.lite.android.armv8.clang/lite/api/benchmark_bin /data/local/tmp/benchmark

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
```
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

# 在 ARMLinux 上运行性能测试
根据[源码编译](../source_compile/compile_env)准备编译环境，建议使用 Docker 配置交叉编译环境。
拉取 [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 代码，切换到特定分支，然后在 Paddle-Lite 根目录下执行编译命令：
```
./lite/tools/build_linux.sh --arch=armv8 --with_benchmark=ON full_publish
```
可选参数：

| 参数 | 说明 | 可选值 | 默认值 |
| :-- | :-- | :-- | :-- |
| arch                  | 目标 ARM 架构    |  armv7 / armv8   |  armv8   |
| toolchain             | 工具链           |  gcc / clang     |  gcc     |
| with_profile          | 逐层时间 profile |  ON / OFF        |  OFF     |
| with_precision_profile| 逐层精度 profile |  ON / OFF        |  OFF     |

编译完成后，会生成`build.lite.*./lite/api/benchmark_bin`二进制文件。

## 运行
需要将如下文件通过`scp`或其他方式上传至 armlinux 设备：
- Paddle 文件（combined 或 uncombined 格式均可）或已经`opt`工具离线优化后的`.nb`文件
- 二进制文件`benchmark_bin`

在 Host 端机器上操作例子如下：
```
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 上传文件到 armlinux 设备

```

然后通过`ssh`登录到 armlinux 设备，执行：
```
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
```
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

# 在 Linux 上运行性能测试
## 编译
根据[源码编译](../source_compile/compile_env)准备编译环境，建议使用 Docker 配置环境。
拉取 [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 代码，切换到特定分支，然后在 Paddle-Lite 根目录下执行编译命令：
```
./lite/tools/build_linux.sh --arch=x86 --with_benchmark=ON full_publish
```
可选参数：

| 参数 | 说明 | 可选值 | 默认值 |
| :-- | :-- | :-- | :-- |
| toolchain             | 工具链           |  gcc / clang     |  gcc     |
| with_profile          | 逐层时间 profile |  ON / OFF        |  OFF     |
| with_precision_profile| 逐层精度 profile |  ON / OFF        |  OFF     |

编译完成后，会生成`build.lite.*./lite/api/benchmark_bin`二进制文件。

## 运行
运行所需文件：
- Paddle 文件（combined 或 uncombined 格式均可）或已经`opt`工具离线优化后的`.nb`文件
- 二进制文件`benchmark_bin`
- `libmklml_intel.so`

在待测试的 Linux 机器上操作例子如下：
```
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 执行性能测试
./build.lite.linux.x86.gcc/lite/api/benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=x86
```
会输出如下信息：
```
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

# 在 OSX 上运行性能测试
## 编译
根据[源码编译](../source_compile/compile_env)准备编译环境，可以使用 Docker 配置环境，也可以使用系统原生开发环境。
拉取 [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 代码，切换到特定分支，然后在 Paddle-Lite 根目录下执行编译命令：
```
./lite/tools/build_macos.sh --with_benchmark=ON x86
```
可选参数：

| 参数 | 说明 | 可选值 | 默认值 |
| :-- | :-- | :-- | :-- |
| toolchain             | 工具链          |  gcc / clang     |  gcc     |
| with_profile          | 逐层时间 profile |  ON / OFF        |  OFF     |
| with_precision_profile| 逐层精度 profile |  ON / OFF        |  OFF     |

编译完成后，会生成`build.lite.*./lite/api/benchmark_bin`二进制文件。

## 运行
运行所需文件：
- Paddle 文件（combined 或 uncombined 格式均可）或已经`opt`工具离线优化后的`.nb`文件
- 二进制文件`benchmark_bin`
- `libmklml.dylib`

在 OSX 机器上操作例子如下：
```
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 设置环境变量
export LD_LIBRARY_PATH=build.lite.x86.opencl/third_party/install/mklml/lib/:$LD_LIBRARY_PATH

# 执行性能测试
./build.lite.x86.opencl/lite/api/benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=x86
```
会输出如下信息：
```
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

# 高阶用法
Benchnark 工具提供了丰富的运行时选项，来满足不同的运行时参数设置。用户可以通过在目标设备上执行`./benchmark_bin --help`获取所有选项介绍。

## 指定不同的 backend
### 在 CPU 上运行模型
- 设备 OS 为 Android 或 ARMLinux 时，通过使用`--backend=arm`来实现
- 设备 OS 为 Linux 或 OSX 时，通过使用`--backend=x86`来实现

### 在 GPU 上运行模型
- 设备 OS 为 Android 或 ARMLinux 时，通过使用`--backend=opencl,arm`来实现
- 设备 OS 为 OSX 时，通过使用`--backend=opencl,x86`来实现

说明：
- 由于 Linux 上运行 OpenCL 必须提前预装 OpenCL 相关驱动库，因此暂不支持使用 Linux 系统上的 GPU 执行模型推理预测
- 当指定在 GPU 上运行模型时，有如下 4 个重要运行时参数，不同设置会对性能有较大影响：
  - `--opencl_cache_dir`：设置 opencl cache 文件的存放路径，当显式设置该选项后，会开启 opencl kernel 预编译 和 auto-tune 功能
  - `--opencl_kernel_cache_file`：设置 opencl kernel cache 文件名字
  - `--opencl_tuned_file`：设置 opencl auto-tune 文件名字
  - `--opencl_tune_mode`：设置 opencl auto-tune 模式

比如在 Android 设备上使用 GPU 运行模型时，推荐使用：
```
adb shell "cd /data/local/tmp/benchmark;
  ./benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=opemncl,arm \
    --opencl_cache_dir=/data/local/tmp \
    --opencl_kernel_cache_file=MobileNetV1_kernel.bin \
    --opencl_tuned_file=MobileNetV1_tuned.bin"
```

### 在新硬件（）上运行模型：
持续开发中。



## 逐层耗时和精度分析
当在编译时设置`--with_profile=ON`时，运行`benchmark_bin`时会输出模型每层的耗时信息；
当在编译时设置`--with_precision_profile=ON`时，运行`benchmark_bin`时会输出模型每层的精度信息。具体可以参见[调试工具](../user_guides/debug)。
