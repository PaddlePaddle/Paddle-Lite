# 测试方法

本文将会介绍，在**Ubuntu:16.04交叉编译环境**下，用安卓手机在终端测试Paddle-Lite的性能，并介绍两种Benchmark方法：

1. **一键Benchmark**：适用于想快速获得常见模型性能的用户，下载预编译好的benchmark可执行文件；
2. **逐步Benchmark**：将**一键Benchmark**流程拆解讲解。

## 环境准备

1. 准备[adb](https://developer.android.com/studio/command-line/adb)等必备软件：
```shell
sudo apt update
sudo apt install -y wget adb
```
2. 检查手机与电脑连接。安卓手机USB连上电脑，打开设置 -> 开启开发者模式 -> 开启USB调试 -> 允许（授权）当前电脑调试手机；
3. 在电脑终端输入`adb devices`命令，查看当前连接到的设备：
```shell
adb devices
```
命令成功执行，显示结果类似下面（序列码略有不同）：
```shell
List of devices attached
712QSDSEMMS7C   device
```

## 一. 一键Benchmark

执行以下命令，完成Benchmark：

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/run_benchmark.sh
sh run_benchmark.sh
```

该`run_benchmark.sh`脚本会：

1. 下载模型，并上传手机：包含mobilenetv1/v2、shufflenetv2、squeezenetv1.1、mnasnet；
2. 下载pre-built android-armv7和android-armv8的可执行文件，并上传手机：`benchmark_bin_v7`和`benchmark_bin_v8`；
3. 自动执行另一个脚本`benchmark.sh`（多台手机连接USB，请在`benchmark.sh`脚本中对`adb`命令后加上测试手机的`serial number`）；
4. 从手机下载benchmark结果`result_armv7.txt`和`result_armv8.txt`，到当前目录，并显示Benchmark结果。

## 二. 逐步Benchmark

### 1. 获取benchmark可执行文件

benchmark_bin文件可以测试PaddleLite的性能，有下面两种方式获得。

#### 方式一：下载benchmark_bin可执行文件

```shell
# Download benchmark_bin for android-armv7
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_bin_v7

# Download benchmark_bin for android-armv8
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_bin_v8
```

#### 方式二：由源码编译benchmark_bin文件

根据[源码编译](../user_guides/source_compile)准备编译环境，拉取PaddleLite最新release发布版代码，并在仓库根目录下，执行：

```shell
###########################################
# Build benchmark_bin for android-armv7   #
###########################################
./lite/tools/ci_build.sh  \
  --arm_os="android" \
  --arm_abi="armv7" \
  --arm_lang="gcc " \
  build_arm

# `benchmark_bin` 在: <paddle-lite-repo>/build.lite.android.armv7.gcc/lite/api/benchmark_bin

###########################################
# Build benchmark_bin for android-armv8   #
###########################################
./lite/tools/ci_build.sh  \
  --arm_os="android" \
  --arm_abi="armv8" \
  --arm_lang="gcc "  \
  build_arm

# `benchmark_bin` 在: <paddle-lite-repo>/build.lite.android.armv8.gcc/lite/api/benchmark_bin
```

> **注意**：为了避免在docker内部访问不到手机的问题，建议编译得到benchmark_bin后退出到docker外面，并且将benchmark_bin文件拷贝到一个临时目录。然后在该临时目录下，按照下面步骤下载模型、拷贝脚本、测试。

### 2. 准备模型

PaddleLite为Benchmark准备好了[常见Benchmark模型](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_models.tgz)。

执行以下命令，下载常见Benchmark模型并解压：

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_models.tgz
tar zxvf benchmark_models.tgz
```

如果测试其他模型，请将模型文件放到 `benchmark_models` 文件夹中。

### 3. benchmark.sh脚本

benchmark测试的执行脚本`benchmark.sh` 位于源码中的`/PaddleLite/lite/tools/benchmark.sh`位置，测试时需要将`benchmark.sh`、 `benchmark_bin` 、 `benchmark_models` 文件复制到同一目录下。

### 4. 测试

从终端进入benchmark.sh、可执行文件（benchmark_bin_v7、benchmark_bin_v8）和模型文件（benchmark_models）所在文件夹。

如果 `benchmark_models` 中所有模型文件都已经使用 `model_optimize_tool` 进行转换，则使用 benchmark.sh 脚本执行如下命令进行测试：

```shell
# Benchmark for android-armv7
sh benchmark.sh ./benchmark_bin_v7 ./benchmark_models result_armv7.txt

# Benchmark for android-armv8
sh benchmark.sh ./benchmark_bin_v8 ./benchmark_models result_armv8.txt
```

如果 `benchmark_models` 中所有模型文件都没有使用 `model_optimize_tool` 进行转换，则执行下面的命令。`benchmark_bin` 会首先转换模型，然后加载模型进行测试。

```shell
# Benchmark for android-armv7
sh benchmark.sh ./benchmark_bin_v7 ./benchmark_models result_armv7.txt true

# Benchmark for android-armv8
sh benchmark.sh ./benchmark_bin_v8 ./benchmark_models result_armv8.txt true
```

测试结束后，armv7和armv8的结果，分别保存在当前目录下的`result_armv7.txt`和`result_armv8.txt`文件中。

**查看测试结果**

在当前目录的`result_armv7.txt`和`result_armv8.txt`文件，查看测试结果。

> 不同手机，不同版本，测试模型的性能数据不同。

```shell
run benchmark armv8
--------------------------------------
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
mnasnet                       min = 19.83500    max = 19.38500    average = 19.65503
mobilenetv1                   min = 32.00600    max = 31.56900    average = 31.81983
mobilenetv2                   min = 22.37900    max = 22.08700    average = 22.28623
shufflenetv2                  min = 10.80400    max = 10.62900    average = 10.68890
squeezenet                    min = 17.67400    max = 17.47900    average = 17.57677

Threads=2 Warmup=10 Repeats=30
mnasnet                       min = 11.85600    max = 11.72000    average = 11.77127
mobilenetv1                   min = 18.75000    max = 18.64300    average = 18.70593
mobilenetv2                   min = 14.05100    max = 13.59900    average = 13.71450
shufflenetv2                  min = 6.67200     max = 6.58300     average = 6.63400
squeezenet                    min = 12.07100    max = 11.33400    average = 11.41253

Threads=4 Warmup=10 Repeats=30
mnasnet                       min = 7.19300     max = 7.02600     average = 7.08480
mobilenetv1                   min = 10.42000    max = 10.29100    average = 10.34267
mobilenetv2                   min = 8.61900     max = 8.46900     average = 8.54707
shufflenetv2                  min = 4.55200     max = 4.41900     average = 4.46477
squeezenet                    min = 8.60000     max = 7.85200     average = 7.98407
--------------------------------------

run benchmark armv7
--------------------------------------
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
mnasnet                       min = 20.98300    max = 20.81400    average = 20.92527
mobilenetv1                   min = 33.19000    max = 32.81700    average = 33.08490
mobilenetv2                   min = 25.91400    max = 25.61700    average = 25.73097
shufflenetv2                  min = 11.14300    max = 10.97600    average = 11.06757
squeezenet                    min = 19.31800    max = 19.20000    average = 19.26530

Threads=2 Warmup=10 Repeats=30
mnasnet                       min = 12.59900    max = 12.46600    average = 12.52207
mobilenetv1                   min = 19.05800    max = 18.94700    average = 18.97897
mobilenetv2                   min = 15.28400    max = 15.11300    average = 15.19843
shufflenetv2                  min = 6.97000     max = 6.81400     average = 6.90863
squeezenet                    min = 12.87900    max = 12.12900    average = 12.22530

Threads=4 Warmup=10 Repeats=30
mnasnet                       min = 7.31400     max = 7.12900     average = 7.20357
mobilenetv1                   min = 11.44000    max = 10.86900    average = 10.94383
mobilenetv2                   min = 9.14900     max = 9.03800     average = 9.09907
shufflenetv2                  min = 4.60600     max = 4.49400     average = 4.53360
squeezenet                    min = 8.27000     max = 8.10600     average = 8.19000
--------------------------------------
```
