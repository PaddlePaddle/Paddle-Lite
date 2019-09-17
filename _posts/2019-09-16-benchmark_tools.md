---
layout: post
title: Benchmark
---

<!--ts-->
  * [Benchmark](#Benchmark)
      * [环境准备](#环境准备)
      * [1. 一键Benchmark](#一-一键benchmark)
      * [2. 逐步Benchmark](#二-逐步Benchmark)
         * [1. 获取benchmark可执行文件](#1-获取benchmark可执行文件)
         * [2. 下载模型](#2-下载模型)
         * [3. benchmark.sh脚本](#3-benchmark-sh脚本)
         * [4. 测试](#4-测试)
<!--te-->

本文将会介绍，在**Ubuntu:16.04交叉编译环境**下，用安卓手机在终端测试Paddle-Lite的性能，并介绍两种Benchmark方法：

1. **一键Benchmark**：适用于想快速获得常见模型性能的用户，下载预编译好的benchmark可执行文件；
2. **逐步Benchmark**：将**一键Benchmark**流程拆解讲解。

# 环境准备

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
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/run_benchmark.sh
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
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_bin_v7

# Download benchmark_bin for android-armv8
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_bin_v8
```

#### 方式二：由源码编译benchmark_bin文件

根据[源码编译]({{ site.baseurl }}/source_compile)准备编译环境，拉取PaddleLite最新release发布版代码，并在仓库根目录下，执行：

```shell
###########################################
# Build benchmark_bin for android-armv7   #
###########################################
./lite/tools/ci_build.sh  \
  --arm_os="android" \
  --arm_abi="armv7" \
  --arm_lang="gcc " \
  build_arm

# build result see: <paddle-lite-repo>/build.lite.android.armv7.gcc/lite/api/benchmark_bin

###########################################
# Build benchmark_bin for android-armv8   #
###########################################
./lite/tools/ci_build.sh  \
  --arm_os="android" \
  --arm_abi="armv8" \
  --arm_lang="gcc "  \
  build_arm

# build result see: <paddle-lite-repo>/build.lite.android.armv8.gcc/lite/api/benchmark_bin
```

> **注意**：为了避免在docker内部访问不到手机的问题，建议编译得到benchmark_bin后退出到docker外面，并且将benchmark_bin文件拷贝到一个临时目录。然后在该临时目录下，按照下面步骤下载模型、拷贝脚本、测试。

### 2. 下载模型

PaddleLite为Benchmark准备好了[常见Benchmark模型](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_models.tar.gz)。

执行以下命令，下载常见Benchmark模型并解压：

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_models.tar.gz
tar zxvf benchmark_models.tar.gz
```

| 模型            | 下载地址                                                        |
| --------------- | ------------------------------------------------------------ |
| MobilenetV1     | [下载](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/mobilenet_v1.tar.gz) |
| MobilenetV2     | [下载](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/mobilenet_v2.tar.gz) |
| ShufflenetV2    | [下载](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/shufflenet_v2.tar.gz) |
| Squeezenet_V1.1 | [下载](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/squeezenet_v11.tar.gz) |
| Mnasnet         | [下载](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/mnasnet.tar.gz) |

> 注：若要使用测试脚本，**对单个模型测试**，请把单个模型放入 `benchmark_models` 文件夹，并确保测试脚本、`benchmark_models`文件夹在同一级的目录。

注：上述模型都已经使用`model_optimize_tool`进行转化，而且Lite移动端只支持加载转化后的模型。如果需要测试其他模型，请先参考[模型转化方法]({{ site.baseurl }}/model_optimize_tool)。


### 3. benchmark.sh脚本

benchmark测试的执行脚本`benchmark.sh` 位于源码中的`/PaddleLite/lite/tools/benchmark.sh`位置，测试时需要将`benchmark.sh`、 `benchmark_bin` 、 `benchmark_models` 文件复制到同一目录下。

### 4. 测试

从终端进入benchmark.sh、可执行文件（benchmark_bin_v7、benchmark_bin_v8）和模型文件（benchmark_models）所在文件夹。

运行 benchmark.sh 脚本执行测试

```shell
# Benchmark for android-armv7
sh benchmark.sh ./benchmark_bin_v7 ./benchmark_models result_armv7.txt

# Benchmark for android-armv8
sh benchmark.sh ./benchmark_bin_v8 ./benchmark_models result_armv8.txt
```
测试结束后，armv7和armv8的结果，分别保存在当前目录下的`result_armv7.txt`和`result_armv8.txt`文件中。

**查看测试结果**

在当前目录的`result_armv7.txt`和`result_armv8.txt`文件，查看测试结果。

```shell
run benchmark armv7
--------------------------------------
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
-- mnasnet               avg = 159.8427 ms
-- mobilenet_v1          avg = 235.0072 ms
-- mobilenet_v2          avg = 173.0387 ms
-- shufflenet_v2         avg = 76.0040 ms
-- squeezenet_v11        avg = 164.2957 ms

Threads=2 Warmup=10 Repeats=30
-- mnasnet               avg = 83.1287 ms
-- mobilenet_v1          avg = 121.6029 ms
-- mobilenet_v2          avg = 86.6175 ms
-- shufflenet_v2         avg = 41.5761 ms
-- squeezenet_v11        avg = 87.8678 ms

Threads=4 Warmup=10 Repeats=30
-- mnasnet               avg = 73.3880 ms
-- mobilenet_v1          avg = 119.0739 ms
-- mobilenet_v2          avg = 85.3050 ms
-- shufflenet_v2         avg = 38.0762 ms
-- squeezenet_v11        avg = 64.2201 ms
--------------------------------------

run benchmark armv8
--------------------------------------
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
-- mnasnet               avg = 165.3073 ms
-- mobilenet_v1          avg = 306.0188 ms
-- mobilenet_v2          avg = 195.1884 ms
-- shufflenet_v2         avg = 99.3692 ms
-- squeezenet_v11        avg = 156.6971 ms

Threads=2 Warmup=10 Repeats=30
-- mnasnet               avg = 90.2290 ms
-- mobilenet_v1          avg = 157.0007 ms
-- mobilenet_v2          avg = 118.1607 ms
-- shufflenet_v2         avg = 68.6804 ms
-- squeezenet_v11        avg = 91.3090 ms

Threads=4 Warmup=10 Repeats=30
-- mnasnet               avg = 179.9730 ms
-- mobilenet_v1          avg = 204.0684 ms
-- mobilenet_v2          avg = 181.6486 ms
-- shufflenet_v2         avg = 123.2728 ms
-- squeezenet_v11        avg = 412.9046 ms
--------------------------------------
```
