# 使用x86 Linux环境编译Paddle Lite / 目标硬件OS为Android

## 简介
如果你的本机环境是X86架构 + Linux操作系统，需要部署模型到Android系统的目标硬件上，则可以参考本文的介绍，通过Android NDK交叉编译工具从源码构建Paddle Lite编译包，用于后续应用程序的开发。
> **说明：**
> - 通常情况下，你不需要自行从源码构建编译包，优先推荐[下载Paddle Lite官方发布的预编译包](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html)，可满足一部分场景的需求。如果官方发布的编译包未覆盖你的场景，或者需要修改Paddle Lite源代码，则可参考本文构建。
> - 本文介绍的编译方法只适用于Paddle Lite v2.6及以上版本。v2.3及之前版本请参考[release/v2.3源码编译方法](https://paddle-lite.readthedocs.io/zh/latest/source_compile/v2.3_compile.html)。

在该场景下Paddle Lite已验证的软硬件配置如下表所示：
|---| 本机环境 | 目标硬件环境 | 
|---|---|---|
|**操作系统**| Linux<br> | Android 4.1及以上（芯片版本为ARMv7时）<br> Android 5.0及以上（芯片版本为ARMv8时）| 
|**芯片层**| x86架构 | arm64-v8a/armeabi-v7a CPU <br> Huawei Kirin NPU <br>MediaTek APU <br> Amlogic NPU <br> OpenCL[^1] <br> 注：查询以上芯片支持的具体型号以及对应的手机型号，可参考[支持硬件列表](https://paddle-lite.readthedocs.io/zh/latest/introduction/support_hardware.html)章节。| 
[^1]：OpenCL是面向异构硬件平台的编译库，Paddle Lite支持在Android系统上运行基于OpenCL的程序。


## 准备编译环境
### 环境要求
 - 操作系统环境：Linux x86_64，推荐Ubuntu Linux
 - python
 - wget
 - adb
 - make
 - g++
 - unzip？
 - curl？
 - C++ 编译依赖项：
	 - GCC
	 - CMake 3.10及以上版本
	 - Git
	 - [Android NDK](https://developer.android.com/ndk/downloads) r17c 及以上版本，注意从 r18 版本开始，NDK 交叉编译工具仅支持 Clang, 不支持 GCC
 - Java 编译依赖项：
	 - Gradle？
	 - OpenJDK？
	 - Android SDK ?
### 环境安装命令
 以 Ubuntu 为例介绍安装命令。其它 Linux 发行版安装步骤类似，在此不再赘述。
 注意需要root用户权限执行如下命令。
```shell
# 1. 安装gcc g++ git make wget python unzip adb curl等基础软件
apt update
apt-get install -y --no-install-recommends \
  gcc g++ git make wget python unzip adb curl

# 2. 安装jdk
apt-get install -y default-jdk

# 3. 安装CMake，以下命令以3.10.3版本为例，其他版本步骤类似。
wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
    tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
    mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 && \  
    ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
    ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake

# 4. 下载linux-x86_64版本的Android NDK，以下命令以r17c版本为例，其他版本步骤类似。
cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip

# 5. 添加环境变量NDK_ROOT指向Android NDK的安装路径
echo "export NDK_ROOT=/opt/android-ndk-r17c" >> ~/.bashrc
source ~/.bashrc
```

## 了解基础编译参数
Paddle Lite仓库中`/lite/tools/build_android.sh`脚本文件用于构建Android版本的编译包，通过修改`build_android.sh`脚本文件中的参数，可满足不同场景编译包的构建需求，常用的基础编译参数如下表所示：
有特殊硬件需求的编译参数见后文。

| 参数 | 说明 | 可选范围 | 默认值 |
| :-- | :-- | :-- | :-- |
| arch          |  目标硬件的 ARM 架构版本   |  armv8 / armv7   |  armv8   |
| toolchain   |  C++语言的编译器工具链？Android NDK从r11开始已建议切换到clang，我们支持ndk17以上，为什么不是默认用clang呢？  |  gcc / clang |  gcc   |
| android_stl   |  链接到的 Android STL 类型  |  c++\_static / c++\_shared  |  c++\_static   |
| with_java   |  是否包含Java编译包,目标应用程序是Java语言时需配置ON?  |  OFF / ON  |  ON   |
| with_static\_lib   |  是否发布静态库，C++静态库？和前面的c++\_static什么关系？什么情况下配这个？  |  OFF / ON  |  OFF   |
| with_cv   |  是否将 cv 函数加入编译包中，业务场景是图像处理类型的（如图像分类、目标检测等）一般需要配置ON？  |  OFF / ON  |  OFF   |
| with_log   |  是否打印日志,打印的什么日志？编译过程中的日志？  |  OFF / ON |  ON   |
| with_exception   |  是否开启异常，是什么异常？打开后如果编译异常会给提示？  |  OFF / ON  |  OFF   |
| with_extra   |  是否编译完整算子（支持序列相关模型，如 OCR 和 NLP），如果配置OFF，则去掉支持序列相关模型的算子？ON的时候是哪些算子，OFF时候是哪些算子，能否有确定的指示，链接到算子表格也行。  |  OFF / ON  | OFF   |
| with_profile   |  是否打开耗时分析，分析的什么耗时，分析这个耗时做什么用？  |  OFF / ON  |  OFF   |
| with_precision\_profile   |  是否打开精度分析，分析模型的精度？如果OFF，则没办法在预测结果中查看模型的精度？  |  OFF / ON  |  OFF   |
| with_arm82\_fp16   |  是否开启半精度算子，哪些算子？什么场景会配置ON？  |  OFF / ON  |  OFF   |
| android_api\_level   |  Android API等级[^2]，在什么情况下配多少？目标硬件安卓版本支持哪些，按最低版本配置吗？比如目标应用程序想支持android6.0版本以上，则至少要配成23？  |  16～27  |  armv7:16 / armv8:21   |

[^2]Paddle Lite 支持的最低安卓版本是4.1（芯片版本为ARMv7时）或5.0（芯片版本为ARMv8时），可通过`--android_api_level`选项设定一个具体的数值，该数值应不低于下表中最低支持的 Android API Level。
| ARM ABI                | armv7 | armv8 |
| :-- | :-- | :-- |
| 支持的最低Android API等级          |  16   |  21   |
| 支持的最低Android版本   |  4.1  |  5.0  |
> **说明：**
以上参数可在下载Paddle Lite源码后直接在`build_android.sh`文件中修改，也可通过命令行指定，具体参见下面编译步骤。

## 编译步骤
运行编译脚本之前，请先检查系统环境变量 `NDK_ROOT` 指向正确的 Android NDK 安装路径。
之后可以下载并构建 Paddle Lite编译包。

```shell
# 1. 检查环境变量 `NDK_ROOT` 指向正确的 Android NDK 安装路径
？？？？

# 1. 下载 Paddle Lite 源码并切换到 release 分支
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite && git checkout 2.9

# (可选) 删除third-party目录，编译脚本会自动从国内CDN下载第三方库文件
# rm -rf third-party

# 2. 编译 Paddle-Lite Android 预测库
./lite/tools/build_android.sh
```
> **说明：**
编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在完成Paddle Lite源码下载后，删除本地仓库根目录下的 third-party 目录，编译脚本会自动下载存储于国内 CDN 的第三方依赖文件压缩包，节省从 GitHub repo 同步第三方库的时间。


## 验证编译结果

如果按`/lite/tools/build_android.sh`中的默认参数执行，成功后会在 `Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8` 生成Paddle Lite编译包，文件目录如下。

```shell
inference_lite_lib.android.armv8/
├── cxx                                               C++ 预测库和头文件
│   ├── include                                       C++ 头文件
│   │   ├── paddle_api.h
│   │   ├── paddle_image_preprocess.h
│   │   ├── paddle_lite_factory_helper.h
│   │   ├── paddle_place.h
│   │   ├── paddle_use_kernels.h
│   │   ├── paddle_use_ops.h
│   │   └── paddle_use_passes.h
│   └── lib                                           C++ 预测库
│       ├── libpaddle_api_light_bundled.a             C++ 静态库
│       └── libpaddle_light_api_shared.so             C++ 动态库
│
├── java                                              Java 预测库
│   ├── jar
│   │   └── PaddlePredictor.jar                       Java JAR 包
│   ├── so
│   │   └── libpaddle_lite_jni.so                     Java JNI 动态链接库
│   └── src
│
└── demo                                              C++ 和 Java 示例代码
    ├── cxx                                           C++ 预测库demo
    └── java                                          Java 预测库demo
```

## OpenCL

| 参数 | 说明 | 可选范围 | 默认值 |
| :-- | :-- | :-- | :-- |
| with_opencl | 是否包含 OpenCL 编译 |  OFF / ON   |  OFF   |


## 华为麒麟 NPU

- 基本参数

| 参数 | 说明 | 可选范围 | 默认值 |
| :-- | :-- | :-- | :-- |
| with\_nnadapter |  是否编译 NNAdapter  | OFF / ON |  OFF   |

- 华为麒麟 NPU

| 参数 | 说明 | 可选范围 | 默认值 |
| :-- | :-- | :-- | :-- |
| nnadapter\_with\_huawei\_kirin\_npu |  是否编译华为麒麟 NPU 的 NNAdapter HAL 库 | OFF / ON |  OFF   |
| nnadapter\_huawei\_kirin\_npu\_sdk\_root |  设置华为 HiAI DDK 目录 |  [hiai_ddk_lib_510](https://paddlelite-demo.bj.bcebos.com/devices/huawei/kirin/hiai_ddk_lib_510.tar.gz) |  空值   |
## 联发科 APU
- 基本参数

| 参数 | 说明 | 可选范围 | 默认值 |
| :-- | :-- | :-- | :-- |
| with\_nnadapter |  是否编译 NNAdapter  | OFF / ON |  OFF   |
- 联发科 APU

| 参数 | 说明 | 可选范围 | 默认值 |
| :-- | :-- | :-- | :-- |
| nnadapter\_with\_mediatek\_apu |  是否编译联发科 APU 的 NNAdapter HAL 库 | OFF / ON |  OFF   |
| nnadapter\_mediatek\_apu\_sdk\_root |  设置联发科 Neuron Adapter SDK 目录 |  [apu_ddk](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/apu_ddk.tar.gz) |  空值   |