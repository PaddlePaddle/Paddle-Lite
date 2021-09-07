# 项目构建（Android）

## 一、简介

本文介绍如何将 Paddle Lite 源代码通过 Android NDK 交叉构建适用于 Android 平台的发布包。

说明：本文适用于 Paddle Lite v2.6 及以上版本，面向对源代码有修改需求的开发者。如果您需要的是 Paddle Lite 正式版本，请直接前往下载我们预先编译的发布包。

## 二、环境

### 2.1 Linux 开发环境

#### 2.1.1 环境要求

- gcc、g++、git、make、wget、python、adb
- Java Environment
- CMake（请使用 3.10 或以上版本）
- Android NDK（支持 ndk-r17c 及之后的所有 NDK 版本, 注意从 ndk-r18 开始，NDK 交叉编译工具仅支持 Clang, 不支持 GCC）

#### 2.1.2 安装命令

以 Ubuntu 为例，安装命令如下：

```shell
# 1. Install basic software
apt update
apt-get install -y --no-install-recommends \
  gcc g++ git make wget python unzip adb curl

# 2. Prepare Java env.
apt-get install -y default-jdk

# 3. Install cmake 3.10 or above
wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
    tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
    mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 && \  
    ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
    ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake

# 4. Download Android NDK for linux-x86_64
#     Note: Skip this step if NDK installed
#     support android-ndk-r17c-linux-x86_64 and other later version such as ndk-r18b or ndk-r20b 
#     ref: https://developer.android.com/ndk/downloads
cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip

# 5. Add environment ${NDK_ROOT} to `~/.bashrc` 
echo "export NDK_ROOT=/opt/android-ndk-r17c" >> ~/.bashrc
source ~/.bashrc

# Note: To other ndk version, the step is similar to the above.
# Take android-ndk-r20b-linux-x86_64 as example:
cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip
cd /opt && unzip /tmp/android-ndk-r20b-linux-x86_64.zip
echo "export NDK_ROOT=/opt/android-ndk-r20b" >> ~/.bashrc
source ~/.bashrc
```

其它 Linux 发行版安装步骤类似，在此不再赘述。

### 2.2 Mac OS 开发环境

#### 2.2.1 环境要求

- gcc、git、make、curl、unzip、java
- CMake（请使用 3.10 版本）
- Android NDK（支持 ndk-r17c 及之后的所有 NDK 版本, 注意从 ndk-r18 开始，NDK 交叉编译工具仅支持 Clang, 不支持 GCC）


#### 2.1.2 安装命令


```bash
# 1. Install basic software
brew install curl gcc git make unzip wget

# 2. Install cmake
mkdir /usr/local/Cellar/cmake/ && cd /usr/local/Cellar/cmake/
wget https://cmake.org/files/v3.10/cmake-3.10.2-Darwin-x86_64.tar.gz
tar zxf ./cmake-3.10.2-Darwin-x86_64.tar.gz
mv cmake-3.10.2-Darwin-x86_64/CMake.app/Contents/ ./3.10.2
ln -s /usr/local/Cellar/cmake/3.10.2/bin/cmake /usr/local/bin/cmake

# 3. Download Android NDK for Mac
#    recommand android-ndk-r17c-darwin-x86_64
#    ref: https://developer.android.com/ndk/downloads
#    Note: Skip this step if NDK installed
cd ~/Documents && curl -O https://dl.google.com/android/repository/android-ndk-r17c-darwin-x86_64.zip
cd ~/Library && unzip ~/Documents/android-ndk-r17c-darwin-x86_64.zip

# 4. Add environment ${NDK_ROOT} to `~/.bash_profile` 
echo "export NDK_ROOT=~/Library/android-ndk-r17c" >> ~/.bash_profile
source ~/.bash_profile

# 5. Install Java Environment 
brew cask install java

```

## 三、构建

### 3.1 构建步骤

运行编译脚本之前，请先检查系统环境变量 `NDK_ROOT` 指向正确的 Android NDK 安装路径，之后可以下载并编译 Paddle-Lite 源码。

```shell
# 1. 下载 Paddle-Lite 源码并切换到 release 分支
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite && git checkout 2.9

# (可选) 删除此目录，编译脚本会自动从国内CDN下载第三方库文件
# rm -rf third-party

# 2. 编译 Paddle-Lite Android 预测库
./lite/tools/build_android.sh
```

**提示：** 编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在 git clone 完 Paddle-Lite 仓库代码后，手动删除本地仓库根目录下的 third-party 目录。编译脚本会自动下载存储于国内 CDN 的第三方依赖的压缩包，节省从 git repo 同步第三方库代码的时间。

### 3.2 构建参数

```shell
--arch: (armv8|armv7)        arm版本，默认为armv8
--toolchain: (gcc|clang)     编译器类型，默认为gcc
--android_stl: (c++_static|c++_shared)   NDK stl库链接方法，默认为静态链接c++_static
--with_java: (OFF|ON)        是否编译Java预测库, 默认为 ON
--with_cv: (OFF|ON)          是否编译CV相关预处理库, 默认为 OFF
--with_log: (OFF|ON)         是否输出日志信息, 默认为 ON
--with_exception: (OFF|ON)   是否在错误发生时抛出异常，默认为 OFF   
--with_extra: (OFF|ON)       是否编译OCR/NLP模型相关kernel&OP，默认为OFF，只编译CV模型相关kernel&OP
--android_api_level: (num)   指定编译时支持的最低Android API Level，默认为Default
```

Paddle-Lite 默认支持的最低安卓版本如下表所示，使用者可以通过`--android_api_level`选项设定一个具体的数值，该数值应不低于下表中最低支持的 Android API Level。

| Paddle-Lite Requird / ARM ABI                | armv7 | armv8 |
| :-- | :-- | :-- |
| Supported Minimum Android API Level          |  16   |  21   |
| Supported Minimum Android Platform Version   |  4.1  |  5.0  |

### 3.3 更多信息

- 根据模型包含算子进行预测库裁剪，请参考 [裁剪预测库](https://paddle-lite.readthedocs.io/zh/latest/source_compile/library_tailoring.html)。
- 编译异构设备的 Android 预测库，请参考 [部署示例](https://paddle-lite.readthedocs.io/zh/latest/index.html)。


## 四、验证

位于 `Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8`:

```shell
inference_lite_lib.android.armv8/
├── cxx                                               C++ 预测库和头文件
│   ├── include                                       C++ 头文件
│   │   ├── paddle_api.h
│   │   ├── paddle_image_preprocess.h
│   │   ├── paddle_lite_factory_helper.h
│   │   ├── paddle_place.h
│   │   ├── paddle_use_kernels.h
│   │   ├── paddle_use_ops.h
│   │   └── paddle_use_passes.h
│   └── lib                                           C++ 预测库
│       ├── libpaddle_api_light_bundled.a             C++ 静态库
│       └── libpaddle_light_api_shared.so             C++ 动态库
│
├── java                                              Java 预测库
│   ├── jar
│   │   └── PaddlePredictor.jar                       Java JAR 包
│   ├── so
│   │   └── libpaddle_lite_jni.so                     Java JNI 动态链接库
│   └── src
│
└── demo                                              C++ 和 Java 示例代码
    ├── cxx                                           C++ 预测库demo
    └── java                                          Java 预测库demo
```
