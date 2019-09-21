---
layout: post
title: 源码编译指南
---

<!--ts-->
* [源码编译指南](#源码编译指南)
  * [一、环境准备](#一环境准备)
     * [1、 Docker开发环境](#1-docker开发环境)
        * [准备Docker镜像](#准备docker镜像)
        * [进入Docker容器](#进入docker容器)
        * [Docker常用命令](#docker常用命令)
     * [2、Linux 开发环境](#2linux-开发环境)
        * [交叉编译环境要求](#交叉编译环境要求)
        * [具体步骤](#具体步骤)
     * [3、Mac OS 开发环境（只支持 tiny publish 编译 ）](#3mac-os-开发环境只支持-tiny-publish-编译-)
        * [交叉编译环境要求](#交叉编译环境要求-1)
        * [具体步骤](#具体步骤-1)
  * [二、编译PaddleLite](#二编译paddlelite)
     * [下载代码](#下载代码)
     * [编译模式与参数](#编译模式与参数)
     * [编译代码](#编译代码)
     * [加速第三方依赖库的下载](#加速第三方依赖库的下载)

<!-- Added by: yanchunwei, at: Thu Aug 29 16:00:45 CST 2019 -->

<!--te-->

Paddle-Lite 提供了移动端的一键源码编译脚本 `lite/tools/build.sh`，编译流程如下：

1. 环境准备（选择其一）：Docker交叉编译环境、Linux交叉编译环境
2. 编译：调用`build.sh`脚本一键编译

## 一、环境准备

目前支持两种编译的环境：

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

### 1、 Docker开发环境

[Docker](https://www.docker.com/) 是一个开源的应用容器引擎, 使用沙箱机制创建独立容器，方便运行不同程序。Docker初学者可以参考[Docker使用方法](https://thenewstack.io/docker-station-part-one-essential-docker-concepts-tools-terminology/)正确安装Docker。

#### 准备Docker镜像

有两种方式准备Docker镜像，推荐方式一：从Dockerhub直接拉取Docker镜像

```shell
# 方式一：从Dockerhub直接拉取Docker镜像
docker pull paddlepaddle/paddle-lite:2.0.0_beta

# 方式二：本地源码编译Docker镜像
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite/lite/tools
mkdir mobile_image
cp Dockerfile.mobile mobile_image/Dockerfile
cd mobile_image
docker build -t paddlepaddle/paddle-lite .
# 镜像编译成功后，可用`docker images`命令，看到`paddlepaddle/paddle-lite`镜像。
```

#### 进入Docker容器

在拉取PaddleLite仓库代码的上层目录，执行如下代码，进入Docker容器：

```shell
docker run -it \
  --name paddlelite_docker \
  -v $PWD/Paddle-Lite:/Paddle-Lite \
  --net=host \
  paddlepaddle/paddle-lite /bin/bash
```

该命令的含义：将容器命名为`paddlelite_docker`即`<container-name>`，将当前目录下的`Paddle-Lite`文件夹挂载到容器中的`/Paddle-Lite`这个根目录下，并进入容器中。至此，完成Docker环境的准备。

#### Docker常用命令

```shell
# 退出容器但不停止/关闭容器：键盘同时按住三个键：CTRL + q + p

# 启动停止的容器
docker start <container-name>

# 从shell进入已启动的容器
docker attach <container-name>

# 停止正在运行的Docker容器
docker stop <container-name>

# 重新启动正在运行的Docker容器
docker restart <container-name>

# 删除Docker容器
docker rm <container-name>
```

### 2、Linux 开发环境

#### 交叉编译环境要求

- gcc、g++、git、make、wget、python、adb
- Java environment
- cmake（建议使用3.10或以上版本）
- Android NDK (建议ndk-r17c)

#### 具体步骤

安装软件部分以 Ubuntu 为例，其他 Linux 发行版类似。

```shell
# 1. Install basic software
apt update
apt-get install -y --no-install-recommends \
  gcc g++ git make wget python unzip adb

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
#     recommand android-ndk-r17c-darwin-x86_64
#     ref: https://developer.android.com/ndk/downloads
cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip

# 5. Add environment ${NDK_ROOT} to `~/.bashrc` 
echo "export NDK_ROOT=/opt/android-ndk-r17c" >> ~/.bashrc
source ~/.bashrc
```

至此，完成 Linux 交叉编译环境的准备。

### 3、Mac OS 开发环境（只支持 tiny publish 编译 ）

#### 交叉编译环境要求

- gcc、git、make、curl、unzip、java
- cmake（建议使用3.10或以上版本）
- 编译Android: Android NDK (建议ndk-r17c)
- 编译IOS: XCode(Version 10.1)

#### 具体步骤

```bash
# 1. Install basic software
brew install -y curl gcc git make unzip

# 2. Install cmake 3.10 or above
brew install -y cmake

# 3. Download Android NDK for Mac
#     recommand android-ndk-r17c-darwin-x86_64
#     ref: https://developer.android.com/ndk/downloads
#     Note: Skip this step if NDK installed
cd ~/Documents && curl -O https://dl.google.com/android/repository/android-ndk-r17c-darwin-x86_64.zip
cd ~/Library && unzip ~/Documents/android-ndk-r17c-darwin-x86_64.zip

# 4. Add environment ${NDK_ROOT} to `~/.bash_profile` 
echo "export NDK_ROOT=~/Library/android-ndk-r17c" >> ~/.bash_profile
source ~/.bash_profile

# 5. Install Java Environment 
brew cask install java
```

至此，完成 Mac 交叉编译环境的准备。

## 二、编译PaddleLite

### 下载代码

```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout <release-version-tag>
```

### 编译模式与参数

编译脚本`./lite/tools/build.sh`，支持三种编译模式：

| 编译模式 | 介绍 | 适用对象 |
|:-------:|-----|:-------:|
| tiny_publish | 编译移动端部署库，无第三方库依赖 | 用户 |
| full_publish | 编译移动端部署库，有第三方依赖如protobuf、glags等，含有可将模型转换为无需protobuf依赖的naive buffer格式的工具，供tiny_publish库使用 | 用户 |
| test | 编译指定`arm_os`、`arm_abi`下的移动端单元测试 | 框架开发者 |

编译脚本`./lite/tools/build.sh`，追加参数说明：

|   参数     |     介绍     |     值     |
|-----------|-------------|-------------|
| --arm_os   |必选，选择安装平台     | `android`、`ios`、`ios64`、`armlinux` |
| --arm_abi  |必选，选择编译的arm版本，其中`armv7hf`为ARMLinux编译时选用| `armv8`、`armv7`、`armv7hf`|
| --arm_lang |arm_os=android时必选，选择编译器 | `gcc`、`clang`|
| --android_stl |arm_os=android时必选，选择静态链接STL或动态链接STL | `c++_static`、`c++_shared`|
| --build_extra | 可选，是否编译控制流相关op、kernel。（**编译demo时必须选择为ON**）      | `ON`、`OFF` |
| target |必选，选择编译模式，`tiny_publish`为编译移动端部署库、`full_publish`为带依赖的移动端部署库、`test`为移动端单元测试、`ios`为编译ios端`tiny_publish` | `tiny_publish`、`full_publish`、`test`、 `ios` |

### 编译代码

注意：编译前请删除lite/api目录下的paddle_use_ops.h和paddle_use_kernels.h

#### 编译`tiny publish`动态库

##### Android
```shell
./lite/tools/build.sh \
  --arm_os=android \
  --arm_abi=armv8 \
  --arm_lang=gcc \
  --android_stl=c++_static \
  tiny_publish
```
##### IOS
```shell
./lite/tools/build.sh \
  --arm_os=ios64 \
  --arm_abi=armv8 \
  ios
```
ios tiny publish支持的编译选项：
* `--arm_os`: 可选ios或者ios64
* `--arm_abi`: 可选armv7和armv8
* 如果mac编译过程中报错："Invalid CMAKE_DEVELOPER_ROOT: does not exist", 运行：
```shell
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```
##### ARMLinux(目前只支持Docker编译)
```shell
./lite/tools/build.sh \
  --arm_os=armlinux \
  --arm_abi=armv7hf \
  --arm_lang=gcc \
  tiny_publish
```

#### 编译`full publish`动态库（**Mac OS下不支持**）

注意：编译前请删除lite/api目录下的paddle_use_ops.h和paddle_use_kernels.h

##### Android
```shell
./lite/tools/build.sh \
  --arm_os=android \
  --arm_abi=armv8 \
  --arm_lang=gcc \
  --android_stl=c++_static \
  full_publish
```
##### ARMLinux(目前只支持Docker编译)
```shell
./lite/tools/build.sh \
  --arm_os=armlinux \
  --arm_abi=armv7hf \
  --arm_lang=gcc \
  full_publish
```

### 编译结果说明

**编译最终产物位置**在 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx` ，如 Android 下 ARMv7 的产物位于：
![](https://user-images.githubusercontent.com/328693/63631174-5c53e580-c656-11e9-8726-d8cf7500a7f2.png)

**目录内容**（可能）如下：

![](https://user-images.githubusercontent.com/328693/63631178-65dd4d80-c656-11e9-804e-c091963f6dc0.png)

**具体内容**说明：

1. `bin`文件夹：可执行工具文件 `paddle_code_generator`、`test_model_bin`

2. `cxx`文件夹：包含c++的库文件与相应的头文件

- `include`  : 头文件
- `lib` : 库文件
  - 打包的静态库文件：
    - `libpaddle_api_full_bundled.a`  ：包含 full_api 和 light_api 功能的静态库
    - `libpaddle_api_light_bundled.a` ：只包含 light_api 功能的静态库
  - 打包的动态态库文件：
    - `libpaddle_full_api_shared.so` ：包含 full_api 和 light_api 功能的动态库
    - `libpaddle_light_api_shared.so`：只包含 light_api 功能的动态库

3. `demo`文件夹：示例 demo ，包含 C++ demo 和  Java demo。

- `cxx`   ： C++示例 demo
  - `mobile_full` :  full_api 的使用示例
  - `mobile_light` : light_api的使用示例
- `java`  ：Java 示例 demo
  - `android`  : Java的 Android 示例

4.  `java` 文件夹：包含 Jni 的动态库文件与相应的 Jar 包

- `jar` :  `PaddlePredictor.jar`
- `so`  : Jni动态链接库  `libpaddle_lite_jni.so`

5. `third_party` 文件夹：第三方库文件`gflags`

**注意：**

1. 只有当`--arm_os=android` 时才会编译出：

- Java库文件与示例：`Java`和`demo/java`

- 动态库文件:`libpaddle_full_api_shared.so`,`libpaddle_light_api_shared.so`

2. `tiny_publish`编译结果不包括 C++ demo和 C++ 静态库，但提供 C++ 的 light_api 动态库、 Jni 动态库和Java demo

### 加速第三方依赖库的下载

移动端相关编译所需的第三方库均位于 `<PaddleLite>/third-party` 目录下，默认编译过程中，会利用`git submodule update --init --recursive`链上相关的第三方依赖的仓库。

为加速`full_publish`、`test`编译模式中对`protobuf`等第三方依赖的下载，`build.sh` 和 `ci_build.sh`支持了从国内 CDN 下载第三方依赖的压缩包。

使用方法：`git clone`完`Paddle-Lite`仓库代码后，手动删除本地仓库根目录下的`third-party`目录：

```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
git checkout <release-version-tag>
cd Paddle-Lite
rm -rf third-party
```

之后再根据本文档，进行后续编译时，便会忽略第三方依赖对应的`submodule`，改为下载第三方压缩包。
