
# 源码编译方法

PaddleLite已经提供官方Release预测库下载，请参考[文档](release_lib)。

PaddleLite 提供了移动端的一键源码编译脚本 `lite/tools/build.sh`，编译流程如下：

1. 环境准备（选择其一）：Docker交叉编译环境、Linux交叉编译环境
2. 编译：调用`build.sh`脚本一键编译

## 一、环境准备

目前支持四种编译的环境：

1. Docker 容器环境，
2. Linux（推荐 Ubuntu 16.04）环境，
3. Mac OS 环境，
4. [Windows 环境](../demo_guides/x86.html#windows)

### 1、 Docker开发环境

[Docker](https://www.docker.com/) 是一个开源的应用容器引擎, 使用沙箱机制创建独立容器，方便运行不同程序。Docker初学者可以参考[Docker使用方法](https://thenewstack.io/docker-station-part-one-essential-docker-concepts-tools-terminology/)正确安装Docker。

#### 准备Docker镜像

有两种方式准备Docker镜像，推荐从Dockerhub直接拉取Docker镜像

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

在拉取Paddle-Lite仓库代码的上层目录，执行如下代码，进入Docker容器：

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

#### Android

##### 交叉编译环境要求

- gcc、g++、git、make、wget、python、adb
- Java environment
- cmake（建议使用3.10或以上版本）
- Android NDK (建议ndk-r17c)

##### 具体步骤

安装软件部分以 Ubuntu 为例，其他 Linux 发行版类似。

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
#     recommand android-ndk-r17c-darwin-x86_64
#     ref: https://developer.android.com/ndk/downloads
cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip

# 5. Add environment ${NDK_ROOT} to `~/.bashrc` 
echo "export NDK_ROOT=/opt/android-ndk-r17c" >> ~/.bashrc
source ~/.bashrc
```

#### ARM Linux

适用于基于 ARMv8 和 ARMv7 架构 CPU 的各种开发板，例如 RK3399，树莓派等，目前支持交叉编译和本地编译两种方式，对于交叉编译方式，在完成目标程序编译后，可通过 scp 方式将程序拷贝到开发板运行。

##### 交叉编译

###### 编译环境要求

- gcc、g++、git、make、wget、python、scp
- cmake（建议使用3.10或以上版本）

###### 具体步骤

安装软件部分以 Ubuntu 为例，其他 Linux 发行版类似。

```shell
# 1. Install basic software
apt update
apt-get install -y --no-install-recommends \
  gcc g++ git make wget python unzip

# 2. Install arm gcc toolchains
apt-get install -y --no-install-recommends \
  g++-arm-linux-gnueabi gcc-arm-linux-gnueabi \
  g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu 

# 3. Install cmake 3.10 or above
wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
    tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
    mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 && \  
    ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
    ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake
```

##### 本地编译（直接在RK3399或树莓派上编译）

###### 编译环境要求

- gcc、g++、git、make、wget、python、pip、python-dev、patchelf
- cmake（建议使用3.10或以上版本）

###### 具体步骤

安装软件部分以 Ubuntu 为例，其他 Linux 发行版本类似。

```shell
# 1. Install basic software
apt update
apt-get install -y --no-install-recomends \
  gcc g++ make wget python unzip patchelf python-dev

# 2. install cmake 3.10 or above
wget https://www.cmake.org/files/v3.10/cmake-3.10.3.tar.gz
tar -zxvf cmake-3.10.3.tar.gz
cd cmake-3.10.3
./configure
make
sudo make install
```

之后可通过cmake --version查看cmake是否安装成功。

至此，完成 Linux 交叉编译环境的准备。

### 3、Mac OS 开发环境

#### 交叉编译环境要求

- gcc、git、make、curl、unzip、java
- cmake（Android编译请使用3.10版本，IOS编译请使用3.15版本）
- 编译Android: Android NDK (建议ndk-r17c)
- 编译IOS: XCode(Version 10.1)

#### 具体步骤

```bash
# 1. Install basic software
brew install  curl gcc git make unzip wget

# 2. Install cmake: mac上实现IOS编译和Android编译要求的cmake版本不一致,可以根据需求选择安装。
# （1）在mac环境编译 Paddle-Lite 的Android版本，需要安装cmake 3.10
#     mkdir /usr/local/Cellar/cmake/ && cd /usr/local/Cellar/cmake/
#     wget https://cmake.org/files/v3.10/cmake-3.10.2-Darwin-x86_64.tar.gz
#     tar zxf ./cmake-3.10.2-Darwin-x86_64.tar.gz
#     mv cmake-3.10.2-Darwin-x86_64/CMake.app/Contents/ ./3.10.2
#     ln -s /usr/local/Cellar/cmake/3.10.2/bin/cmake /usr/local/bin/cmake
# （2）在mac环境编译 Paddle-Lite 的IOS版本，需要安装cmake 3.15
#     mkdir /usr/local/Cellar/cmake/ && cd /usr/local/Cellar/cmake/
#     cd /usr/local/Cellar/cmake/
#     wget https://cmake.org/files/v3.15/cmake-3.15.2-Darwin-x86_64.tar.gz
#     tar zxf ./cmake-3.15.2-Darwin-x86_64.tar.gz
#     mv cmake-3.15.2-Darwin-x86_64/CMake.app/Contents/ ./3.15.2
#     ln -s /usr/local/Cellar/cmake/3.15.2/bin/cmake /usr/local/bin/cmake

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

# 6. 编译IOS需要安装XCode(Version 10.1)，可以在App Store里安装。安装后需要启动一次并执行下面语句。
# sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

至此，完成 Mac 交叉编译环境的准备。

**注意**: Mac上编译Paddle-Lite的full_publish版本时，Paddle-Lite所在路径中不可以含有中文字符

## 二、编译PaddleLite

`develop分支`和`release/v2.6.0`之后版本的源码编译请参考以下说明，release/v2.3之前版本（包括v2.3）源码编译请参考[release/v2.3源码编译方法](./Compile/v2.3_compile)。

### Android 预测库编译方法

Paddle-Lite支持在 “Docker 环境、Linux 环境、Mac 环境” 源码编译Android 预测库

**编译方法参见**：[Android预测库编译方法](./Compile/Android)


### iOS 预测库编译方法

Paddle-Lite只支持在 “Mac 环境” 源码编译iOS 预测库

**编译方法参见**：[iOS预测库编译方法](./Compile/iOS)


### Linux 预测库编译方法

**编译方法参见**：[Linux预测库编译方法](./Compile/Linux)


### 加速第三方依赖库的下载

如出现源码编译耗时过长，一般是第三方库下载过慢或失败导致：

- 移动端相关编译所需的第三方库均位于 `<PaddleLite>/third-party` 目录下，默认编译过程中，会利用`git submodule update --init --recursive`链上相关的第三方依赖的仓库。

- 为加速`full_publish`、`test`编译模式中对`protobuf`等第三方依赖的下载，`build.sh` 和 `ci_build.sh`支持了从国内 CDN 下载第三方依赖的压缩包。

可使用本节方法加速第三方库下载过程，以加速编译：

- **加速方法**：`git clone`完`Paddle-Lite`仓库代码后，手动删除本地仓库根目录下的`third-party`目录：

```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
git checkout <release-version-tag>
cd Paddle-Lite
rm -rf third-party
```

之后再根据本文档，进行后续编译时，便会忽略第三方依赖对应的`submodule`，改为直接下载第三方压缩包。
