# Android开发文档

用户可通过如下两种方式进行编译:

- 基于macOS 、Linux交叉编译
- 基于Docker容器编译

## 基于macOS 、Linux交叉编译

需要: NDK17及以上、cmake 3.0及以上

### 执行编译

在paddle-mobile根目录中，执行以下命令：

```shell

cd tools
sh build.sh android

# 如果想编译只支持某些特定网络的库 (可以控制包体积, 编译出来的库就只包含了支持这些特定模型的算子), 可以使用

sh build.sh android  mobilenet googlenet

# 当然这些网络是需要在 cmakelist  中配置的(https://github.com/PaddlePaddle/paddle-mobile/blob/73769e7d05ef4820a115ad3fb9b1ca3f55179d03/CMakeLists.txt#L216), 目前配置了几个常见模型

```

执行完毕后，生成的`so`位于`build/release/`目录中：  

- jni 头文件位于 [https://github.com/PaddlePaddle/paddle-mobile/tree/develop/src/io/jni](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/src/io/jni)  
- c++ 头文件位于 [https://github.com/PaddlePaddle/paddle-mobile/blob/develop/src/io/paddle_inference_api.h](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/src/io/paddle_inference_api.h)   

单测可执行文件位于`test/build`目录中。

如果有环境问题, 可以看接下来的环节

### 环境配置

##### 下载Android NDK

如果你的电脑安装了Android Studio, 可以在 Android Studio 中直接下载安装`NDK`或者可以在 [https://developer.android.com/ndk/](https://developer.android.com/ndk/) 这里自行下载，也可以通过以下命令获取：

- Mac平台

```shell
wget https://dl.google.com/android/repository/android-ndk-r17b-darwin-x86_64.zip
unzip android-ndk-r17b-darwin-x86_64.zip
```

- Linux平台

```shell
wget https://dl.google.com/android/repository/android-ndk-r17b-linux-x86_64.zip
unzip android-ndk-r17b-linux-x86_64.zip
```

##### 设置环境变量
工程中自带的独立工具链会根据环境变量`NDK_ROOT`查找NDK，因此需要配置环境变量：

```shell
export NDK_ROOT = "path to ndk"
```

##### 安装 CMake

- Mac平台

mac 平台下可以使用`homebrew`安装

```shell
brew install cmake
```

- Linux平台

linux 下可以使用`apt-get`进行安装

```shell
apt-get install cmake

```

##### Tips:
如果想要获得体积更小的库，可选择编译支持指定模型结构的库。
如执行如下命令：

```shell
sh build.sh android googlenet
```

会得到一个支持googlnet的体积更小的库。

## 基于Docker容器编译

### 1. 安装 docker

安装 docker 的方式，参考官方文档 [https://docs.docker.com/install/](https://docs.docker.com/install/)

### 2. 使用 docker 搭建构建环境

首先进入 paddle-mobile 的目录下，执行 `docker build`
以 Linux/Mac 为例 (windows 建议在 'Docker Quickstart Terminal' 中执行)

```shell
$ docker build -t paddle-mobile:dev - < Dockerfile
```
使用 `docker images` 可以看到我们新建的 image

```shell
$ docker images
REPOSITORY      TAG     IMAGE ID       CREATED         SIZE
paddle-mobile   dev     33b146787711   45 hours ago    372MB
```
### 3. 使用 docker 构建
进入 paddle-mobile 目录，执行 docker run

```shell
$ docker run -it --mount type=bind,source=$PWD,target=/paddle-mobile paddle-mobile:dev
root@5affd29d4fc5:/ # cd /paddle-mobile
# 生成构建 android 产出的 Makefile
root@5affd29d4fc5:/ # rm CMakeCache.txt
root@5affd29d4fc5:/ # cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-android-neon.cmake
# 生成构建 linux 产出的 Makefile
root@5affd29d4fc5:/ # rm CMakeCache.txt
root@5affd29d4fc5:/ # cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-linux-gnueabi.cmake
```
### 4. 设置编译选项

可以通过 ccmake 设置编译选项

```
root@5affd29d4fc5:/ # ccmake .
                                                     Page 1 of 1
 CMAKE_ASM_FLAGS
 CMAKE_ASM_FLAGS_DEBUG
 CMAKE_ASM_FLAGS_RELEASE
 CMAKE_BUILD_TYPE
 CMAKE_INSTALL_PREFIX             /usr/local
 CMAKE_TOOLCHAIN_FILE             /paddle-mobile/tools/toolchains/arm-android-neon.cmake
 CPU                              ON
 DEBUGING                         ON
 FPGA                             OFF
 LOG_PROFILE                      ON
 MALI_GPU                         OFF
 NET                              googlenet
 USE_EXCEPTION                    ON
 USE_OPENMP                       OFF
```
修改选项后，按 `c`, `g` 更新 Makefile
### 5. 构建
使用 make 命令进行构建

```
root@5affd29d4fc5:/ # make
```
### 6. 查看构建产出

构架产出可以在 host 机器上查看，在 paddle-mobile 的目录下，build 以及`test/build`下，可以使用`adb`指令或`scp`传输到`device`上执行

## 测试

在编译完成后，我们提供了自动化的测试脚本，帮助用户将运行单测文件所需要的模型及库文件push到Android设备

执行下面的脚本，该脚本会下载测试需要的 [mobilenet和test_image_1x3x224x224_float(预处理过的 NCHW 文件) 文件](http://mms-graph.bj.bcebos.com/paddle-mobile/opencl_test_src.zip)，在项目下的`test`目录创建模型和图片文件夹，并将`mobilenet`复制到`paddle-mobile/test/models`目录下，将`test_image_1x3x224x224_float`复制到`paddle-mobile/test/images`目录下


```shell
cd tools
sh ./prepare_images_and_models.sh
```

* 执行下面命令将可执行文件和预测需要的文件部署到手机

```shell
cd tools/android-debug-script
sh push2android.sh
```

* mobilenet cpu模型预测结果

假设mobilenet和`test_image_1x3x224x224_float`文件已经推送到手机上，执行下面命令进行mobilenet cpu的预测

```shell
adb shell
cd /data/local/tmp/bin/
export LD_LIBRARY_PATH=.
./test-mobilenet
```
