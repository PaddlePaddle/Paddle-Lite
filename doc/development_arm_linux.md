# ARM Linux开发文档

在ARM Linux如Raspberrypi3，或Firefly-RK3399上编译paddle-mobile。

## 预先安装

```shell
$ sudo apt update
$ sudo apt-get install -y cmake git
$ git clone https://github.com/PaddlePaddle/paddle-mobile.git
```

## 编译

在paddle-mobile根目录中，执行以下命令：

```shell
# 进入paddle-mobile根目录
$ cd <your-paddle-mobile>

# 可选：开启GPU支持，在CMakeLists.txt开启GPU_CL选项为ON
$ cp /usr/lib/aarch64-linux-gnu/libMali.so ./third_party/opencl/
$ cp /usr/lib/aarch64-linux-gnu/libOpenCL.so ./third_party/opencl/
$ ln -s ./third_party/opencl/libMali.so ./third_party/opencl/

# 编译
$ cd ./tools
$ /bin/bash build.sh arm_linux
```

- 动态库`so`文件位于`<paddle-mobile-repo>/build/release/arm-linux/build`目录；  
- 单元测试位于`<paddle-model-repo>/test/build`目录，若只编译如`googlenet`，可以执行`bash build.sh arm_linux googlenet`。

## 运行

接着刚刚的命令，执行MobileNet模型：

```shell
# 导入编译好的动态库路径到LD_LIBRARY_PATH中
$ cd ../build/release/arm-linux/build
$ export LD_LIBRARY_PATH=.

# 执行MobileNet
# 可选：GPU执行./test-mobilenetgpu
$ cd ../../../../test/build/
$ ./test-mobilenet

# 执行顺利会打印如下日志
load cost :0ms
 Max element is 0.985921 at position 954
predict cost :121.462ms
如果结果Nan请查看: test/images/g_test_image_1x3x224x224_banana 是否存在?
```

注意：  
1. 如果本地仓库中`test`目录下没有模型，脚本会自动下载官方demo模型并解压；  
2. 因为ARM Linux设备算力限制，编译卡死重启机器尝试单线程编译（修改`tools/build.sh`中`build_for_arm_linux`的编译为`make -j`），或指定编译某个模型（如googlenet）或扩大系统的swap交换空间。

## 其它

- 若编译中提示有不识别的编译选项等ARM Linux平台的编译问题，可尝试修改`tools/build.sh`中的相关编译参数；  
- Android平台请参考Android开发文档.

