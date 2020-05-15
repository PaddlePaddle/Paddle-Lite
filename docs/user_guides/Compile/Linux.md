
# 编译Linux预测库

**注意：本编译方法只适用于release/v2.6.0之后版本（包括 v2.6.0)**
**注意：本编译方法暂时只适用于ARM的设备**

安装了ArmLinux的编译环境，可以下载并编译 Paddle-Lite源码

```shell
# 1. 下载Paddle-Lite源码 并切换到release分支
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite && git checkout release/v2.6

# 2. 编译Paddle-Lite Android预测库 (armv8, gcc编译)
./lite/tools/build_linux.sh
```


### 编译结果

位于 `Paddle-Lite/build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8` :

```shell
inference_lite_lib.armlinux.armv8/
|-- cxx                                     C++ 预测库和头文件
|   |-- include                             C++ 头文件
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                 C++预测库
|       |-- libpaddle_api_light_bundled.a   C++静态库
|       `-- libpaddle_light_api_shared.so   C++动态库
|
|-- demo                          
|   `-- python                              python预测库demo
|
|-- python                                  Python预测库(需要打开with_python选项)
|   |-- install
|   |   `-- dist
|   |       `-- paddlelite-*.whl            python whl包 
|   |-- lib
|       `-- lite.so                         python预测库   
```


### 编译命令

- 默认编译方法: (armv8, gcc)                                           
```shell
./lite/tools/build_linux.sh
```

- 打印 help 信息：

```shell
./lite/tools/build_linux.sh help
```

- 其他可选编译命令：

```shell
--arch: (armv8|armv7|armv7hf)   arm版本，默认为armv8
--toolchain: (gcc|clang)        编译器类型，默认为gcc
--with_extra: (OFF|ON)          是否编译OCR或NLP相关模型的kernel&OP，默认为OFF，只编译CV模型相关kernel&OP
--with_python: (OFF|ON)         是否编译python预测库, 默认为 OFF
--with_cv: (OFF|ON)             是否编译CV相关预处理库, 默认为 OFF
--with_log: (OFF|ON)            是否输出日志信息, 默认为 ON
```
**注意：with_python现在仅支持armlinux的本地编译，尚不支持docker环境和ubuntu环境**

- 裁剪预测库方法（只编译模型中的kernel&OP，降低预测库体积）:

```shell
./lite/tools/build_linux.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir
```
```shell
--with_strip: (OFF|ON);   是否根据输入模型裁剪预测库，默认为OFF
--opt_model_dir:          输入模型的绝对路径，需要为opt转化之后的模型
```
详情请参考:  [裁剪预测库](https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html)


- 使用 rockchip npu 方法：

```shell
--with_rockchip_npu: (OFF|ON);   是否编译编译 huawei_kirin_npu 的预测库，默认为OFF
--rockchip_npu_sdk_root:     `rockchip_npu DDK`文件的绝对路径
```
详情请参考：[PaddleLite使用RK NPU预测部署](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/rockchip_npu.html)

- 使用 baidu xpu 方法：

```shell
--with_baidu_xpu: (OFF|ON);   是否编译编译 baidu_xpu 的预测库，默认为OFF
--baidu_xpu_sdk_root:     `baidu_xpu DDK`文件的绝对路径
```
详情请参考：[PaddleLite使用百度XPU预测部署](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/baidu_xpu.html)
