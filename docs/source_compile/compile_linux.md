
# 源码编译 (ARMLinux)

**注意：** 以下编译方法只适用于release/v2.6.0及之后版本(包括 v2.6.0)。release/v2.3及之前版本(包括 v2.3)请参考[release/v2.3源码编译方法](v2.3_compile.md)。

**注意：** 本编译方法暂时只适用于ARM的设备。


如果您还没有配置好ArmLinux编译环境，请先根据[编译环境准备](compile_env)中的内容，根据您的开发环境安装编译ArmLinux预测库所需的编译环境。

```shell
# 1. 下载Paddle-Lite源码 并切换到release分支
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite && git checkout release/v2.6

# (可选) 删除此目录，编译脚本会自动从国内CDN下载第三方库文件
# rm -rf third-party

# 2. 编译Paddle-Lite Linux(arm)预测库 (armv8, gcc编译)
./lite/tools/build_linux.sh
```

**提示：** 编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在git clone完Paddle-Lite仓库代码后，手动删除本地仓库根目录下的third-party目录。编译脚本会自动下载存储于国内 CDN 的第三方依赖的压缩包，节省从git repo同步第三方库代码的时间。

### 编译结果

位于 `Paddle-Lite/build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8` :

```shell

inference_lite_lib.armlinux.armv8
├── cxx                                           C++ 预测库和头文件
│   ├── include                                   C++ 头文件
│   │   ├── paddle_api.h
│   │   ├── paddle_image_preprocess.h
│   │   ├── paddle_lite_factory_helper.h
│   │   ├── paddle_place.h
│   │   ├── paddle_use_kernels.h
│   │   ├── paddle_use_ops.h
│   │   └── paddle_use_passes.h
│   └── lib                                       C++ 预测库
│       ├── libpaddle_api_light_bundled.a         C++ 静态库
│       └── libpaddle_light_api_shared.so         C++ 动态库
├── demo
│   └── python                                    Python 预测库demo
│
└── python                                        Python 预测库(需要打开with_python选项)
    ├── install
    │   └── dist
    │       └── paddlelite-*.whl                  Python whl包
    └── lib
        └── lite.so                               Python 预测库
```


### 编译命令

- 默认编译方法: (armv8, gcc)                                           
```shell
# 默认配置是4线程编译，如果您的设备配置较低（树莓派3B等），可能遇到未知编译错误,
# 建议通过 ```export LITE_BUILD_THREADS=1``` 设置为单线程编译
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
--with_extra: (OFF|ON)          是否编译OCR/NLP模型相关kernel&OP，默认为OFF，只编译CV模型相关kernel&OP
--with_python: (OFF|ON)         是否编译python预测库, 默认为 OFF
--python_version: (2.7|3.5|3.7) 编译whl的Python版本，默认为 None
--with_cv: (OFF|ON)             是否编译CV相关预处理库, 默认为 OFF
--with_log: (OFF|ON)            是否输出日志信息, 默认为 ON
--with_exception: (OFF|ON)      是否在错误发生时抛出异常，默认为 OFF   
```
**注意：with_python现在仅支持armlinux的本地编译，尚不支持docker环境和ubuntu环境**

- 裁剪预测库方法（只编译模型中的kernel&OP，降低预测库体积），详情请参考:  [裁剪预测库](library_tailoring)

```shell
./lite/tools/build_linux.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir

# 编译选项说明
--with_strip: (OFF|ON);   是否根据输入模型裁剪预测库，默认为OFF
--opt_model_dir:          输入模型的绝对路径，需要为opt转化之后的模型
```

- 编译 瑞芯微(Rockchip) NPU 预测库方法，详情请参考：[PaddleLite使用RK NPU预测部署](../demo_guides/rockchip_npu)

```shell
--with_rockchip_npu: (OFF|ON)    是否编译编译 huawei_kirin_npu 的预测库，默认为OFF
--rockchip_npu_sdk_root          Rockchip NP DDK文件的绝对路径
```


- 编译 百度(Baidu) XPU 预测库方法, 详情请参考：[PaddleLite使用百度XPU预测部署](../demo_guides/baidu_xpu)

```shell
--with_baidu_xpu: (OFF|ON)    是否编译编译 baidu_xpu 的预测库，默认为OFF
--baidu_xpu_sdk_root          Baidu XPU DDK文件的绝对路径
```
