
# 源码编译 (Android) 

Paddle Lite提供了Android平台的官方Release预测库下载，我们优先推荐您直接下载[Paddle Lite预编译库](../quick_start/release_lib.html#android-toolchain-gcc)。

**注意：** 以下编译方法只适用于release/v2.6.0及之后版本(包括 v2.6.0)。release/v2.3及之前版本(包括 v2.3)请参考[release/v2.3源码编译方法](v2.3_compile.md)。

如果您还没有配置好Andriod交叉编译环境，请先根据[编译环境准备](compile_env)中的内容，根据您的开发环境安装编译Android预测库所需的编译环境。运行编译脚本之前，请先检查环变量`NDK_ROOT`指向正确的Andriod NDK安装路径，之后可以下载并编译 Paddle-Lite源码。

```shell
# 1. 下载Paddle-Lite源码 并切换到release分支
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite && git checkout release/v2.6

# (可选) 删除此目录，编译脚本会自动从国内CDN下载第三方库文件
# rm -rf third-party

# 2. 编译Paddle-Lite Android预测库 (armv8, gcc编译, 静态链接ndk stl)
./lite/tools/build_android.sh
```

**提示：** 编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在git clone完Paddle-Lite仓库代码后，手动删除本地仓库根目录下的third-party目录。编译脚本会自动下载存储于国内 CDN 的第三方依赖的压缩包，节省从git repo同步第三方库代码的时间。

### 编译结果

位于`Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8`:

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

### 编译命令

- 默认编译方法: (armv8, gcc, c++_static)                                           
```                                        shell
./lite/tools/build_android.sh
```

- 打印 help 信息：

```shell
./lite/tools/build_android.sh help
```

- 其他可选编译命令：

```shell
--arch: (armv8|armv7)        arm版本，默认为armv8
--toolchain: (gcc|clang)     编译器类型，默认为gcc
--android_stl: (c++_static|c++_shared)   NDK stl库链接方法，默认为静态链接c++_static
--with_java: (OFF|ON)        是否编译Java预测库, 默认为 ON
--with_cv: (OFF|ON)          是否编译CV相关预处理库, 默认为 OFF
--with_log: (OFF|ON)         是否输出日志信息, 默认为 ON
--with_exception: (OFF|ON)   是否在错误发生时抛出异常，默认为 OFF   
--with_extra: (OFF|ON)       是否编译OCR/NLP模型相关kernel&OP，默认为OFF，只编译CV模型相关kernel&OP
```

- 裁剪预测库方法（只编译模型中的kernel&OP，降低预测库体积），详情请参考:  [裁剪预测库](library_tailoring)

```shell
./lite/tools/build_android.sh --with_strip=ON --opt_model_dir=%YourOptimizedModelDir%

# 编译选项说明
--with_strip: (OFF|ON)   是否根据输入模型裁剪预测库，默认为OFF
--opt_model_dir          输入模型的绝对路径，需要为opt转化之后的模型
```



- 编译 Android npu 预测库方法，详情请参考：[PaddleLite使用华为NPU（Kirin SoC）预测部署](../demo_guides/huawei_kirin_npu)

```shell
./lite/tools/build_android.sh --with_huawei_kirin_npu=ON \
                              --huawei_kirin_npu_sdk_root=%YourNpuSdkPath%

# 编译选项说明
--with_huawei_kirin_npu: (OFF|ON)   是否编译编译huawei_kirin_npu 的预测库，默认为OFF
--huawei_kirin_npu_sdk_root         Huawei HiAi DDK文件的绝对路径，可从以下网址下载 
                                    https://developer.huawei.com/consumer/cn/hiai/
```


- 编译Android opencl 预测库方法，详情请参考：[PaddleLite使用OpenCL预测部署](../demo_guides/opencl)

```shell
./lite/tools/build_android.sh --with_opencl=ON

# 编译选项说明
--with_opencl: (OFF|ON);  是否编译opencl预测库, 默认为 OFF
```
