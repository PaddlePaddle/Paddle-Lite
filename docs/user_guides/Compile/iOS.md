
# 编译iOS预测库

**注意：本编译方法只适用于release/v2.6.0之后版本（包括 v2.6.0)**

安装了iOS的编译环境，可以下载并编译 Paddle-Lite源码

```shell
# 1. 下载Paddle-Lite源码 并切换到release分支
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite && git checkout release/v2.6.0

# 2. 编译Paddle-Lite Android预测库 (armv8, gcc编译, 静态链接ndk stl)
./lite/tools/build_ios.sh
```



### 编译结果

位于`Paddle-Lite/build.ios.ios64.armv8/inference_lite_lib.ios64.armv8`:

```shell
inference_lite_lib.ios64.armv8                iOS预测库和头文件
|-- include                                        C++头文件
|   |-- paddle_api.h                         
|   |-- paddle_image_preprocess.h
|   |-- paddle_lite_factory_helper.h
|   |-- paddle_place.h
|   |-- paddle_use_kernels.h
|   |-- paddle_use_ops.h
|   `-- paddle_use_passes.h
`-- lib                                            C++预测库（静态库）
    `-- libpaddle_api_light_bundled.a
```



### 编译命令

- 默认编译方法: (armv8)                                           
```                                        shell
./lite/tools/build_ios.sh
```

- 打印 help 信息：

```shell
./lite/tools/build_ios.sh help
```

- 其他可选编译命令：

```shell
--arch: (armv8|armv7)        arm版本，默认为armv8
--with_cv: (OFF|ON)          是否编译CV相关预处理库, 默认为 OFF
--with_log: (OFF|ON)         是否输出日志信息, 默认为 ON
--with_extra: (OFF|ON)       是否编译OCR或NLP相关模型的kernel&OP，默认为OFF，只编译CV模型相关kernel&OP
```

- 裁剪预测库方法（只编译模型中的kernel&OP，降低预测库体积）:

```shell
./lite/tools/build_android.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir
```
```shell
--with_strip: (OFF|ON);   是否根据输入模型裁剪预测库，默认为OFF
--opt_model_dir:          输入模型的绝对路径，需要为opt转化之后的模型
```
详情参考:  [裁剪预测库](https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html)
