# 源码编译 (Arm MacOS)

**注意：** 以下编译方法只适用于**develop(commit id:647845) 以及release/v2.9.1** 及之后版本

如果您还没有配置好编译环境，请先根据[编译环境准备](compile_env.html#mac-os)中的内容，根据您的开发环境安装编译预测库所需的编译环境。

# 编译 Paddle-Lite Arm MacOs 预测库 (armv8)

/lite/tools/build_macos.sh arm64

**提示：** 编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在 git clone 完 Paddle-Lite 仓库代码后，手动删除本地仓库根目录下的 third-party 目录。编译脚本会自动下载存储于国内 CDN 的第三方依赖的压缩包，节省从 git repo 同步第三方库代码的时间。

### 编译结果

位于`Paddle-Lite/build.macos.armmacos.armv8/inference_lite_lib.armmacos.armv8`:

```shell
inference_lite_lib.armmacos.armv8
├── include                                                C++头文件
│   ├── paddle_api.h
    ├── paddle_image_preprocess.h
│   ├── paddle_lite_factory_helper.h
│   ├── paddle_place.h
│   ├── paddle_use_kernels.h
│   ├── paddle_use_ops.h
│   └── paddle_use_passes.h
└── lib                                                    C++预测库（静态库）
    └── libpaddle_api_light_bundled.a
```

### 编译命令

 默认编译方法: (armv8)                                           
```shell
./lite/tools/build_macos.sh arm64
```

- 打印 help 信息：

```shell
./lite/tools/build_ios.sh help
```

- 其他可选编译命令(Arm Macos M1 芯片只支持 Armv8,不支持 Armv7）：
```shell
--build_cv: (OFF|ON)          是否编译CV相关预处理库, 默认为 OFF
--with_log: (OFF|ON)         是否输出日志信息, 默认为 ON
--with_exception: (OFF|ON)   是否在错误发生时抛出异常，默认为 OFF
--build_extra: (OFF|ON)       是否编译OCR/NLP模型相关kernel&OP，默认为OFF，只编译CV模型相关kernel&OP
```

 裁剪预测库方法（只编译模型中的kernel&OP，降低预测库体积），详情请参考:  [裁剪预测库](library_tailoring)

```shell
./lite/tools/build_macos.sh --with_strip=ON --opt_model_dir=%YourOptimizedModelDir%

# 编译选项说明
--with_strip: (OFF|ON);   是否根据输入模型裁剪预测库，默认为OFF
--opt_model_dir:          输入模型的绝对路径，需要为opt转化之后的模型
```
