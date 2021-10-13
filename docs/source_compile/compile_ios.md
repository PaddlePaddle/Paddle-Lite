
# 使用 MacOS 构建 / 目标终端为 iOS

## 一、简介

本文介绍在 MacOS 操作系统环境下，如何将 Paddle Lite 进行源码编译生成 iOS 预测库。

说明：自 release/v2.10 版本起，Paddle Lite 支持了 Metal 后端。如果您需要 Paddle Lite 正式版本，请直接 [前往下载](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html) 我们预先构建发布的预测库包。

## 二、环境配置

- Xcode IDE
- CMake（请使用 3.15 或以上版本）
- git、make、wget、python

如果您还没有配置好iOS编译环境，请先根据 [编译环境准备](compile_env.html#mac-os) 中的内容，根据您的开发环境安装编译 iOS 预测库所需的编译环境。

## 三、构建

### 3.1 构建步骤

```shell
# 1. 下载 Paddle Lite 源码并切换到特定 release 分支，如 release/v2.10
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite && git checkout release/v2.10

# (可选) 删除此目录，编译脚本会自动从国内 CDN 下载第三方库文件
# rm -rf third-party

# 2. 编译 Paddle Lite ios 预测库
./lite/tools/build_ios.sh
```

**提示：** *编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在 git clone 完 Paddle Lite 仓库代码后，手动删除本地仓库根目录下的 third-party 目录。编译脚本会自动下载存储于国内 CDN 的第三方依赖的压缩包，节省从 git repo 同步第三方库代码的时间。*

### 3.2 构建参数

build_ios.sh 的构建参数：

| 参数 | 说明 | 可选范围 | 默认值 |
| :-- | :-- | :-- | :-- |
| arch           |  目标 ARM 架构   |  armv8 / armv7   |  armv8   |
| with_log       |  是否打印日志  |  OFF / ON |  ON   |
| with_exception |  是否开启异常  |  OFF / ON  |  OFF   |
| with_extra     |  是否编译完整算子（支持序列相关模型，如 OCR 和 NLP）  |  OFF / ON  | OFF   |
| with_metal     |  是否编译支持 Metal 的预测库  | OFF / ON  |  OFF  |
| with_cv        |  是否将 cv 函数编译到库中  |  OFF / ON  |  OFF   |
| ios_deployment_target  | 运行系统的最低版本 | 9.0 及以上 | 9.0 |

备注：
1. 执行`/lite/tools/build_ios.sh help`可输出各选项的使用说明信息；
2. 根据模型包含算子进行预测库裁剪的具体操作，请参考 [裁剪预测库](https://paddle-lite.readthedocs.io/zh/latest/source_compile/library_tailoring.html)。

## 四、验证

### 4.1 编译 Paddle Lite iOS CPU 预测库 (armv8)
执行`./lite/tools/build_ios.sh`，成功后会在`Paddle-Lite/build.ios.ios64.armv8/inference_lite_lib.android.armv8`下生成如下文件：

```shell
inference_lite_lib.ios64.armv8
├── include                                                C++头文件
│   ├── paddle_api.h
│   ├── paddle_image_preprocess.h
│   ├── paddle_lite_factory_helper.h
│   ├── paddle_place.h
│   ├── paddle_use_kernels.h
│   ├── paddle_use_ops.h
│   └── paddle_use_passes.h
└── lib                                                    C++预测库（静态库）
    └── libpaddle_api_light_bundled.a
```
### 4.2 编译 Paddle Lite iOS GPU 预测库 (armv8)

执行`./lite/tools/build_ios.sh --with_metal=ON`，成功后会在`Paddle-Lite/build.ios.metal.ios64.armv8/inference_lite_lib.ios64.armv8.metal`下生成如下文件：

```shell
inference_lite_lib.ios64.armv8
├── include                                                C++头文件
│   ├── paddle_api.h
│   ├── paddle_image_preprocess.h
│   ├── paddle_lite_factory_helper.h
│   ├── paddle_place.h
│   ├── paddle_use_kernels.h
│   ├── paddle_use_ops.h
│   └── paddle_use_passes.h
├── metal                                                  metallib文件
│   └── lite.metallib
└── lib                                                    C++预测库（静态库）
    └── libpaddle_api_light_bundled.a
```
