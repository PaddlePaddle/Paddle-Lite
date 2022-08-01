# Metal 部署示例

Paddle Lite 支持在 iOS 和 macOS 系统上运行基于 Metal 的程序

## 1. 编译

### 1.1 编译环境

macOS 操作系统（支持 x86/arm 架构），并且成功安装 Xcode。

详见 [**源码编译指南**](https://paddlelite.paddlepaddle.org.cn/source_compile/macos_compile_ios.html) 章节。

### 1.2 编译 Paddle Lite Metal 库 iOS 范例

注：以 `ios/metal` 为目标、macOS M1 芯片、CMake3.21 作为编译开发环境为例。

#### (1) 下载代码
```bash
# 下载 Paddle Lite 源码
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
# 切换到 release 分支
git checkout <release-version-tag>
```
#### (2) 源码编译
```bash
cd Paddle-Lite

# (可选) 删除 third-party 目录，编译脚本会自动从国内 CDN 下载第三方库文件
# rm -rf third-party

# 请在 Paddle-Lite 当前目录下执行脚本
./lite/tools/build_ios.sh --with_metal=ON
```
其他可选择编译选项
- `with_extra`: `[OFF | ON]`，编译全量 op 和 kernel，包含控制流 NLP 相关的 op 和 kernel 体积会大，编译时间长；
- `with_profile`: `[OFF | ON]`，是否使用 Profiler 编译；
- `with_xcode`: `[OFF | ON]`， 是否使用 Xcode 编译；
- `with_cv`: `[OFF | ON]`，编译 ARM CPU Neon 实现的的 cv 预处理模块；
- `with_exception`: `[OFF | ON]`，是否开启 C++ 异常；
- `with_log`: `[ON | OFF]`，是否在执行过程打印日志；


> 说明： 编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在完成 Paddle Lite 源码下载后，删除本地仓库根目录下的 third-party 目录，编译脚本会自动下载存储于国内 CDN 的第三方依赖文件压缩包，节省从 GitHub repo 同步第三方库的时间。

#### (3) 编译产物说明

编译产物位于 `build.ios.metal.ios64.armv8` 下的 `inference_lite_lib.ios64.armv8.metal` 文件夹内，文件目录如下：

```bash
inference_lite_lib.ios64.armv8.metal
├── include                                                C++ 头文件
│   ├── paddle_api.h
│   ├── paddle_image_preprocess.h
│   ├── paddle_lite_factory_helper.h
│   ├── paddle_place.h
│   ├── paddle_use_kernels.h
│   ├── paddle_use_ops.h
│   └── paddle_use_passes.h
├── lib                                                    C++ 预测库（静态库）
│   └── libpaddle_api_light_bundled.a
└── metal                                                  metallib 文件    
    └── lite.metallib
```
### 1.3 编译 Paddle Lite Metal 库 macOS 范例

注：以 `macOS/metal` 为目标、macOS M1 芯片、CMake3.21 作为编译开发环境为例。

#### (1) 源码编译
```bash
cd Paddle-Lite

# (可选) 删除 third-party 目录，编译脚本会自动从国内 CDN 下载第三方库文件
# rm -rf third-party

# 请在 Paddle-Lite 当前目录下执行脚本
./lite/tools/build_macos.sh --with_metal=ON
```
其他可选择编译选项
- `with_extra`: `[OFF | ON]`，编译全量 op 和 kernel，包含控制流 NLP 相关的 op 和 kernel 体积会大，编译时间长；
- `with_profile`: `[OFF | ON]`，是否使用 Profiler 编译；
- `with_xcode`: `[OFF | ON]`， 是否使用 Xcode 编译；
- `with_cv`: `[OFF | ON]`，编译 ARM CPU Neon 实现的的 cv 预处理模块；
- `with_exception`: `[OFF | ON]`，是否开启 C++ 异常；
- `with_log`: `[ON | OFF]`，是否在执行过程打印日志；
- `with_python`: `[OFF | ON]`，是否生成 python whl 安装包；

#### (2) 编译产物说明

编译产物位于 `build.macos.armmacos.armv8.metal` 下的 `inference_lite_lib.armmacos.armv8.metal` 文件夹内，文件目录如下：

```bash
inference_lite_lib.armmacos.armv8.metal
├── cxx                                               C++ 预测库和头文件
│   ├── include                                       C++ 头文件
│   │   ├── paddle_api.h
│   │   ├── paddle_image_preprocess.h
│   │   ├── paddle_lite_factory_helper.h
│   │   ├── paddle_place.h
│   │   ├── paddle_use_kernels.h
│   │   ├── paddle_use_ops.h
│   │   └── paddle_use_passes.h
│   └── lib                                           C++ 预测库
│       ├── libpaddle_api_light_bundled.a             C++ 静态库(轻量库)
│       └── libpaddle_light_api_shared.dylib          C++ 动态库(轻量库)
│       ├── libpaddle_api_full_bundled.a.a            C++ 静态库(全量库)
│       └── libpaddle_full_api_shared.dylib           C++ 动态库(全量库)
│
├── metal                                              metallib 文件    
│    └── lite.metallib
│
│
└── demo                                               C++ 示例代码
    └── cxx                                            C++ 预测库 demo
        └── armmacos_mobile_light_demo                 
            └── mobilenetv1_light_api.cc
```

## 2. 运行示例

下面以 macOS 的环境为例，介绍 opt 转换得到的 Metal 模型如何在 iOS/macOS 设备上执行基于 Metal 的 ARM GPU 推理过程。

### 2.1 iOS demo 部署方法
在 iOS demo 部署过程中，需要将 inference_lite_lib.ios64.armv8.metal 文件中的所有编译产物手动复制到 Xcode 工程当中，其中包括： 
- include 文件中的所有 C++ 头文件；
- lib 文件中的静态库文件 libpaddle_api_light_bundled.a；
- metal 文件中的 Metal 库文件 lite.metallib；

详细 iOS demo 部署方法参考
[ iOS工程示例 ](https://paddlelite.paddlepaddle.org.cn/v2.10/demo_guides/ios_app_demo.html#ios-demo)

### 2.2 macOS demo 部署方法
安装好 Xcode 后，在 demo 目录下添加由 opt 转换得到的 Metal 模型，并运行以下命令：
```
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ -isysroot $(xcrun --sdk macosx --show-sdk-path) -target arm64-macos11 -std=c++11 -I .{$Paddle-Lite}/build.macos.armmacos.armv8/inference_lite_lib.armmacos.armv8/cxx/include .{$Paddle-Lite}/build.macos.armmacos.armv8/inference_lite_lib.armmacos.armv8/cxx/lib/libpaddle_api_light_bundled.a ./mobilenet_light_api.cc -o mobilenet_light_api -framework MetalPerformanceShaders -framework Metal -framework Foundation -framework CoreGraphics -DMETAL=ON
```


注意:  将 mobilenetv1_light_api.cc 文件中以下两个头文件注释去掉。
```
#include "include/paddle_use_ops.h"
#include "include/paddle_use_kernels.h"
 ```
