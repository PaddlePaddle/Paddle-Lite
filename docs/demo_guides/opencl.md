# OpenCL 部署示例

Paddle Lite 利用跨平台计算框架 OpenCL 将计算映射到 GPU 上执行，以充分利用 GPU 硬件算力，提高推理性能。在执行时会优先在 GPU 上执行算子，如果算子没有 GPU 实现，则该算子会回退到 CPU 上执行。

## 1. 支持现状
- Android/ARMLinux 系统下:
  - 高通骁龙 Adreno 系列 GPU，包括但不限于 Adreno 888+/888/875/865/855/845/835/625 等具体型号
  - ARM Mali 系列 GPU (具体为支持 Midgard、Bifrost、Valhall 这三个 GPU 架构下的 GPU)，如 Mali G76 MP16 (Valhall 架构，华为 P40 Pro), Mali-G72 MP3 (Bifrost 架构，OPPO R15), Mali T860（Midgard 架构，RK3399）
  - PowerVR 系列 GPU，如 PowerVR Rogue GE8320，对应芯片联发科 MT8768N
- macOS 系统下：
  -  Intel 集成显卡
  -  Apple Silicon 芯片，如 M1, M1 Pro
- Windows 64 位系统下：
  - Intel 集成显卡
  - NVIDIA/AMD 独立显卡

## 2. 在 Android 系统上运行
### 2.1 编译预测库
Paddle Lite 同时支持在 Linux x86 环境和 macOS 环境下编译适用于 Android 的库。
- 如果宿主机是 Linux x86 环境，请根据 [Linux x86 环境下编译适用于 Android 的库](../source_compile/linux_x86_compile_android) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。
- 如果宿主机是 macOS 环境，请根据 [macOS 环境下编译适用于 Android 的库](../source_compile/macos_compile_android) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。

重点编译命令为：

```shell
# 有 2 种编译方式，tiny_publish 方式编译，适用于实际部署；full_publish 方式编译，会生成更多编译产物。
# 编译时二选一即可。
# 方式 1：tiny_publish 方式编译，适用于部署
./lite/tools/build_android.sh --with_opencl=ON
# 方式 2：full_publish 方式编译，会生成更多编译产物
./lite/tools/build_android.sh --with_opencl=ON full_publish
# 注：
#    编译帮助请执行: ./lite/tools/build_android.sh help
#    为了方便调试，建议在编译时加入选项 --with_log=ON
```


编译成功后，会在`Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl`目录下生成编译产物，主要目录结构如下：

```shell
|-- cxx                                          C++ 预测库和头文件
|   |-- include                                  C++ 头文件
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                      C++ 预测库
|       |-- libpaddle_light_api_shared.so        C++ 动态库（轻量库，tiny_publish/full_publish 编译产物）
|       `-- libpaddle_full_api_shared.so         C++ 动态库（全量库，full_publish 编译产物）
|-- java                                         Java 预测库
|   |-- jar                                      Java JAR 包
|   |   `-- PaddlePredictor.jar
|   |-- so                                       Java JNI 动态链接库
|   |   `-- libpaddle_lite_jni.so
|   `-- src
|       `-- com
|-- demo                                         示例代码
|   |-- cxx                                      C++ 预测库示例
|   |   |-- Makefile.def
|   |   |-- README.md
|   |   |-- mobile_light                         mobilenetv1 推理（tiny_publish/full_publish 编译产物）
|   |   |-- mobile_full                          mobilenetv1 推理（full_publish 编译产物）
|   `-- java                                     Java 预测库示例
|       |-- README.md
|       `-- android
`-- opencl                                       OpenCL 核函数
    `-- cl_kernel
        |-- buffer
        |-- cl_common.h
        `-- image
```

关于增量编译：
- 在项目根目录执行 `python ./lite/tools/cmake_tools/gen_opencl_code.py ./lite/backends/opencl/cl_kernel ./lite/backends/opencl/opencl_kernels_source.cc`，该命令会自动更新 `opencl_kernels_source.cc`；
- 然后进入 build 目录（如 `build.lite.android.armv8.gcc` ）下执行 `make publish_inference` 或者待编译的单测的可执行文件名（如 `make test_fc_image_opencl`），cl kernel 文件的内容会随着编译自动打包到产物包如 `.so` 中或者对应单测可执行文件中。

### 2.2 运行示例
mobile_light 示例为使用 `MobileConfig` 加载并解析 `opt` 优化过的 `.nb` 模型，执行推理预测。mobile_full 示例为使用 `CxxConfig` 直接加载并解析 Paddle 模型，在运行时进行图优化操作，执行推理预测。

#### mobile_light 示例
`opt` 工具相关文档：
- [opt 工具的获取](../user_guides/model_optimize_tool)
- [opt 的使用说明](../user_guides/opt/opt_bin)

具体执行步骤如下：
```shell
# 1. 准备 .nb 模型
# 使用 opt 工具手动转换
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz && tar zxvf mobilenet_v1.tar.gz
./build.opt/lite/api/opt --model_dir=./mobilenet_v1 \
                         --valid_targets=opencl,arm \
                         --optimize_out=mobilenetv1_opt_opencl

# 2. 编译
cd build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_light
make
cd -

# 3. 推送可执行文件、预测库、模型文件到手机（请提前确保手机已连接到宿主机并可通过 adb devices 命令查询到设备）
adb shell mkdir /data/local/tmp/opencl
adb push build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_light/mobilenetv1_light_api /data/local/tmp/opencl/
adb push build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/opencl/
adb push mobilenetv1_opt_opencl.nb data/local/tmp/opencl/

# 4. 在宿主机上运行
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/opencl/; \
           export GLOG_v=4; \
           /data/local/tmp/opencl/mobilenetv1_light_api \
           /data/local/tmp/opencl/mobilenetv1_opt_opencl.nb \
           1,3,224,224 \
           100 10 0 1 1 0"
           # repeats=100
           # warmup=10
           # power_mode=0 绑定大核
           # thread_num=1
           # accelerate_opencl=1 开启 opencl kernel cache & tuning，仅当模型运行在 opencl 后端时该选项才会生效
           # print_output=0 不打印模型输出 tensors 详细数据
```

#### mobile_full 示例
```shell
# 1. 准备 Paddle 模型
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz && tar zxvf mobilenet_v1.tar.gz

# 2. 编译
cd build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_full
make
cd -

# 3. 推送可执行文件、预测库、模型文件到手机（请提前确保手机已连接到宿主机并可通过 adb devices 命令查询到设备）
adb shell mkdir /data/local/tmp/opencl
adb push build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/demo/cxx/mobile_full/mobilenetv1_full_api /data/local/tmp/opencl/
adb push build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/cxx/lib/libpaddle_full_api_shared.so /data/local/tmp/opencl/
adb push mobilenet_v1 /data/local/tmp/opencl/

# 4. 在宿主机上运行
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/opencl/; \
           export GLOG_v=4; \
           /data/local/tmp/opencl/mobilenetv1_full_api \
               --model_dir=/data/local/tmp/opencl/mobilenet_v1 \
               --optimized_model_dir=/data/local/tmp/opencl/mobilenetv1_opt_opencl \
               --warmup=10 \
               --repeats=100 \
               --use_gpu=true"
```

## 3. 在 ARMLinux 系统上运行
### 3.1 编译预测库
Paddle Lite 同时支持在 Linux x86 环境下和 ARMLinux 环境下编译适用于 ARMLinux 的库。
- 如果宿主机是 Linux x86 环境，请根据 [Linux x86 环境下编译适用于 ARMLinux 的库](../source_compile/linux_x86_compile_arm_linux) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。
- 如果宿主机是 ARMLinux 环境，请根据 [ARMLinux 环境下编译适用于 ARMLinux 的库](../source_compile/arm_linux_compile_arm_linux) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。

重点编译命令为：

```shell
# 有 2 种编译方式，tiny_publish 方式编译，适用于实际部署；full_publish 方式编译，会生成更多编译产物。
# 编译时二选一即可。
#
# 宿主机是 Linux x86 环境或 ARMLinux 环境时
# 方式 1：tiny_publish 方式编译，适用于部署
./lite/tools/build_linux.sh --with_opencl=ON
# 方式 2：full_publish 方式编译，会生成更多编译产物
./lite/tools/build_linux.sh --with_opencl=ON full_publish
#
# 注：
#    编译帮助请执行: ./lite/tools/build_linux.sh help
#    build_linux.sh 脚本中默认已开启 LOG
```

编译成功后，会在`Paddle-Lite/build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl`目录下生成编译产物，主要目录结构如下：

```shell
|-- cxx                                          C++ 预测库和头文件
|   |-- include                                  C++ 头文件
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                      C++ 预测库
|       |-- libpaddle_full_api_shared.so         C++ 动态库（全量库，full_publish 编译产物）
|       `-- libpaddle_light_api_shared.so        C++ 动态库（轻量库，tiny_publish/full_publish 编译产物）
`-- demo                                         示例代码
    `-- cxx                                      C++ 预测库示例
        |-- mobilenetv1_light                    mobilenetv1 推理（tiny_publish/full_publish 编译产物）
        `-- mobilenetv1_full                     mobilenetv1 推理（full_publish 编译产物）
```

### 3.2 运行示例
mobilenetv1_light 示例为使用 `MobileConfig` 加载并解析 `opt` 优化过的 `.nb` 模型，执行推理预测。mobilenetv1_full 示例为使用 `CxxConfig` 直接加载并解析 Paddle 模型，在运行时进行图优化操作，执行推理预测。

#### mobilenetv1_light 示例
`opt` 工具相关文档：
- [opt 工具的获取](../user_guides/model_optimize_tool)
- [opt 的使用说明](../user_guides/opt/opt_bin)

以宿主机为 Linux x86 环境为例，具体执行步骤如下：
```shell
# 1. 准备 .nb 模型
# 使用 opt 工具手动转换
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz && tar zxvf mobilenet_v1.tar.gz
./build.opt/lite/api/opt --model_dir=./mobilenet_v1 \
                         --valid_targets=opencl,arm \
                         --optimize_out=mobilenetv1_opt_opencl

# 2. 编译
cd build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/demo/cxx/mobilenetv1_light
bash build.sh
cd -

# 3. 拷贝可执行文件、预测库、模型文件到设备：可通过 scp 或其他方式将三个文件拷贝到开发板
ssh name@ip
mkdir ~/opencl
exit
scp build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/demo/cxx/mobilenetv1_light/mobilenetv1_light_api name@ip:~/opencl
scp build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/cxx/lib/libpaddle_light_api_shared.so name@ip:~/opencl
scp -r mobilenetv1_opt_opencl.nb name@ip:~/opencl

# 4. 在设备上运行
ssh name@ip
cd ~/opencl
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH; \
export GLOG_v=4; \
./mobilenetv1_light_api \
    mobilenetv1_opt_opencl.nb \
    1,3,224,224 \
    100 10 0 1 1 0
    # repeats=100
    # warmup=10
    # power_mode=0 绑定大核
    # thread_num=1
    # accelerate_opencl=1 开启 opencl kernel cache & tuning，仅当模型运行在 opencl 后端时该选项才会生效
    # print_output=0 不打印模型输出 tensors 详细数据
```

#### mobilenetv1_full 示例
```shell
# 1. 准备 Paddle 模型
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz && tar zxvf mobilenet_v1.tar.gz

# 2. 编译
cd build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/demo/cxx/mobilenetv1_full
bash build.sh
cd -

# 3. 拷贝可执行文件、预测库、模型文件到设备：可通过 scp 或其他方式将三个文件拷贝到开发板
ssh name@ip
mkdir ~/opencl
exit
scp build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/demo/cxx/mobilenetv1_full/mobilenetv1_full_api name@ip:~/opencl
scp build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/cxx/lib/libpaddle_full_api_shared.so name@ip:~/opencl
scp -r mobilenet_v1 name@ip:~/opencl

# 4. 在设备上运行
ssh name@ip
cd ~/opencl
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH; \
export GLOG_v=4; \
./mobilenetv1_full_api \
    --model_dir=./mobilenet_v1 \
    --optimized_model_dir=mobilenetv1_opt_opencl \
    --warmup=10 \
    --repeats=100 \
    --use_gpu=true
```

## 4. 在 macOS 系统上运行
### 4.1 编译预测库
Paddle Lite 支持在 macOS 环境下编译适用于 macOS 的库。宿主机必须是 macOS 环境，请根据 [macOS 环境下编译适用于 macOS 的库](../source_compile/macos_compile_macos) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。

重点编译命令为：

```shell
# 宿主机是 macOS x86 环境时
./lite/tools/build_macos.sh --with_opencl=ON x86
# 宿主机是 macOS arm64 环境时
./lite/tools/build_macos.sh --with_opencl=ON arm64
#
# 注：
#    编译帮助请执行: ./lite/tools/build_macos.sh help
#    build_linux.sh 脚本中默认已开启 LOG
```

以宿主机为 macOS x86 环境为例，编译成功后，会在`Paddle-Lite/build.lite.x86.opencl/inference_lite_lib`目录下生成编译产物，主要目录结构如下：

```shell
|-- cxx                                          C++ 预测库和头文件
|   |-- include                                  C++ 头文件
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                      C++ 预测库
|       |-- libpaddle_api_light_bundled.a        C++ 静态库(轻量库)
|       |-- libpaddle_light_api_shared.dylib     C++ 动态库(轻量库)
|       |-- libpaddle_api_full_bundled.a         C++ 静态库(全量库)
|       `-- libpaddle_full_api_shared.dylib      C++ 动态库(全量库)
`-- demo                                         示例代码
    `-- cxx                                      C++ 预测库示例
        |-- mobilenetv1_light                    mobilenetv1 推理（tiny_publish/full_publish 编译产物）
        `-- mobilenetv1_full                     mobilenetv1 推理（full_publish 编译产物）
```

### 4.2 运行示例
mobilenetv1_light 示例为使用 `MobileConfig` 加载并解析 `opt` 优化过的 `.nb` 模型，执行推理预测。mobilenetv1_full 示例为使用 `CxxConfig` 直接加载并解析 Paddle 模型，在运行时进行图优化操作，执行推理预测。

#### mobilenetv1_light 示例
`opt` 工具相关文档：
- [opt 工具的获取](../user_guides/model_optimize_tool)
- [opt 的使用说明](../user_guides/opt/opt_bin)

以宿主机为 macOS x86 环境为例，具体执行步骤如下：
```shell
# 1. 准备 .nb 模型
# 使用 opt 工具手动转换
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz && tar zxvf mobilenet_v1.tar.gz
./build.opt/lite/api/opt --model_dir=./mobilenet_v1 \
                         --valid_targets=opencl,x86 \
                         --optimize_out=mobilenetv1_opt_x86_opencl
# 备注：当宿主机为 macOS arm64 环境时，如上命令中的 --valid_targets 应设置为 opencl,arm，其他命令保持不变。

# 2. 编译
cd build.lite.x86.opencl/inference_lite_lib/demo/cxx/mobilenetv1_light
bash build.sh
cd -

# 3. 运行
export GLOG_v=4
./build.lite.x86.opencl/inference_lite_lib/demo/cxx/mobilenetv1_light/mobilenet_light_api \
    ./mobilenetv1_opt_x86_opencl.nb \
    1,3,224,224 \
    100 10 0
    # repeats=100
    # warmup=10
    # print_output=0 不打印模型输出 tensors 详细数据
```

#### mobilenetv1_full 示例
```shell
# 1. 准备 Paddle 模型
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz && tar zxvf mobilenet_v1.tar.gz

# 2. 编译
cd build.lite.x86.opencl/inference_lite_lib/demo/cxx/mobilenetv1_full
bash build.sh
cd -

# 3. 运行
export GLOG_v=4
./build.lite.x86.opencl/inference_lite_lib/demo/cxx/mobilenetv1_full/mobilenet_full_api \
    ./mobilenet_v1 \
    1,3,224,224 \
    100 10 0
    # repeats=100
    # warmup=10
    # print_output=0 不打印模型输出 tensors 详细数据
```

## 5. 在 Windows 64 位系统上运行
### 5.1 编译预测库
Paddle Lite 支持在 Windows 环境下编译适用于 Windows 的库。请根据 [Windows 环境下编译适用于 Windows 的库](../source_compile/windows_compile_windows) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。

重点编译命令为：

```shell
lite\tools\build_windows.bat with_opencl
# 注：
#    编译帮助请执行: lite\tools\build_windows.bat help
#    build_windows.bat 中默认已开启 LOG
```

编译成功后，会在`Paddle-Lite\build.lite.x86.opencl\inference_lite_lib`目录下生成编译产物。

### 5.2 运行示例
mobilenetv1_light 示例为使用 `MobileConfig` 加载并解析 `opt` 优化过的 `.nb` 模型，执行推理预测。mobilenetv1_full 示例为使用 `CxxConfig` 直接加载并解析 Paddle 模型，在运行时进行图优化操作，执行推理预测。

#### mobilenetv1_light 示例
`opt` 工具相关文档：
- [opt 工具的获取](../user_guides/model_optimize_tool)
- [opt 的使用说明](../user_guides/opt/opt_bin)

具体执行步骤如下：
```shell
# 1. 准备 .nb 模型
# 使用 opt 工具手动转换
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz && tar zxvf mobilenet_v1.tar.gz
build.opt\lite\api\opt --model_dir=./mobilenet_v1 \
                       --valid_targets=opencl,x86 \
                       --optimize_out=mobilenetv1_opt_x86_opencl

# 2. 编译
cd build.lite.x86.opencl\inference_lite_lib\demo\cxx\mobilenetv1_light
build.bat
cd -

# 3. 运行
export GLOG_v=4
.\build.lite.x86.opencl\inference_lite_lib\demo\cxx\mobilenetv1_light\mobilenet_light_api \
    ./mobilenetv1_opt_x86_opencl.nb \
    1,3,224,224 \
    100 10 0
    # repeats=100
    # warmup=10
    # print_output=0 不打印模型输出 tensors 详细数据
```

#### mobilenetv1_full 示例
```shell
# 1. 准备 Paddle 模型
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz && tar zxvf mobilenet_v1.tar.gz

# 2. 编译
cd build.lite.x86.opencl\inference_lite_lib\demo\cxx\mobilenetv1_full
build.bat
cd -

# 3. 运行
export GLOG_v=4
.\build.lite.x86.opencl\inference_lite_lib\demo\cxx\mobilenetv1_full\mobilenet_full_api \
    ./mobilenet_v1 \
    1,3,224,224 \
    100 10 0
    # repeats=100
    # warmup=10
    # print_output=0 不打印模型输出 tensors 详细数据
```

## 6. 如何在 Code 中使用

即编译产物 `demo/cxx/mobile_light` 目录下的代码，在线版参考 GitHub 仓库[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)，其中也包括判断当前设备是否支持 OpenCL 的方法;

注：这里给出的链接会跳转到线上最新 develop 分支的代码，很可能与您本地的代码存在差异，建议参考自己本地位于 `lite/demo/cxx/` 目录的代码，查看如何使用。

## 7. 性能分析和精度分析
关于性能和精度分析，请详细查阅[性能测试](../performance/benchmark_tools)中的【逐层耗时和精度分析】章节。

在编译预测库时，使能性能分析和精度分析功能的命令如下：
Android 平台下：
```shell
# 开启性能分析，会打印出每个 op 耗时信息和汇总信息
./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_opencl=ON --with_extra=ON --with_profile=ON full_publish
# 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_opencl=ON --with_extra=ON --with_precision_profile=ON full_publish
```

macOS x86 平台下：
```shell
# 开启性能分析，会打印出每个 op 耗时信息和汇总信息
./lite/tools/build.sh --build_opencl=ON --build_extra=ON --with_profile=ON x86
# 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
./lite/tools/build.sh --build_opencl=ON --build_extra=ON --with_precision_profile=ON x86
```

Windows x86 平台下：
```shell
# 开启性能分析，会打印出每个 op 耗时信息和汇总信息
.\lite\tools\build_windows.bat with_opencl with_extra with_profile
# 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
.\lite\tools\build_windows.bat with_opencl with_extra with_precision_profile
```

## 8. 关键 API 接口
### 判断设备是否支持 OpenCL
函数 `IsOpenCLBackendValid` 用来检查设备是否支持 OpenCL，该函数内部会依次进行 OpenCL 驱动库检查、库函数检查、精度检查，检查均通过后返回 `true`，否则返回 `false`.
- 函数声明[ paddle_api.h ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/api/paddle_api.h)
- 使用示例[ mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)

### 设置 OpenCL kernel 缓存文件的路径
函数 `set_opencl_binary_path_name` 用来开启 OpenCL kernel 缓存功能，并设置缓存文件名和存放路径。使用该函数可以避免在线编译 OpenCL kernel，进而提高首帧运行速度。推荐在工程代码中使用该函数。

```c++
  /// \brief Set path and file name of generated OpenCL compiled kernel binary.
  ///
  /// If you use GPU of specific soc, using OpenCL binary will speed up the
  /// initialization.
  ///
  /// \param path  Path that OpenCL compiled kernel binay file stores in. Make
  /// sure the path exist and you have Read&Write permission.
  /// \param name  File name of OpenCL compiled kernel binay.
  /// \return void
  void set_opencl_binary_path_name(const std::string& path,
                                   const std::string& name);
```

- 函数声明[ paddle_api.h ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/api/paddle_api.h)
- 使用示例[ mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)

### 设置 OpenCL Auto-tune 策略
函数 `set_opencl_tune` 用来自动选择当前硬件和模型下的最优 OpenCL 卷积算子实现方案，并将找到的算法配置序列化到文件中。该函数通过预先试跑，找到最优的算法。推荐在 benchmark 时使用该函数。

```c++

  /// \brief Set path and file name of generated OpenCL algorithm selecting file.
  ///
  /// If you use GPU of specific soc, using OpenCL binary will speed up the
  /// running time in most cases. But the first running for algorithm selecting
  /// is timg-costing.
  ///
  /// \param tune_mode  Set a tune mode:
  ///        CL_TUNE_NONE: turn off
  ///        CL_TUNE_RAPID: find the optimal algorithm in a rapid way(less time-cost)
  ///        CL_TUNE_NORMAL: find the optimal algorithm in a noraml way(suggestion)
  ///        CL_TUNE_EXHAUSTIVE: find the optimal algorithm in a exhaustive way(most time-costing)
  /// \param path  Path that OpenCL algorithm selecting file stores in. Make
  /// sure the path exist and you have Read&Write permission.
  /// \param name  File name of OpenCL algorithm selecting file.
  /// \param lws_repeats  Repeat number for find the optimal local work size .
  /// \return void
  void set_opencl_tune(CLTuneMode tune_mode = CL_TUNE_NONE,
                       const std::string& path = "",
                       const std::string& name = "",
                       size_t lws_repeats = 4);
```

- 函数声明[ paddle_api.h ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/api/paddle_api.h)
- 使用示例[ mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)

### 设置运行时精度
函数 `set_opencl_precision` 用来设置 OpenCL 运行时精度为 fp32 或 fp16。

OpenCL 的 fp16 特性是 OpenCL 标准的一个扩展，当前绝大部分移动端设备都支持该特性。Paddle-Lite 的 OpenCL 实现同时支持如上两种运行时精度。
- 在 Android/ARMLinux 系统下默认使用 fp16 计算，可通过调用该函数配置为 fp32 精度计算；
- 在 macOS/Windows 64 位系统下默认使用 fp32 计算，其中 macOS 系统下由于苹果驱动原因只能支持 fp32 精度；Windows 64 位系统下，Intel 集成显卡只能支持 fp32 精度计算，NVIDIA 独立显卡可以支持 fp32/fp16 两种精度计算。如果设备不支持 fp16，在编译预测库时开启 log 的前提下，Paddle-Lite OpenCL 后端代码会有报错提示。

```c++
  /// \brief Set runtime precision on GPU using OpenCL backend.
  ///
  /// \param p
  ///          CL_PRECISION_AUTO: first fp16 if valid, default
  ///          CL_PRECISION_FP32: force fp32
  ///          CL_PRECISION_FP16: force fp16
  /// \return void
  void set_opencl_precision(CLPrecisionType p = CL_PRECISION_AUTO);
```

- 函数声明[ paddle_api.h ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/api/paddle_api.h)
- 使用示例[ mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)


## 9. 常见问题

1. OpenCL 计算过程中大多以 `cl::Image2D` 的数据排布进行计算，不同 gpu 支持的最大 `cl::Image2D` 的宽度和高度有限制，模型输入的数据格式是 buffer 形式的 `NCHW` 数据排布方式。要计算你的模型是否超出最大支持（大部分手机支持的 `cl::Image2D` 最大宽度和高度均为 16384），可以通过公式 `image_h = tensor_n * tensor_h, image_w=tensor_w * (tensor_c + 3) / 4` 计算当前层 `NCHW` 排布的 Tensor 所需的 `cl::Image2D` 的宽度和高度。如果某一层的 Tensor 维度大于如上限制，则会会在日志中输出超限提示。
2. 当前版本的 Paddle Lite OpenCL 后端不支持量化模型作为输入；支持 fp32 精度的模型作为输入，在运行时会根据运行时精度配置 API `config.set_opencl_precision()` 来设定运行时精度（fp32 或 fp16）。
3. 部署时需考虑不支持 OpenCL 的情况，可预先使用 API `bool ::IsOpenCLBackendValid()` 判断，对于不支持的情况加载 CPU 模型，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
4. 对性能不满足需求的场景，可以考虑使用调优 API `config.set_opencl_tune(CL_TUNE_NORMAL)`，首次会有一定的初始化耗时，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
5. 对精度要求较高的场景，可以考虑通过 API `config.set_opencl_precision(CL_PRECISION_FP32)` 强制使用 `FP32` 精度，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
6. 对首次加载耗时慢的问题，可以考虑使用 API `config.set_opencl_binary_path_name(bin_path, bin_name)`，提高首次推理时，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
7. Paddle Lite OpenCL 后端代码尚未完全支持动态 shape，因此在运行动态 shape 的模型时可能会报错。
8. 使用 OpenCL 后端进行部署时，模型推理速度并不一定会比在 CPU 上执行快。GPU 适合运行较大计算强度的负载任务，如果模型本身的单位算子计算密度较低，则有可能出现 GPU 推理速度不及 CPU 的情况。在面向 GPU 设计模型结构时，需要尽量减少低计算密度算子的数量，比如 slice、concat 等，具体可参见[使用 GPU 获取最佳性能](../performance/gpu.md)中的【优化建议】章节。
