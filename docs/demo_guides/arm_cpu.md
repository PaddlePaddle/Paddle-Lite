# Arm

Lite 支持在 Android/iOS/ARMLinux 等移动端设备上运行高性能的 CPU 预测库，目前支持 Ubuntu 环境下 armv8、armv7 的交叉编译。

## 1. 编译

### 1.1 编译环境

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

详见 **源码编译指南-环境准备** 章节。

### 1.2 编译 Paddle Lite ARM CPU 库范例

注：以 `android/armv8` 目标、Docker 容器的编译开发环境为例，CMake3.10，android-ndk-r17c 位于 `/opt/` 目录下。

#### 针对 Paddle Lite 用户的编译命令(无单元测试,有编译产物,适用于 benchmark)

- `arm_abi`: `[armv7 | armv8]`；
- `toolchain`: `[gcc | clang]`；
- `build_extra`: `[OFF | ON]`，编译全量 op 和 kernel，包含控制流 NLP 相关的 op 和 kernel 体积会大，编译时间长；
- `build_cv`: `[OFF | ON]`，编译 ARM CPU Neon 实现的的 cv 预处理模块；
- `android_stl`: `[c++_shared | c++_static | gnu_static | gnu_shared]`，Paddle Lite 的库以何种方式链接 `android_stl` ，选择 `c++_shared` 得到的动态库体积更小，但使用时候记得上传 Paddle Lite 所编译版本（armv7 或 armv8 ）一致的 `libc++_shared.so`, 默认使用 `c++_static`。

```bash
假设当前位于处于 Paddle Lite 源码根目录下

导入 NDK_ROOT 变量，注意检查 NDK 安装目录若与本示例是否不同
$ export NDK_ROOT=/opt/android-ndk-r17c

删除上一次 CMake 自动生成的 `.h` 文件
$ rm ./lite/api/paddle_use_kernels.h
$ rm ./lite/api/paddle_use_ops.h

设置编译参数并开始编译
For android-armv8: cpu+cv+extra
$ ./lite/tools/build_android.sh \
    --arch=armv8 \
    --toolchain=clang \
    --with_log=OFF \
    --with_extra=ON \
    --with_cv=ON

For android-armv7: cpu+cv+extra
$ ./lite/tools/build_android.sh \
    --arch=armv7 \
    --toolchain=clang \
    --with_log=OFF \
    --with_extra=ON \
    --with_cv=ON

For android-armv8-(v8.2+FP16): cpu+FP16+cv+extra
update NDK version > 19
$ export NDK_ROOT=/opt/android-ndk-r20b
$ ./lite/tools/build_android.sh \
    --arch=armv8 \
    --toolchain=clang \
    --with_log=OFF \
    --with_extra=ON \
    --with_arm82_fp16=ON \
    --with_cv=ON

注：编译帮助请执行: `./lite/tools/build_android.sh` help
```

### 1.3 编译 Paddle Lite ARM LINUX 库范例

```bash
假设当前位于处于 Paddle Lite 源码根目录下

删除上一次 CMake 自动生成的 `.h` 文件
$ rm ./lite/api/paddle_use_kernels.h
$ rm ./lite/api/paddle_use_ops.h

设置编译参数并开始编译
For linux-armv8: cpu+cv+extra
$ ./lite/tools/build_linux.sh \
    --arch=armv8 \
    --toolchain=gcc \
    --with_log=OFF \
    --with_extra=ON \
    --with_cv=ON

For linux-armv7: cpu+cv+extra
$ ./lite/tools/build_linux.sh \
    --arch=armv7 \
    --toolchain=gcc \
    --with_log=OFF \
    --with_extra=ON \
    --with_cv=ON

For linux-armv7hf: cpu+cv+extra
$ ./lite/tools/build_linux.sh \
    --arch=armv7hf \
    --toolchain=gcc \
    --with_log=OFF \
    --with_extra=ON \
    --with_cv=ON

注：编译帮助请执行: `./lite/tools/build_linux.sh` help
```

### 1.4 编译产物说明

以ARM v8 Android为例，编译产物位于 `build.lite.android.armv8.clang` 下的 `lite` 文件夹内。这里仅罗列关键产物：

- `api`: 包含了基于 API 接口和模型的各种可执行的单测文件
- `tests`:该目录包含了多个层面的可执行的单测文件
   - `kernels`: 包含已支持 OP 的各种可执行的单测文件，如 `activation` OP 单测；
   - `benchmark`: 提供便利化脚本用于 convolution/pooling 等算子性能的批量测试
   - `math`: 包含各类卷积算子如 `GEMM`、`GEMV` 等可执行的单测文件
```bash
.
|-- api
|   |-- *.a
|   |-- *.so
|   |-- test_model_bin
|   |-- test_mobilenetv1
|   |-- test_mobilenetv1_int8
|   |....
|-- kernel
|   |-- apu
|   |-- arm
|   |   |-- *.a(example:libconv_compute_arm.a,  libmul_compute_arm.a etc.)
|   |-- bm
|   |-- cuda
|   |-- host
|   |....
|-- tests
|   |-- api
|   |   |-- test_inception_v4_fp32_arm
|   |   |-- test_mobilenet_v1_int8_dygraph_arm
|   |   |-- test_nlp_lstm_int8_arm
|   |   |...
|   |-- benchmark
|   |   |-- get_activation_latency
|   |   |-- get_batchnorm_latency
|   |   |-- get_conv_latency
|   |   |...
|   |-- cv
|   |   |-- image_convert_test
|   |   |-- image_profiler_test
|   |-- kernels
|   |   |-- test_kernel_activation_compute
|   |   |-- test_kernel_expand_as_compute
|   |   |-- test_kernel_group_norm_compute
|   |   |...
|   |-- math
|   |   |-- conv_compute_test
|   |   |-- sgemm_compute_test
|   |   |-- sgemv_compute_test
|   |   |...
....
```

## 2. 运行图像分类示例程序

- 下载示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；

`run_with_adb.sh` 只能在连接设备的系统上运行，不能在 Docker 环境执行（可能无法找到设备），也不能在设备上运行。入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。

  运行适用于 ARM CPU 的 mobilenetv1 模型
  ```shell
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  以armv8为例
  $ ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a

    Top1 Egyptian cat - 0.482871
    Top2 tabby, tabby cat - 0.471594
    Top3 tiger cat - 0.039779
    Top4 lynx, catamount - 0.002430
    Top5 ping-pong ball - 0.000508
    Preprocess time: 4.716000 ms, avg 4.716000 ms, max 4.716000 ms, min 4.716000 ms
    Prediction time: 33.408000 ms, avg 33.408000 ms, max 33.408000 ms, min 33.408000 ms
    Postprocess time: 4.499000 ms, avg 4.499000 ms, max 4.499000 ms, min 4.499000 ms
  ```
  运行适用于 ARM Linux 的 mobilenetv1 模型
  ```shell
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  以armv7hf为例
  $ ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux armhf
  ```
- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/inputs` 目录下，同时将图片文件名添加到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/list.txt` 中；

- 如果需要重新编译示例程序，直接运行

  ```shell
  For android+armv8-a
  $ ./build.sh android arm64-v8a

  For android+armv7
  $ ./build.sh android armeabi-v7a

  For linux+armv8
  $ ./build.sh linux arm64

  For linux+armv7hf
  $ ./build.sh linux armhf
  ```

### 更新支持Android+Arm CPU的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout develop
  ```

- 编译并生成 PaddleLite+Arm for android+armv8.2, android+armv8-a, android+armv7, linux+armv8 and linux+armhf的部署库

  - For android+armv8.2+fp16

    - tiny_publish 编译

      ```shell
      $ ./lite/tools/build_android.sh     --arch=armv8     --toolchain=clang     --with_log=OFF     --with_extra=ON     --with_arm82_fp16=ON
      ```

    - 替换头文件和库

      ```shell
      清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
      
      替换 include 目录
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
      
      替换 android 的 libpaddle_light_api_shared.so 动态库
      $ cp build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      ```
  - For android+armv8-a

    - tiny_publish 编译

      ```shell
      $ ./lite/tools/build_android.sh     --arch=armv8     --toolchain=clang     --with_log=OFF     --with_extra=ON     
      ```

    - 替换头文件和库

      ```shell
      清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
      
      替换 include 目录
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
      
      替换 android 的 libpaddle_light_api_shared.so 动态库
      $ cp build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      ```
  - For android+armv7

    - tiny_publish 编译

      ```shell
      $ ./lite/tools/build_android.sh     --arch=armv7     --toolchain=clang     --with_log=OFF     --with_extra=ON     --with_cv=ON
      ```

    - 替换头文件和库

      ```shell
      清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include/
      
      替换 include 目录
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include/
      
      替换 android 的 libpaddle_light_api_shared.so 动态库
      $ cp build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```
  - For linux+armv8

    - tiny_publish 编译

      ```shell
      $ ./lite/tools/build_linux.sh     --arch=armv8     --toolchain=gcc     --with_log=OFF     --with_extra=ON     --with_cv=ON
      ```

    - 替换头文件和库

      ```shell
      清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      
      替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      
      替换 android 的 libpaddle_light_api_shared.so 动态库
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```
  - For linux+armv7hf

    - tiny_publish 编译

      ```shell
      $ ./lite/tools/build_linux.sh     --arch=armv7hf     --toolchain=gcc     --with_log=OFF     --with_extra=ON     --with_cv=ON
      ```

    - 替换头文件和库

      ```shell
      清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      
      替换 include 目录
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      
      替换 android 的 libpaddle_light_api_shared.so 动态库
      $ cp build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      ```
## 3. 高级特性

Android 平台下分析：
### FP16预测库
- 目前支持以FP16精度进行推理，编译时需要加入--with_arm82_fp16=ON编译选项，同时交叉编译的ndk版本要保证NDK version > 19
- 如果运行 FP16 预测库，模型在 OPT 转换的时候需要加上 `--enable_fp16=1` 选项，这样转换的模型会选择 **FP16 kernel** 实现。并且，FP16 预测库和 FP16 模型**只在支持 ARMv8.2 架构的手机**上运行，如小米 9，华为 Meta30 等。
- 当前 Paddle Lite只支持 **ARMv8 架构**的 FP16 运算。

### 开启性能分析，会打印出每个 op 耗时信息和汇总信息

```bash
$ ./lite/tools/build_android.sh \
  --arch=armv8 \
  --toolchain=clang \
  --with_log=OFF \
  --with_extra=ON \
  --with_cv=ON \
  --with_profile=ON \
test
```

### 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息

```bash
# 开启性能分析，会打印出每个 op 耗时信息和汇总信息
$ ./lite/tools/build_android.sh \
  --arch=armv8 \
  --toolchain=gcc \
  --with_log=OFF \
  --with_extra=ON \
  --with_cv=ON \
  --with_profile=ON \
  --with_precision_profile=ON \
  test
```

详细输出信息的说明可查阅 [Profiler 工具](../user_guides/profiler)。
