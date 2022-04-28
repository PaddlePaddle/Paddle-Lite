# Android NNAPI 部署示例

Paddle Lite 已支持 Android NNAPI 的预测部署。
其接入原理是加载并分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 Android NNAPI 进行网络构建，在线生成并执行模型。
需要注意由于不同 SoC 芯片对 Android NNAPI 的支持程度不同，其底层各个计算 IP（CPU、GPU、DSP、NPU 等）对 Android NNAPI 的支持也不同，性能也会有较大区别。
用户可以在运行日志中，搜索关键字『Available devices:』来查看当前 SoC 对 Android NNAPI 的支持情况。

## 支持现状

### 已支持的设备

- 所有支持 Android NNAPI 的终端设备，系统版本 Android 8.1(Oreo) 及以上，Android SDK version 需在 27 及以上。

### 已支持的 Paddle 模型

#### 模型
- [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)
- [resnet50_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/resnet50_int8_224_per_layer.tar.gz)
- [ssd_mobilenet_v1_relu_int8_300_per_layer](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_int8_300_per_layer.tar.gz)
- [mobilenet_v1_fp32_224](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224.tar.gz)
- [resnet50_fp32_224](https://paddlelite-demo.bj.bcebos.com/models/resnet50_fp32_224.tar.gz)
- [ssd_mobilenet_v1_relu_voc_fp32_300](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)

#### 性能
- 测试环境
  - 编译环境
    - Ubuntu 16.04，NDK-r17c with GCC for Android armeabi-v7a

  - 硬件环境
    - 小米10
      - 高通 骁龙865
      - Android SDK version 29
    - 华为P40pro
      - 华为 Kirin990-5G
      - Android SDK version 29

- 测试方法
  - warmup=1，repeats=5，统计平均时间，单位是 ms
  - 线程数为 1，`paddle::lite_api::PowerMode CPU_POWER_MODE` 设置为 ` paddle::lite_api::PowerMode::LITE_POWER_HIGH`
  - 分类模型的输入图像维度是{1, 3, 224, 224}，检测模型的维度是{1, 3, 300, 300}
  - 华为 Kirin NPU 对 Android NNAPI 的支持程度较高，但是由于其量化方式与 Paddle 有较大出入，量化模型无法发挥 NPU 加速特性，所以 fp32 模型性能较好
  - 高通 骁龙系列芯片（855 以后），DSP，GPU 等 IP 支持 Android NNAPI，但其 HTA|HTP 暂不支持 Android NNAPI
  - 不同 SoC 对 Android NNAPI 的支持程度不同，如下仅举例华为 Kirin990-5G 和 高通 骁龙865

- 测试结果

  |模型 |||
  |---|---|---|
  |  |骁龙865(ms) | Kirin990-5G(ms) |
  |mobilenet_v1_int8_224_per_layer|  10.109|  47.903|
  |resnet50_int8_224_per_layer|  18.622|  354.566|
  |ssd_mobilenet_v1_relu_voc_int8_300_per_layer|  23.214|  68.312|
  |mobilenet_v1_fp32_224|  12.250|  3.563|
  |resnet50_fp32_224|  44.903|  8.762|
  |ssd_mobilenet_v1_relu_voc_fp32_300|  23.112|  8.647|

### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

### 测试设备( 小米10)

### 准备设备环境

- 进入手机shell 输入命令 getprop ro.build.version.sdk 查看 Android SDK version，需在 27 及以上

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[编译环境准备](../source_compile/docker_env)中的 Docker 开发环境进行配置。

### 运行图像分类示例程序

- 下载 Paddle Lite 通用示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - images
            - tabby_cat.jpg # 测试图片
            - tabby_cat.raw # 经过 convert_to_raw_image.py 处理后的 RGB Raw 图像
          - labels
            - synset_words.txt # 1000 分类 label 文件
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.android.arm64-v8a # arm64-v8a 编译工作目录
            - image_classification_demo # 已编译好的，适用于 amd64-v8a 的示例程序
          - build.android.armeabi-v7a # armeabi-v7a 编译工作目录
            - image_classification_demo # 已编译好的，适用于 arm64 的示例程序
            ...
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run_with_adb.sh # 示例程序 adb 运行脚本
          — run_with_ssh.sh # 示例程序 ssh 运行脚本
          - run.sh # 示例程序运行脚本
      - libs
        - PaddleLite
          - android
            - armeabi-v7a
              - include
              - lib
                - android_nnapi # NNAdapter 运行时库、Android NNAPI device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libandroid_nnapi.so # NNAdapter device HAL 库
              - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
              - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
              - libc++_shared.so
            ...
        - OpenCV # OpenCV 预编译库
  ```

- Android shell 端的示例程序
  - 按照以下命令分别运行转换后的 ARM CPU 模型和 Android NNAPI 模型，比较它们的性能和结果；

    ```shell
    注意：
    1）`run_with_adb.sh` 不能在 Docker 环境执行，否则可能无法找到设备，也不能在设备上运行。
    2）`run_with_ssh.sh` 不能在设备上运行，且执行前需要配置目标设备的 IP 地址、SSH 账号和密码。
    3）`build.sh` 根据入参生成针对不同操作系统、体系结构的二进制程序，需查阅注释信息配置正确的参数值。
    4）`run_with_adb.sh` 入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。
    5）`run_with_ssh.sh` 入参包括模型名称、操作系统、体系结构、目标设备、ip地址、用户名、用户密码等，需查阅注释信息配置正确的参数值。
    6）下述命令行示例中涉及的设备序列号等均为示例环境，请用户根据自身实际设备环境修改。
    在 ARM CPU 上运行 mobilenetv1 全量化模型
    $ cd PaddleLite-generic-demo/image_classification_demo/shell
    $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a cpu d3869b25
      ...
      iter 0 cost: 102.673004 ms
      iter 1 cost: 102.539001 ms
      iter 2 cost: 102.505005 ms
      iter 3 cost: 102.626007 ms
      iter 4 cost: 102.735992 ms
      warmup: 1 repeat: 5, average: 102.615802 ms, max: 102.735992 ms, min: 102.505005 ms
      results: 3
      Top0  Egyptian cat - 0.512545
      Top1  tabby, tabby cat - 0.402567
      Top2  tiger cat - 0.067904
      Preprocess time: 2.070000 ms
      Prediction time: 102.615802 ms
      Postprocess time: 0.280000 ms

    基于 Android NNAPI 上运行 mobilenetv1 全量化模型
    $ cd PaddleLite-generic-demo/image_classification_demo/shell
    $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a android_nnapi d3869b25
      ...
      iter 0 cost: 10.193001 ms
      iter 1 cost: 10.142000 ms
      iter 2 cost: 11.538000 ms
      iter 3 cost: 9.292000 ms
      iter 4 cost: 10.304001 ms
      warmup: 1 repeat: 5, average: 10.293800 ms, max: 14.538000 ms, min: 9.292000 ms
      results: 3
      Top0  Egyptian cat - 0.672723
      Top1  tabby, tabby cat - 0.672723
      Top2  tiger cat - 0.128695
      Preprocess time: 2.098000 ms
      Prediction time: 10.293800 ms
      Postprocess time: 0.260000 ms
    ```

  - 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/images` 目录下，然后调用 `convert_to_raw_image.py` 生成相应的 RGB Raw 图像，最后修改 `run_with_adb.sh` 的 IMAGE_NAME 变量即可；
  - 重新编译示例程序：
  ```shell
  注意：
  1）请根据 `buid.sh` 配置正确的参数值。
  2）需在 Docker 环境中编译。
  
  # 对于 arm64-v8a
  ./build.sh android arm64-v8a
  
  # 对于 armeabi-v7a
  ./build.sh android armeabi-v7a
  ```

### 更新模型

- 通过 Paddle 训练，或 X2Paddle 转换得到 MobileNetv1 foat32 模型[ mobilenet_v1_fp32_224_fluid ](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)；
- 如果需要使用量化模型，则参考[模型量化-静态离线量化](../user_guides/quant_aware)使用 PaddleSlim 对 `float32` 模型进行量化（注意：由于 DSP 只支持量化 OP，在启动量化脚本时请注意相关参数的设置），最终得到全量化MobileNetV1 模型[ mobilenet_v1_int8_224_per_layer ](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)；
- 参考[模型转化方法](../user_guides/model_optimize_tool)，利用 opt 工具转换生成 Android NNAPI 模型，仅需要将 `valid_targets` 设置为 andriod_nnapi, arm 即可。

  ```shell
  # 注意：
  1）PaddleLite-generic-demo 中已经包含了类似 opt 工具优化生成 nb 模型的功能。

  $ cd PaddleLite-generic-demo/image_classification_demo/assets/models
  $ ./opt --model_dir=mobilenet_v1_int8_224_per_layer \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=android_nnapi,arm
  ```

- 注意：opt 生成的模型只是标记了 Android NNAPI 支持的 Paddle 算子，并没有真正生成 Android NNAPI 模型，只有在执行时才会将标记的 Paddle 算子转成 `Android NNAPI` 调用实现组网，最终生成并执行模型。

### 更新支持 Android NNAPI 的 Paddle Lite 库

- 下载 Paddle Lite 源码；

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 `PaddleLite+Android NNAPI` for armv8 and armv7 的部署库

  - For armv8
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --android_stl=c++_shared --with_extra=ON --with_cv=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_android_nnapi=ON
      ```

    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --android_stl=c++_shared --with_extra=ON --with_cv=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_android_nnapi=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/android_nnapi/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libandroid_nnapi.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/android_nnapi/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      ```

  - For armv7
    - tiny_publis h编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_cv=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_android_nnapi=ON
      ```
    
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_cv=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_android_nnapi=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/android_nnapi/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libandroid_nnapi.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/android_nnapi/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```

- 替换头文件后需要重新编译示例程序

## 其它说明

- 不同 SoC 芯片对 Android NNAPI 的支持差异较大，目前移动端芯片支持较好，边缘端、车载等需要根据具体芯片情况确认。
