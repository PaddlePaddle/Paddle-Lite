# 高通 QNN 部署示例

Paddle Lite 已支持高通 QNN 在 x86 （模拟器）和 ARM 设备（例如SA8295P）上进行预测部署。
目前支持子图接入方式，其接入原理是在线分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 Qualcomm QNN 组网 API 进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- 高通 8295 芯片

### 已支持的设备

- SA8295P

### 已支持的 Paddle 模型

- [mobilenet_v1_fp32_224](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224.tar.gz)
- [resnet50_fp32_224](https://paddlelite-demo.bj.bcebos.com/models/resnet50_fp32_224.tar.gz)
- [ssd_mobilenet_v1_relu_voc_fp32_300](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)
- [yolov3_darknet53_270e_coco](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz)
- [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)
- [resnet50_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/resnet50_int8_224_per_layer.tar.gz)
- [ssd_mobilenet_v1_relu_voc_int8_300_per_layer](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_int8_300_per_layer.tar.gz)

### 已支持（或部分支持）的 Paddle 算子（ Kernel 接入方式）

- 您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。


## 参考示例演示

### 测试设备( 高通 SA8295P)

<img src="https://cdn.shopify.com/s/files/1/0082/8762/products/adp8295-1.png?v=1643929832" alt="qualcomm_qnn" style="zoom:75%;" />

### 准备设备环境

- 设备安装 QNX 和 Android 双系统，网口和串口都已配置，配置详情请咨询高通。

### 运行图像分类示例程序

- 下载示例程序 [ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后清单如下：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - images
            - tabby_cat.jpg # 测试图片
            - tabby_cat.raw # 经过 convert_to_raw_image.py 处理后的 RGB Raw 图像
          - labels
            - synset_words.txt # 1000 分类 label 文件
          - models
            - resnet50_fp32_224 # Paddle non-combined 格式的 resnet50 float32 模型
              - __model__ # Paddle fluid 模型组网文件，可拖入 https://lutzroeder.github.io/netron/ 进行可视化显示网络结构
              - bn2a_branch1_mean # Paddle fluid 模型参数文件
              - bn2a_branch1_scale
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.linux.amd64 # 已编译好的，适用于 linux amd64
            - image_classification_demo # 已编译好的，适用于 linux amd64 的示例程序
          - build.qnx.arm64 # 已编译好的，适用于 qnx arm64
            - image_classification_demo # 已编译好的，适用于 qnx arm64 的示例程序
            ...
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序 ssh 运行脚本
          - run_with_adb.sh # 示例程序 adb 运行脚本
      - libs
        - PaddleLite
          - qnx
            - amd64
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - qualcomm_qnn  # 高通 QNN 运行时库、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libqualcomm_qnn.so # NNAdapter device HAL 库
                  - libqualcomm_qnn_cpu_custom_op_package.so # 高通 QNN CPU 自定义算子库
                  - libqualcomm_qnn_htp_custom_op_package.so # 高通 QNN HTP 自定义算子库
                  - libQnnHtp.so    # 下列为高通 QNN 在真机的 QNX 系统上运行时所需库
                  - libQnnCpu.so
                  - libQnn*.so
                  - hexagon-v68/lib/unsigned/ # 高通在 QNX 运行的 dsp 库
          - android
            - arm64-v8a
            - armeabi-v7a
          - linux
            - amd64
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - qualcomm_qnn  # 高通 QNN 运行时库、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libqualcomm_qnn.so # NNAdapter device HAL 库
                  - libqualcomm_qnn_cpu_custom_op_package.so # 高通 QNN CPU 自定义算子库
                  - libqualcomm_qnn_htp_custom_op_package.so # 高通 QNN HTP 自定义算子库
                  - libQnnHtp.so  # 下列为高通 QNN 在 x86 模拟器上运行时所需库
                  - libQnnCpu.so
                  - libQnn*.so
                - cpu
                	- libiomp5.so # Intel OpenMP 库
                	- libmklml_intel.so # Intel MKL 库
                	- libmklml_gnu.so # GNU MKL 库
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - arm64
              - include
              - lib
            - armhf
              ...
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
      - yolo_detection_demo # 基于 yolov3 的目标检测示例程序
      - model_test # 基于简单模型测试的示例程序
      - tools # 工具库，包含模型处理工具、编译脚本和代码风格检查工具
  ```

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；

- 执行以下命令分别比较 **mobilenet_v1_int8_224_per_layer** 和 **mobilenet_v1_fp32_224** 模型的性能和结果；

  - 运行 **mobilenet_v1_int8_224_per_layer** 模型

    - Intel CPU ( QNN x86 Simulator )

      ```shell
      $ unset FILE_TRANSFER_COMMAND
      $ ./run.sh mobilenet_v1_int8_224_per_layer linux amd64 qualcomm_qnn
        warmup: 1 repeat: 5, average: 1644.109009 ms, max: 1660.884033 ms, min: 1625.844971 ms
        results: 3
        Top0  tabby, tabby cat - 0.456282
        Top1  Egyptian cat - 0.456282
        Top2  tiger cat - 0.070197
        Preprocess time: 1.345000 ms
        Prediction time: 1644.109009 ms
        Postprocess time: 0.112000 ms
      ```

    - Qualcomm 8295P EVB ( Android )

      ```shell
      $ adb -s 858e5789 root
      $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android arm64-v8a qualcomm_qnn 858e5789
        warmup: 1 repeat: 5, average: 1.659400 ms, max: 2.342000 ms, min: 1.446000 ms
        results: 3
        Top0  Egyptian cat - 0.456282
        Top1  tabby, tabby cat - 0.456282
        Top2  tiger cat - 0.070197
        Preprocess time: 1.171000 ms
        Prediction time: 1.659400 ms
        Postprocess time: 0.187000 ms
      ```

    - Qualcomm 8295P EVB ( QNX )

      ```shell
      $ export FILE_TRANSFER_COMMAND=lftp
      $ adb -s 858e5789 root
      $ rm -rf ../assets/models/cache
      # 用于生成 cache 和 nb
      $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android arm64-v8a qualcomm_qnn 858e5789 null cache
      # 上板远程执行
      $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache
        warmup: 1 repeat: 5, average: 1.200000 ms, max: 2.000000 ms, min: 1.000000 ms
        results: 3
        Top0  Egyptian cat - 0.456282
        Top1  tabby, tabby cat - 0.456282
        Top2  tiger cat - 0.070197
        Preprocess time: 1.000000 ms
        Prediction time: 1.200000 ms
        Postprocess time: 0.000000 ms
      ```

  - 将测试模型改为 **mobilenet_v1_fp32_224**，执行命令：

    - Intel CPU ( QNN x86 Simulator )

      ```shell
      $ unset FILE_TRANSFER_COMMAND
      $ ./run.sh mobilenet_v1_fp32_224 linux amd64 qualcomm_qnn
        warmup: 1 repeat: 5, average: 686.677405 ms, max: 693.025024 ms, min: 680.283020 ms
        results: 3
        Top0  tabby, tabby cat - 0.529133
        Top1  Egyptian cat - 0.419681
        Top2  tiger cat - 0.045172
        Preprocess time: 0.951000 ms
        Prediction time: 686.677405 ms
        Postprocess time: 0.110000 ms
      ```

    - Qualcomm 8295P EVB ( Android )

      ```shell
      $ adb -s 858e5789 root
      $ ./run_with_adb.sh mobilenet_v1_fp32_224 android arm64-v8a qualcomm_qnn 858e5789
        warmup: 1 repeat: 5, average: 10893.910547 ms, max: 10955.571289 ms, min: 10795.490234 ms
        results: 3
        Top0  tabby, tabby cat - 0.529133
        Top1  Egyptian cat - 0.419681
        Top2  tiger cat - 0.045172
        Preprocess time: 1.193000 ms
        Prediction time: 10893.910547 ms
        Postprocess time: 0.213000 ms
      
      # 以 FP16 方式运行
      $ ./run_with_adb.sh mobilenet_v1_fp32_224 android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true"
        warmup: 1 repeat: 5, average: 1.883000 ms, max: 1.977000 ms, min: 1.847000 ms
        results: 3
        Top0  tabby, tabby cat - 0.531738
        Top1  Egyptian cat - 0.417480
        Top2  tiger cat - 0.045013
        Preprocess time: 1.329000 ms
        Prediction time: 1.883000 ms
        Postprocess time: 0.110000 ms
      ```

    - Qualcomm 8295P EVB ( QNX )

      ```shell
      $ export FILE_TRANSFER_COMMAND=lftp
      $ adb -s 858e5789 root
      $ rm -rf ../assets/models/cache
      # 用于生成 cache 和 nb
      $ ./run_with_adb.sh mobilenet_v1_fp32_224 android arm64-v8a qualcomm_qnn 858e5789 null cache
      # 上板远程执行
      $ ./run_with_ssh.sh mobilenet_v1_fp32_224 qnx arm64 qualcomm_qnn 192.168.1.1 22 root root null cache
        warmup: 1 repeat: 5, average: 10910.345508 ms, max: 11007.915039 ms, min: 10865.540039 ms
        results: 3
        Top0  tabby, tabby cat - 0.529133
        Top1  Egyptian cat - 0.419681
        Top2  tiger cat - 0.045172
        Preprocess time: 1.000000 ms
        Prediction time: 10910.345508 ms
        Postprocess time: 0.000000 ms
      
      # 以 FP16 方式运行
      $ rm -rf ../assets/models/cache
      $ ./run_with_adb.sh mobilenet_v1_fp32_224 android arm64-v8a qualcomm_qnn 858e5789 "QUALCOMM_QNN_ENABLE_FP16=true" cache
      $ ./run_with_ssh.sh mobilenet_v1_fp32_224 qnx arm64 qualcomm_qnn 192.168.1.1 22 root root "QUALCOMM_QNN_ENABLE_FP16=true" cache
        warmup: 1 repeat: 5, average: 1.600000 ms, max: 2.000000 ms, min: 1.000000 ms
        results: 3
        Top0  tabby, tabby cat - 0.531738
        Top1  Egyptian cat - 0.417480
        Top2  tiger cat - 0.045013
        Preprocess time: 1.000000 ms
        Prediction time: 1.600000 ms
        Postprocess time: 0.000000 ms
      ```

- 如果需要更改测试图片，请将图片拷贝到 **`PaddleLite-generic-demo/image_classification_demo/assets/images`** 目录下，修改并执行 **`convert_to_raw_image.py`** 生成相应的 RGB Raw 图像，最后修改 `run.sh` 的 IMAGE_NAME 即可；

- 如果需要重新编译示例程序，直接运行

  ```shell
  # linux amd64
  $ ./build.sh linux amd64
  # android arm64-v8a  需设置 Android ndk 交叉编译环境
  $ ./build.sh android arm64-v8a
  # qnx arm64  需设置 qnx 交叉编译环境
  $ ./build.sh qnx arm64
  ```

### 准备编译环境

- 为了保证编译环境一致，建议根据下述约束进行环境配置。

  ```shell
  cmake版本：3.16 # 下载参考链接：https://cmake.org/files/v3.16/cmake-3.16.0-rc1-Linux-x86_64.tar.gz
  gcc版本：7.1以上(8.4.0、9.3.0已验证) # 需要支持c++17
  clang版本：6.0以上（9.0已验证） # 需要支持c++17
  ```

### 更新支持 Qualcomm Qnn 的 Paddle Lite 库

- 下载 Paddle Lite 源码；

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  ```

- 获取 qualcomm_qnn 适配源码（暂未开源）；

  ```shell
  # 在 Paddle-Lite/lite/backends/nnadapter/nnadapter/src/driver 目录里下载 qualcomm_qnn 代码
  ```

- 请向高通索取 QNN SDK ，解压后目录为 qnn-v1.15.0.220706112757_38277;

- 请向高通索取 Hexagon SDK，解压后目录为：Hexagon_SDK；

- 下述为各个平台下的预测库编译命令，根据自身所需进行编译。

- **编译 Linux x86 simulator 预测库**

  ```shell
  # 注：编译时 nnadapter_qualcomm_qnn_sdk_root 和 nnadapter_qualcomm_hexagon_sdk_root 两个变量需要使用绝对路径，采用相对路径可能会产生编译问题
  # 注：请使用 clang 编译
  $ export CC=<path/to/clang>
  $ export CXX=<path/to/clang++>
  $ cd Paddle-Lite
  $ ./lite/tools/build_linux.sh \
      --arch=x86 \
      --with_extra=ON \
      --with_log=ON \
      --toolchain=clang \
      --with_exception=ON \
      --with_nnadapter=ON \
      --nnadapter_with_qualcomm_qnn=ON \
      --nnadapter_qualcomm_qnn_sdk_root=<path/to/qnn-v1.15.0.220706112757_38277> \
      --nnadapter_qualcomm_hexagon_sdk_root=<path/to/Hexagon_SDK/4.3.0.0> \
      full_publish
  # 替换 x86 simulator预测库
  $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include
  $ cp -rf build.lite.linux.x86.clang/inference_lite_lib/cxx/include PaddleLite-generic-demo/libs/PaddleLite/linux/amd64
  $ cp build.lite.linux.x86.clang/inference_lite_lib/cxx/lib/libpaddle*.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
  $ cp build.lite.linux.x86.clang/inference_lite_lib/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/qualcomm_qnn/
  $ cp build.lite.linux.x86.clang/inference_lite_lib/cxx/lib/libqualcomm_qnn* PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/qualcomm_qnn/
  # 将高通 QNN SDK 中的依赖的库拷贝到 demo 程序中 
  $ cp <path/to/qnn-v1.15.0.220706112757_38277>/target/x86_64-linux-clang/lib/* PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/qualcomm_qnn
  ```

- **编译 Android arm64-v8a/ameabi-v7a 预测库**

  ```shell
  # Android arm64-v8a
  $ cd Paddle-Lite
  $ ./lite/tools/build_android.sh \
      --arch=armv8 \
      --with_extra=ON \
      --with_log=ON \
      --toolchain=clang \
      --with_exception=ON \
      --with_nnadapter=ON \
      --android_stl=c++_shared \
      --nnadapter_with_qualcomm_qnn=ON \
      --nnadapter_qualcomm_qnn_sdk_root=<path/to/qnn-v1.15.0.220706112757_38277> \
      --nnadapter_qualcomm_hexagon_sdk_root=<path/to/Hexagon_SDK/4.3.0.0> \
      full_publish
  # 替换 Android arm64-v8a 预测库
  $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include
  $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
  $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/qualcomm_qnn/
  $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libqualcomm_qnn* PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/qualcomm_qnn/
  $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libpaddle*.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
  # 将高通 QNN SDK 中的依赖的库拷贝到 demo 程序中
  $ cp <path/to/qnn-v1.15.0.220706112757_38277>/target/aarch64-android/lib/*  PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/qualcomm_qnn
  $ cp <path/to/qnn-v1.15.0.220706112757_38277>/target/hexagon-v68/lib/unsigned/* PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/qualcomm_qnn/hexagon-v68/lib/unsigned

  # Android ameabi-v7a
  $ cd Paddle-Lite
  $ ./lite/tools/build_android.sh \
      --arch=armv7 \
      --with_extra=ON \
      --with_log=ON \
      --android_stl=c++_shared \
      --toolchain=clang \
      --with_exception=ON \
      --with_nnadapter=ON \
      --nnadapter_with_qualcomm_qnn=ON \
      --nnadapter_qualcomm_qnn_sdk_root=<path/to/qnn-v1.15.0.220706112757_38277> \
      --nnadapter_qualcomm_hexagon_sdk_root=<path/to/Hexagon_SDK/4.3.0.0> \
      full_publish
  # 替换 Android ameabi-v7a 预测库
  $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include
  $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include/
  $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/qualcomm_qnn/
  $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libqualcomm_qnn* PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/qualcomm_qnn/
  $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle*.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
  #将高通 QNN SDK 中的依赖的库拷贝到 demo 程序中
  $ cp <path/to/qnn-v1.15.0.220706112757_38277>/target/arm-android/lib/*  PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/qualcomm_qnn
  $ cp <path/to/qnn-v1.15.0.220706112757_38277>/target/hexagon-v68/lib/unsigned/*  PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/qualcomm_qnn/hexagon-v68/lib/unsigned
  ```

- **编译 QNX arm64 预测库**

  1. 请向高通索取 QNX 工具链，解压后目录为：SDP;

  2. 请向高通索取 License，解压后目录为：qnx-key；

  3. 设置 QNN 交叉编译环境

    ```shell
  $ source SDP/qnx710/qnxsdp-env.sh
  $ cd qnx-key && tar xvf qnx_license_700_710.tar.gz && cd ..
  $ export QNX_CONFIGURATION=$(pwd)/qnx-key/home/jone/.qnx
  $ cp -r $(pwd)/qnx-key/home/jone/.qnx ~/
    ```

  4. 编译 QNX 预测库

    ```shell
  $ cd Paddle-Lite
  $ ./lite/tools/build_qnx.sh \
      --with_extra=ON \
      --with_log=ON \
      --with_nnadapter=ON \
      --nnadapter_with_qualcomm_qnn=ON \
      --nnadapter_qualcomm_qnn_sdk_root=<path/to/qnn-v1.15.0.220706112757_38277> \
      --nnadapter_qualcomm_hexagon_sdk_root=<path/to/Hexagon_SDK/4.3.0.0>\
      full_publish
  # 替换 QNX 预测库
  $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/qnx/arm64/include
  $ cp -rf build.lite.qnx.armv8.gcc/inference_lite_lib.qnx.armv8.nnadapter/cxx/include PaddleLite-generic-demo/libs/PaddleLite/qnx/arm64
  $ cp build.lite.qnx.armv8.gcc/inference_lite_lib.qnx.armv8.nnadapter/cxx/lib/libpaddle*.so PaddleLite-generic-demo/libs/PaddleLite/qnx/arm64/lib/
  $ cp build.lite.qnx.armv8.gcc/inference_lite_lib.qnx.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/qnx/arm64/lib/qualcomm_qnn/
  $ cp build.lite.qnx.armv8.gcc/inference_lite_lib.qnx.armv8.nnadapter/cxx/lib/libqualcomm_qnn* PaddleLite-generic-demo/libs/PaddleLite/qnx/arm64/lib/qualcomm_qnn/
    ```
  5. 将高通 QNN SDK 中的依赖的库拷贝到 demo 程序中 
    ```
      $ cp <path/to/qnn-v1.15.0.220706112757_38277>/target/aarch64-qnx/lib/*  PaddleLite-generic-demo/libs/PaddleLite/qnx/arm64/lib/qualcomm_qnn
      $ cp <path/to/qnn-v1.15.0.220706112757_38277>/target/hexagon-v68/lib/unsigned/*  PaddleLite-generic-demo/libs/PaddleLite/qnx/arm64/lib/qualcomm_qnn/hexagon-v68/lib/unsigned
    ```


## 高级特性

- 高级参数

  - QUALCOMM_QNN_DEVICE_TYPE:

    指定使用的高通设备，Options: "CPU", "GPU", "DSP", "HTP"。

  - QUALCOMM_QNN_LOG_LEVEL:

    指定日志等级，Options: "error", "warn", "info", "verbose", "debug"

  - QUALCOMM_QNN_SKIP_SYMM2ASYMM

    指定是否跳过将输入输出从对称量化转非对称量化的步骤，Options:  "true", "false", "1", "0"。

  - QUALCOMM_QNN_ENABLE_FP16

    指定是否开启 FP16 功能，Options:  "true", "false", "1", "0"。

## FAQ
### 1. 在 QNX 系统上面使用 HTP 推理, 设置 DSP 运行库路径变量
- 指定 DSP 运行库设置如下变量
  ```
  ADSP_LIBRARY_PATH
  或者
  CDSP0_LIBRARY_PATH
  CDSP1_LIBRARY_PATH
  ```

- 如果设置上述变量后，还是会有环境问题导致推理异常（非 Demo 本身问题），该问题一般发生在 QNN SDK 版本切换时(例如: QNN SDK v1.12 <-> QNN SDK v1.15 版本之间的切换)，可以将 <path/to/qnn-v1.15.0.220706112757_38277>/target/hexagon-v68/lib/unsigned/libQnnHtp* 拷贝到板子上 /mnt/etc/images/cdsp0 路径下。如果遇到权限问题拷贝失败，可以通过在 QNX 中执行 `mount -uw /mnt` 命令解决。

## 其它说明

- 如需更进一步的了解相关部署细节，请联系 shentanyue01@baidu.com；
