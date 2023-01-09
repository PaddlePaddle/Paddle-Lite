# Arm

Paddle Lite 支持在 Android/iOS/ARMLinux 等移动端设备上运行高性能的 CPU 预测库，目前支持 Ubuntu 环境下 armv8、armv7 的交叉编译。

## 支持现状

### 已支持的芯片

- 高通 888+/888/Gen1/875/865/855/845/835/625/8155/8295P 等
- 麒麟 810/820/985/990/990 5G/9000E/9000 等

### 已支持的设备

- HUAWEI Mate 30 系列，荣耀 V20 系列，nova 6 系列，P40 系列，Mate Xs
- HUAWEI nova 5 系列，nova 6 SE，荣耀 9X 系列，荣耀 Play4T Pro
- 小米 6，小米 8，小米 10，小米 12，小米 MIX2，红米 10X，红米 Note8pro
- 高通 8295 EVK

### 已验证支持的 Paddle 模型

#### 模型
- 图像分类
  - [DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DarkNet/DarkNet53.tar.gz)
  - [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz)
  - [DPN68](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DPN/DPN68.tar.gz)
  - [EfficientNetB0](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/EfficientNetB0.tgz)
  - [GhostNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/GhostNet/GhostNet_x1_0.tar.gz)
  - [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz)
  - [HRNet-W18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/HRNet/HRNet_W18_C.tar.gz)
  - [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz)
  - [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz)
  - [MobileNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV1.tgz)
  - [MobileNet-v2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV2.tgz)
  - [MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_large_x1_0.tgz)
  - [MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_small_x1_0.tgz)
  - [PP-LCNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/PPLCNet/PPLCNet_x0_25.tar.gz)
  - [Res2Net50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/Res2Net/Res2Net50_26w_4s.tar.gz)
  - [ResNet-101](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz)
  - [ResNet-18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet18.tgz)
  - [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz)
  - [ResNeXt50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz)
  - [SE_ResNet50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/SENet/SE_ResNet50_vd.tar.gz)
  - [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz)
- 目标检测
  - [PPYOLO_tiny](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_tiny_650e_coco.tar.gz)
  - [ssd_mobilenet_v1_relu_voc_fp32_300](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)
  - [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz)
  - [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v3_large_270e_coco.tgz)
  - [YOLOv3-ResNet50_vd](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz)
- 姿态检测
  - [PP-TinyPose](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/tinypose_128x96.tar.gz)
- 关键点检测
  - [HigherHRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/higherhrnet_hrnet_w32_640.tgz)
  - [HRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/hrnet_w32_384x288.tgz)
- 文本检测 & 文本识别 & 端到端检测识别
  - [ch_ppocr_mobile_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_det_infer.tgz)
  - [ch_ppocr_mobile_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_rec_infer.tgz)
  - [ch_ppocr_server_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz)
  - [ch_ppocr_server_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz)
  - [ch_PP-OCRv2_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz)
  - [CRNN-mv3-CTC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz)
- 生成网络
  - [ESRGAN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleGAN/esrgan_psnr_x4_div2k.tgz)
- 视频分类
  - [PP-TSN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz)

## 参考示例演示

### 测试设备
- Android arm64-v8a/armeabi-v7a: HUAWEI P40pro
- Linux arm64: RK3399
- Linux armhf: Raspberry Pi 4B

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考 [Docker 统一编译环境搭建](../source_compile/docker_env) 中的 Docker 开发环境进行配置。

### 运行图像分类示例程序

- 下载 Paddle Lite 通用示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - configs
            - imagenet_224.txt # config 文件
            - synset_words.txt # 1000 分类 label 文件
          - datasets
            - test # dataset
              - inputs
                - tabby_cat.jpg # 输入图片
              - outputs
                - tabby_cat.jpg # 输出图片
              - list.txt # 图片清单
          - models
            - resnet50_fp32_224 # Paddle non-combined 格式的 resnet50 float32 模型
              - __model__ # Paddle fluid 模型组网文件，可拖入 https://lutzroeder.github.io/netron/ 进行可视化显示网络结构
              - bn2a_branch1_mean # Paddle fluid 模型参数文件
              - bn2a_branch1_scale
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.linux.amd64 # 已编译好的，适用于 amd64
            - demo # 已编译好的，适用于 amd64 的示例程序
          - build.linux.arm64 # 已编译好的，适用于 arm64
            - demo # 已编译好的，适用于 arm64 的示例程序
            ...
          ...
          - demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序 ssh 运行脚本
          - run_with_adb.sh # 示例程序 adb 运行脚本
      - libs
        - PaddleLite
          - android
            - arm64-v8a
            - armeabi-v7a
          - linux
            - arm64
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - armhf
              - include
              - lib
        - OpenCV # OpenCV 预编译库
      - object_detection_demo # 目标检测示例程序
  ```

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；

- 执行以下命令观察 mobilenet_v1_int8_224_per_layer 模型的性能和结果；

  ```shell
  运行 mobilenet_v1_int8_224_per_layer 模型

  For android arm64-v8a
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android arm64-v8a cpu <adb设备号>

    Top1 Egyptian cat - 0.503239
    Top2 tabby, tabby cat - 0.419854
    Top3 tiger cat - 0.065506
    Top4 lynx, catamount - 0.007992
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000494
    [0] Preprocess time: 6.712000 ms Prediction time: 16.859000 ms Postprocess time: 6.026000 ms
    Preprocess time: avg 6.712000 ms, max 6.712000 ms, min 6.712000 ms
    Prediction time: avg 16.859000 ms, max 16.859000 ms, min 16.859000 ms
    Postprocess time: avg 6.026000 ms, max 6.026000 ms, min 6.026000 ms

  For android armeabi-v7a
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android armeabi-v7a cpu <adb设备号>

    Top1 Egyptian cat - 0.502124
    Top2 tabby, tabby cat - 0.413927
    Top3 tiger cat - 0.071703
    Top4 lynx, catamount - 0.008436
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000563
    [0] Preprocess time: 6.717000 ms Prediction time: 44.779000 ms Postprocess time: 6.444000 ms
    Preprocess time: avg 6.717000 ms, max 6.717000 ms, min 6.717000 ms
    Prediction time: avg 44.779000 ms, max 44.779000 ms, min 44.779000 ms
    Postprocess time: avg 6.444000 ms, max 6.444000 ms, min 6.444000 ms

  For linux arm64
  本地执行
  $ ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 cpu
  通过 SSH 远程执行
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 cpu <IP地址> 22 <用户名> <密码>

    Top1 Egyptian cat - 0.503239
    Top2 tabby, tabby cat - 0.419854
    Top3 tiger cat - 0.065506
    Top4 lynx, catamount - 0.007992
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000494
    Preprocess time: 12.637000 ms, avg 12.637000 ms, max 12.637000 ms, min 12.637000 ms
    Prediction time: 78.751000 ms, avg 78.751000 ms, max 78.751000 ms, min 78.751000 ms
    Postprocess time: 9.969000 ms, avg 9.969000 ms, max 9.969000 ms, min 9.969000 ms

  For linux armhf
  本地执行
  $ ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux armhf cpu
  通过 SSH 远程执行
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux armhf cpu <IP地址> 22 <用户名> <密码>

    Top1 Egyptian cat - 0.502124
    Top2 tabby, tabby cat - 0.413927
    Top3 tiger cat - 0.071703
    Top4 lynx, catamount - 0.008436
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000563
    Preprocess time: 12.541000 ms, avg 12.541000 ms, max 12.541000 ms, min 12.541000 ms
    Prediction time: 96.863000 ms, avg 96.863000 ms, max 96.863000 ms, min 96.863000 ms
    Postprocess time: 13.324000 ms, avg 13.324000 ms, max 13.324000 ms, min 13.324000 ms
  ```

- 如果需要更改测试模型为 resnet50 ，执行命令修改为如下：

  ```shell
  For android arm64-v8a
  $ ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a cpu <adb设备号>

  For android armeabi-v7a
  $ ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android armeabi-v7a cpu <adb设备号>

  For linux arm64
  本地执行
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 cpu
  通过 SSH 远程执行
  $ ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 cpu <IP地址> 22 <用户名> <密码>

  For linux armhf
  本地执行
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux armhf cpu
  通过 SSH 远程执行
  $ ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux armhf cpu <IP地址> 22 <用户名> <密码>
  ```

- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/inputs` 目录下，同时将图片文件名添加到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/list.txt` 中；

- 如果需要重新编译示例程序，直接运行

  ```shell
  For android arm64-v8a
  $ ./build.sh android arm64-v8a

  For android armeabi-v7a
  $ ./build.sh android armeabi-v7a

  For linux arm64
  $ ./build.sh linux arm64
  
  For linux armhf
  $ ./build.sh linux armhf
  ```

### 更新支持 Arm 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 armv8 和 armv7 的部署库

  - For android arm64-v8a（注：--with_arm82_fp16=ON 编译选项可在部分机型启用 FP16 能力，但要求 NDK 版本 > 19 ）
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv8 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv8 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      ```

  - For android armeabi-v7a（注：--with_arm82_fp16=ON 编译选项可在部分机型启用 FP16 能力，但要求 NDK 版本 > 19 ）
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```

- 编译并生成 arm64 和 armhf 的部署库

  - For linux arm64
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_cv=ON --with_exception=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_cv=ON --with_exception=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

  - For linux armhf
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_cv=ON --with_exception=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_cv=ON --with_exception=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      ```

- 替换头文件后需要重新编译示例程序

## 高级特性

- 性能分析和精度分析

  android 平台下分析：

  - 开启性能分析，会打印出每个 op 耗时信息和汇总信息

  ```bash
  $ ./lite/tools/build.sh \
    --arm_os=android \
    --arm_abi=armv8 \
    --build_extra=on \
    --build_cv=on \
    --arm_lang=clang \
    --with_profile=ON \
  test
  ```

  - 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息

  ```bash
  # 开启性能分析，会打印出每个 op 耗时信息和汇总信息
  $ ./lite/tools/build.sh \
    --arm_os=android \
    --arm_abi=armv8 \
    --build_extra=on \
    --build_cv=on \
    --arm_lang=clang \
    --with_profile=ON \
    --with_precision_profile=ON \
    test
  ```

  详细输出信息的说明可查阅 [Profiler 工具](../user_guides/profiler)。

- FP16 模型推理

  - 单测编译的时候，需要添加 `--build_arm82_fp16=ON` 选项，即：

  ```bash
  $ export NDK_ROOT=/disk/android-ndk-r20b #ndk_version > 19
  $ ./lite/tools/build.sh \
    --arm_os=android \
    --arm_abi=armv8 \
    --build_extra=on \
    --build_cv=on \
    --arm_lang=clang \
    --build_arm82_fp16=ON \
    test
  ```

  - 模型在 OPT 转换的时候，需要添加 `--enable_fp16=1` 选项，完成 FP16 模型转换，即：

  ```bash
  $ ./build.opt/lite/api/opt \
    --optimize_out_type=naive_buffer \
    --enable_fp16=1 \
    --optimize_out caffe_mv1_fp16 \
    --model_dir ./caffe_mv1
  ```

  - 执行

    - 推送 OPT 转换后的模型至设备, 运行时请将 `use_optimize_nb` 设置为1

    ```bash
    将转换好的模型文件推送到 `/data/local/tmp/arm_cpu` 目录下
    $ adb push caffe_mv1_fp16.nb /data/local/tmp/arm_cpu/
    $ adb shell chmod +x /data/local/tmp/arm_cpu/test_mobilenetv1

    $ adb shell "\
       /data/local/tmp/arm_cpu/test_mobilenetv1 \
       --use_optimize_nb=1 \
       --model_dir=/data/local/tmp/arm_cpu/caffe_mv1_fp16 \
       --input_shape=1,3,224,224 \
       --warmup=10 \
       --repeats=100"
    ```

    - 推送原始模型至设备, 运行时请将 `use_optimize_nb` 设置为0， `use_fp16` 设置为1；（`use_fp16` 默认为0）

    ```bash
    将 fluid 原始模型文件推送到 `/data/local/tmp/arm_cpu` 目录下
    $ adb push caffe_mv1 /data/local/tmp/arm_cpu/
    $ adb shell chmod +x /data/local/tmp/arm_cpu/test_mobilenetv1

    $ adb shell "export GLOG_v=1; \
        /data/local/tmp/arm_cpu/test_mobilenetv1 \
        --use_optimize_nb=0 \
        --use_fp16=1 \
        --model_dir=/data/local/tmp/arm_cpu/caffe_mv1 \
        --input_shape=1,3,224,224 \
        --warmup=10 \
       --repeats=100"
    ```

  注：如果想输入真实数据，请将预处理好的输入数据用文本格式保存。在执行的时候加上 `--in_txt=./*.txt` 选项即可
