# 华为麒麟 NPU

Paddle Lite 是首款支持华为自研达芬奇架构 NPU（Kirin 810/990 SoC 搭载的 NPU）的预测框架。
原理是在线分析 Paddle 模型，首先将 Paddle 算子转成 NNAdapter 标准算子，其次再转换为 HiAI IR，最后调用HiAI IR/Builder/Runtime APIs 生成并执行 HiAI 模型。

## 支持现状

### 已支持的芯片

- Kirin 810/820/985/990/990 5G/9000E/9000

### 已支持的设备

- Kirin 9000：HUAWEI Mate 40pro 系列
- Kirin 9000E：HUAWEI Mate 40 系列
- Kirin 990 5G：HUAWEI Mate 30pro 系列，P40pro 系列
- Kirin 990：HUAWEI Mate 30 系列，荣耀 V20 系列，nova 6 系列，P40 系列，Mate Xs
- Kirin 985：HUAWEI nova 7 5G，nova 7 Pro 5G，荣耀 30
- Kirin 820：HUAWEI nova 7 SE 5G，荣耀 30S
- Kirin 810：HUAWEI nova 5 系列，nova 6 SE，荣耀 9X 系列，荣耀 Play4T Pro

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

#### 性能
- 测试环境
  - 编译环境
    - Ubuntu 16.04，NDK-r17c with GCC for Android arm64-v8a
    - HIAI DDK 版本：v510

  - 硬件环境
    - Kirin 810
      - HUAWEI Nova 5，Kirin 810
      - CPU：2 x Cortex A76 2.27GHz + 6 x Cortex A55 1.88GHz
      - NPU：Da Vinci 架构，1 x Ascend D100 Lite

    - Kirin 990
      - HUAWEI Mate 30，Kirin 990
      - CPU：2 x Cortex-A76 Based 2.86 GHz + 2 x Cortex-A76 Based 2.09 GHz + 4 x Cortex-A55 1.86 GHz
      - NPU：Da Vinci 架构，1  x Ascend Lite + 1 x Ascend Tiny

    - Kirin 990 5G
      - HUAWEI P40pro，Kirin 990 5G
      - CPU：2 x Cortex-A76 Based 2.86GHz + 2 x Cortex-A76 Based 2.36GHz + 4 x Cortex-A55 1.95GHz
      - NPU：Da Vinci 架构，2 x Ascend Lite + 1 x Ascend Tiny

- 测试方法
  - warmup=1, repeats=5，统计平均时间，单位是 ms
  - 线程数为1，`paddle::lite_api::PowerMode CPU_POWER_MODE` 设置为 ` paddle::lite_api::PowerMode::LITE_POWER_HIGH `
  - 分类模型的输入图像维度是{1, 3, 224, 224}，检测模型的维度是{1, 3, 300, 300}

- 测试结果

  |模型 |Kirin 810||Kirin 990||Kirin 990 5G||
  |---|---|---|---|---|---|---|
  |  |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |
  |mobilenet_v1_fp32_224|  38.358801|  5.903400|  30.234800|   3.352000|  31.567600|  2.992200|
  |resnet50_fp32_224|  224.719998|  18.087400|  176.660199|  9.825800|  186.572998|  7.645400|
  |ssd_mobilenet_v1_relu_voc_fp32_300|  80.059001|  30.157600|  63.044600|  22.901200|  68.458200|  21.399200|

### 已支持（或部分支持）NNAdapter 的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

### 测试设备(HUAWEI Mate30 5G)
![huwei_mate30_5g](https://paddlelite-demo.bj.bcebos.com/devices/huawei/kirin/huawei_mate30_5g.jpg)

### 准备设备环境

- 由于 HiAI DDK 可能依赖特定版本的ROM，建议用户更新至最新版 EMUI 系统，具体参考华为官方[手机升级指南](https://consumer.huawei.com/cn/support/update/)。

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
            - mobilenet_v1_fp32_224 # Paddle non-combined 格式的 mobilenet_v1 float32 模型
              - __model__ # Paddle fluid 模型组网文件，可使用 netron 查看网络结构
              - conv1_bn_mean # Paddle fluid 模型参数文件
              - subgraph_partition_config_file.txt # 自定义子图分割配置文件
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.android.arm64-v8a # arm64-v8a 编译工作目录
            - demo # 已编译好的，适用于 amd64-v8a 的示例程序
          - build.android.armeabi-v7a # armeabi-v7a 编译工作目录
            - demo # 已编译好的，适用于 arm64 的示例程序
            ...
          ...
          - demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run_with_adb.sh # 示例程序 adb 运行脚本
      - libs
        - PaddleLite
          - android
            - arm64-v8a
              - include # Paddle Lite 头文件
              - lib
                - huawei_kirin_npu # 华为麒麟 NPU HiAI DDK、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libhuawei_kirin_npu.so # NNAdapter device HAL 库
                  - libhiai.so # HiAI DDK
                  ...
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - armeabi-v7a
            	- include
              - lib
            ...
        - OpenCV # OpenCV 预编译库
      - object_detection_demo # 目标检测示例程序
  ```

- Android shell 端的示例程序
  - 按照以下命令分别运行转换后的 ARM CPU 模型和华为 Kirin NPU 模型，比较它们的性能和结果；
  ```shell
  1）由于 HiAI 的限制，需要 root 权限才能执行 shell 示例程序。
  2）`run_with_adb.sh` 只能在连接设备的系统上运行，不能在 Docker 环境执行（可能无法找到设备），也不能在设备上运行。
  3）`build.sh` 需要在 Docker 环境中执行，否则，需要将 `build.sh` 的 ANDROID_NDK 修改为当前环境下的 NDK 路径。
  4）`build.sh` 根据入参生成针对不同操作系统、体系结构的二进制程序，需查阅注释信息配置正确的参数值。
  5）`run_with_adb.sh` 入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。
  6）对于需要使能自定义子图分割文件的模型，请注意将 `run_with_adb.sh` 中 line12 行首 '#' 删除。

  运行适用于 ARM CPU 的 mobilenetv1 模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  $ ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a

    Top1 Egyptian cat - 0.482871
    Top2 tabby, tabby cat - 0.471594
    Top3 tiger cat - 0.039779
    Top4 lynx, catamount - 0.002430
    Top5 ping-pong ball - 0.000508
    Preprocess time: 4.716000 ms, avg 4.716000 ms, max 4.716000 ms, min 4.716000 ms
    Prediction time: 33.408000 ms, avg 33.408000 ms, max 33.408000 ms, min 33.408000 ms
    Postprocess time: 4.499000 ms, avg 4.499000 ms, max 4.499000 ms, min 4.499000 ms

  运行适用于华为 Kirin NPU 的 mobilenetv1 模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  $ ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a huawei_kirin_npu

    Top1 Egyptian cat - 0.479004
    Top2 tabby, tabby cat - 0.475342
    Top3 tiger cat - 0.039642
    Top4 lynx, catamount - 0.002363
    Top5 ping-pong ball - 0.000499
    Preprocess time: 5.132000 ms, avg 5.132000 ms, max 5.132000 ms, min 5.132000 ms
    Prediction time: 3.154000 ms, avg 3.154000 ms, max 3.154000 ms, min 3.154000 ms
    Postprocess time: 5.275000 ms, avg 5.275000 ms, max 5.275000 ms, min 5.275000 ms
  ```
- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/inputs` 目录下，同时将图片文件名添加到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/list.txt` 中；
- 重新编译示例程序：  
  ```shell
  注意：
  1）请根据 `buid.sh` 配置正确的参数值。
  2）需在 `Docker` 环境中编译。

  For arm64-v8a
  $ ./build.sh android arm64-v8a

  For armeabi-v7a
  $ ./build.sh android armeabi-v7a
  ```

- 注意：opt 生成的模型只是标记了华为 Kirin NPU 支持的 Paddle 算子，并没有真正生成华为 Kirin NPU 模型，只有在执行时才会将标记的 Paddle 算子转成 `HiAI IR` 并组网得到 `HiAI IRGraph`，然后生成并执行华为 Kirin NPU 模型（具体原理请参考 Pull Request[#2576](https://github.com/PaddlePaddle/Paddle-Lite/pull/2576)）；
- 不同模型，不同型号（ROM 版本）的华为手机，在执行阶段，由于某些 Paddle 算子无法完全转成 `HiAI IR`，或目标手机的 HiAI 版本过低等原因，可能导致 HiAI 模型无法成功生成，在这种情况下，Paddle Lite 会调用 ARM CPU 版算子进行运算完成整个预测任务。

### 更新支持华为 Kirin NPU 的 Paddle Lite 库

- 下载 Paddle Lite 源码和最新版 HiAI DDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/kirin/hiai_ddk_lib_510.tar.gz
  $ tar -xvf hiai_ddk_lib_510.tar.gz
  ```

- 编译并生成 `PaddleLite+NNAdapter+HuaweiKirinNPU` for armv8 and armv7 的部署库

  - For armv8
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_huawei_kirin_npu=ON --nnadapter_huawei_kirin_npu_sdk_root=$(pwd)/hiai_ddk_lib_510
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_huawei_kirin_npu=ON --nnadapter_huawei_kirin_npu_sdk_root=$(pwd)/hiai_ddk_lib_510 full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/

      替换 NNAdapter 运行时库
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/huawei_kirin_npu/

      替换 NNAdapter device HAL 库
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libhuawei_kirin_npu.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/huawei_kirin_npu/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      ```

  - For armv7
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_huawei_kirin_npu=ON --nnadapter_huawei_kirin_npu_sdk_root=$(pwd)/hiai_ddk_lib_510
      ```
    
    - full_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_huawei_kirin_npu=ON --nnadapter_huawei_kirin_npu_sdk_root=$(pwd)/hiai_ddk_lib_510 full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include/
      
      替换 NNAdapter 运行时库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/huawei_kirin_npu/
      
      替换 NNAdapter device HAL 库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libhuawei_kirin_npu.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/huawei_kirin_npu/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```
  
      备注：由于 HiAI DDK 的 so 库均基于 `c++_shared` 构建，建议将 android stl 设置为 `c++_shared`，更多选项还可以通过 `./lite/tools/build_android.sh help` 查看。

- 替换头文件后需要重新编译示例程序

## 其它说明

- 华为达芬奇架构的 NPU 内部大量采用 `float16` 进行运算，因此，预测结果会存在偏差，但大部分情况下精度不会有较大损失，可参考[ Paddle-Lite-Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo)中Image Classification Demo for Android 对同一张图片 CPU 与华为 Kirin NPU 的预测结果。
- 华为 Kirin 810/990 Soc 搭载的自研达芬奇架构的 NPU，与 Kirin 970/980 Soc 搭载的寒武纪 NPU 不一样，同样的，与 Hi3559A、Hi3519A 使用的 NNIE 也不一样，Paddle Lite 只支持华为自研达芬奇架构 NPU。
- 我们正在持续增加能够适配 HiAI IR 的 Paddle 算子 `bridge/converter`，以便适配更多 Paddle 模型，同时华为研发同学也在持续对 `HiAI IR` 性能进行优化。
