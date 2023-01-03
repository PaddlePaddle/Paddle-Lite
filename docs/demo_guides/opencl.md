# OpenCL

Paddle Lite 利用跨平台计算框架 OpenCL 将计算映射到 GPU 上执行，以充分利用 GPU 硬件算力，提高推理性能。在执行时会优先在 GPU 上执行算子，如果算子没有 GPU 实现，则该算子会回退到 CPU 上执行。

## 支持现状

### 已支持的芯片

- 高通骁龙 Adreno 系列 GPU：Adreno 888+/888/875/865/855/845/835/625 等
- ARM Mali 系列 GPU(具体为支持 Midgard、Bifrost、Valhall 这三个 GPU 架构下的 GPU)：Mali G76 MP16 (Valhall 架构，华为 P40 Pro), Mali-G72 MP3 (Bifrost 架构，OPPO R15), Mali T860（Midgard 架构，RK3399）等
- PowerVR 系列 GPU：如 PowerVR Rogue GE8320，对应芯片联发科 MT8768N 等
- macOS 系统下：
  -  Intel 集成显卡
  -  Apple Silicon 芯片，如 M1, M1 Pro
- Windows 64 位系统下：
  - Intel 集成显卡
  - NVIDIA/AMD 独立显卡

### 已支持的设备

- 包括但不限于包含上述 GPU 的设备

### 已验证支持的 Paddle 模型

#### 模型

- 图像分类
  - [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz)
  - [DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DarkNet/DarkNet53.tar.gz)
  - [DeiT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DeiT/DeiT_base_patch16_224.tar.gz)
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
  - [ShuffleNetV2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ShuffleNetV2_x1_0.tgz)
  - [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz)
  - [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz)
  - [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz)
  - [ViT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/VisionTransformer/ViT_base_patch16_224.tar.gz)
- 目标检测
  - [Picodet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/picodet_m_416_coco.tar.gz)
  - [PP-YOLO_mbv3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_mbv3_large_coco.tar.gz)
  - [PP-YOLO_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_r50vd_dcn_1x_coco.tar.gz)
  - [PPYOLO_tiny](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_tiny_650e_coco.tar.gz)
  - [PP-YOLOv2_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolov2_r50vd_dcn_365e_coco.tar.gz)
  - [SSD-MobileNetV1(1.8)](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)
  - [SSD-MobileNetV1(2.0+)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_mobilenet_v1_300_120e_voc.tgz)
  - [SSDLite-MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_large.tar.gz)
  - [SSDLite-MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_small.tar.gz)
  - [SSD-VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_vgg16_300_240e_voc.tgz)
  - [YOLOv3-DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz)
  - [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz)
  - [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v3_large_270e_coco.tgz)
  - [YOLOv3-ResNet50_vd](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz)
  - [YOLOv4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov4_cspdarknet.tgz)
- 姿态检测
  - [PP-TinyPose](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/tinypose_128x96.tar.gz)
- 人脸检测
  - [BlazeFace](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/blazeface_1000e.tgz)
  - [FaceBoxes](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/faceboxes.tgz)
- 关键点检测
  - [HigherHRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/higherhrnet_hrnet_w32_640.tgz)
  - [HRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/hrnet_w32_384x288.tgz)
- 文本检测 & 文本识别 & 端到端检测识别
  - [ch_ppocr_mobile_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_det_infer.tgz)
  - [ch_ppocr_mobile_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_rec_infer.tgz)
  - [ch_ppocr_server_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz)
  - [ch_ppocr_server_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz)
  - [ch_PP-OCRv2_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_det_infer.tar.gz)
  - [ch_PP-OCRv2_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz)
  - [CRNN-mv3-CTC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz)
  - [e2e_server_pgnetA](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/e2e_server_pgnetA.tar.gz)
- 自然语言处理 & 语义理解
  - [BERT](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/bert_base_uncased.tgz)
  - [ERNIE](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_1.0.tgz)
  - [ERNIE-TINY](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_tiny.tgz)
  - [Transformer](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/transformer.tar.gz)
- 生成网络
  - [ESRGAN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleGAN/esrgan_psnr_x4_div2k.tgz)
- 推荐系统
  - [DeepFM](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/deepfm.tar.gz)
  - [NAML](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/naml.tar.gz)
  - [NCF](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/ncf.tar.gz)
  - [Wide&Deep](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/wide_deep.tar.gz)
- 图像分割
  - [BiseNetV2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/bisenet.tar.gz)
  - [DeepLabV3+(CityScapes)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.tar.gz)
  - [PP-HumanSeg-Lite](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/pphumanseg_lite_generic_192x192_with_softmax.tar.gz)
  - [PP-HumanSeg-Server(DeepLabV3+)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.tar.gz)
  - [SegFormer](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/segformer.tar.gz)
  - [STDCSeg](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/stdcseg.tar.gz)
  - [U-Net](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/unet_cityscapes_1024x512_160k.tgz)
- 视频分类
  - [PP-TSN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz)

- [开源模型支持列表](../quick_start/support_model_list)

## 参考示例演示

### 准备本地编译环境

- 在 Android 系统上运行
  Paddle Lite 同时支持在 Linux x86 环境和 macOS 环境下编译适用于 Android 的库。
  - 如果宿主机是 Linux x86 环境，请根据 [Linux x86 环境下编译适用于 Android 的库](../source_compile/linux_x86_compile_android) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。
  - 如果宿主机是 macOS 环境，请根据 [macOS 环境下编译适用于 Android 的库](../source_compile/macos_compile_android) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。
- 在 ARMLinux 系统上运行
  Paddle Lite 同时支持在 Linux x86 环境下和 ARMLinux 环境下编译适用于 ARMLinux 的库。
  - 如果宿主机是 Linux x86 环境，请根据 [Linux x86 环境下编译适用于 ARMLinux 的库](../source_compile/linux_x86_compile_arm_linux) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。
  - 如果宿主机是 ARMLinux 环境，请根据 [ARMLinux 环境下编译适用于 ARMLinux 的库](../source_compile/arm_linux_compile_arm_linux) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。
- 在 macOS 系统上运行
  Paddle Lite 支持在 macOS 环境下编译适用于 macOS 的库。
  - 宿主机必须是 macOS 环境，请根据 [macOS 环境下编译适用于 macOS 的库](../source_compile/macos_compile_macos) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。
- 在 Windows 64 位系统上运行
  Paddle Lite 支持在 Windows 环境下编译适用于 Windows 的库。
  - 宿主机必须是 Windows 环境，请根据 [Windows 环境下编译适用于 Windows 的库](../source_compile/windows_compile_windows) 中的说明，依次准备编译环境、了解基础编译参数、执行编译步骤。

### 运行图像分类示例程序

- 下载示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后清单如下：

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
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - opencl
                  - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                  - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - armeabi-v7a
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - opencl
                  - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                  - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
          - linux
            - amd64
              ...
            - arm64
              ...
            - armhf
              ...
        - OpenCV # OpenCV 预编译库
      - object_detection_demo # 目标检测示例程序
  ```

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；

- 执行以下命令观察 mobilenet_v1_fp32_224 模型的性能和结果；

  ```shell
  运行 mobilenet_v1_fp32_224 模型
    
  For android arm64-v8a
  $ ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a opencl <adb设备号>

    Top1 Egyptian cat - 0.481445
    Top2 tabby, tabby cat - 0.470215
    Top3 tiger cat - 0.042389
    Top4 lynx, catamount - 0.002506
    Top5 ping-pong ball - 0.000542
    [0] Preprocess time: 10.048000 ms Prediction time: 12.671000 ms Postprocess time: 12.871000 ms
    Preprocess time: avg 10.048000 ms, max 10.048000 ms, min 10.048000 ms
    Prediction time: avg 12.671000 ms, max 12.671000 ms, min 12.671000 ms
    Postprocess time: avg 12.871000 ms, max 12.871000 ms, min 12.871000 ms

  For android armeabi-v7a
  $ ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android armeabi-v7a opencl <adb设备号>

    Top1 Egyptian cat - 0.481445
    Top2 tabby, tabby cat - 0.470215
    Top3 tiger cat - 0.042389
    Top4 lynx, catamount - 0.002506
    Top5 ping-pong ball - 0.000542
    [0] Preprocess time: 10.223000 ms Prediction time: 12.882000 ms Postprocess time: 11.180000 ms
    Preprocess time: avg 10.223000 ms, max 10.223000 ms, min 10.223000 ms
    Prediction time: avg 12.882000 ms, max 12.882000 ms, min 12.882000 ms
    Postprocess time: avg 11.180000 ms, max 11.180000 ms, min 11.180000 ms

  For linux arm64
  本地执行
  $ ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 opencl
  通过 SSH 远程执行
  $ ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 opencl <IP地址> 22 <用户名> <密码>

  For linux armhf
  本地执行
  $ ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux armhf opencl
  通过 SSH 远程执行
  $ ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux armhf opencl <IP地址> 22 <用户名> <密码>
  ```

- 如果需要更改测试模型为 resnet50，执行命令修改为如下：

  ```shell
  For android arm64-v8a
  $ ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a opencl <adb设备号>

  For android armeabi-v7a
  $ ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android armeabi-v7a opencl <adb设备号>

  For linux arm64
  本地执行
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 opencl
  通过 SSH 远程执行
  $ ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 opencl <IP地址> 22 <用户名> <密码>

  For linux armhf
  本地执行
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux armhf opencl
  通过 SSH 远程执行
  $ ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux armhf opencl <IP地址> 22 <用户名> <密码>
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

### 更新支持 OpenCL 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 armv8 和 armv7 的部署库

  - For android arm64-v8a
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv8 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON --with_opencl=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv8 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON --with_opencl=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/opencl/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.opencl/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/opencl/
      ```

  - For android armeabi-v7a
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON --with_opencl=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON --with_opencl=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.opencl/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.opencl/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/opencl/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.opencl/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/opencl/
      ```

- 编译并生成 arm64 和 armhf 的部署库

  - For linux arm64
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_cv=ON --with_exception=ON --with_opencl=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_cv=ON --with_exception=ON --with_opencl=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/opencl/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc.opencl/inference_lite_lib.armlinux.armv8.opencl/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/opencl/
      ```

  - For linux armhf
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_cv=ON --with_exception=ON --with_opencl=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_cv=ON --with_exception=ON --with_opencl=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.linux.armv7hf.gcc.opencl/inference_lite_lib.armlinux.armv7hf.opencl/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv7hf.gcc.opencl/inference_lite_lib.armlinux.armv7hf.opencl/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/opencl/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv7hf.gcc.opencl/inference_lite_lib.armlinux.armv7hf.opencl/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/opencl/
      ```

- 替换头文件后需要重新编译示例程序

## 高级特性

- 性能分析和精度分析

  关于性能和精度分析，请详细查阅[性能测试](../performance/benchmark_tools)中的【逐层耗时和精度分析】章节。

  在编译预测库时，使能性能分析和精度分析功能的命令如下：
  Android 平台下：
  ```shell
  开启性能分析，会打印出每个 op 耗时信息和汇总信息
  $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_opencl=ON --with_extra=ON --with_profile=ON full_publish

  开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
  $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_opencl=ON --with_extra=ON --with_precision_profile=ON full_publish
  ```

  macOS x86 平台下：
  ```shell
  开启性能分析，会打印出每个 op 耗时信息和汇总信息
  $ ./lite/tools/build.sh --build_opencl=ON --build_extra=ON --with_profile=ON x86

  开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
  $ ./lite/tools/build.sh --build_opencl=ON --build_extra=ON --with_precision_profile=ON x86
  ```

  Windows x86 平台下：
  ```shell
  开启性能分析，会打印出每个 op 耗时信息和汇总信息
  $ .\lite\tools\build_windows.bat with_opencl with_extra with_profile

  开启精度分析，会打印出每个 op 输出数据的均值和标准差信息
  $ .\lite\tools\build_windows.bat with_opencl with_extra with_precision_profile
  ```

- 判断设备是否支持 OpenCL
  函数 `IsOpenCLBackendValid` 用来检查设备是否支持 OpenCL，该函数内部会依次进行 OpenCL 驱动库检查、库函数检查、精度检查，检查均通过后返回 `true`，否则返回 `false`.
  - 函数声明[ paddle_api.h ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/api/paddle_api.h)
  - 使用示例[ mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)

- 设置 OpenCL kernel 缓存文件的路径
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

- 设置 OpenCL Auto-tune 策略
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

- 设置运行时精度
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

- 设置 OpenCL 混合内存对象推理
  OpenCL 大部分算子支持 cl::Image2D 数据排布，少部分算子支持 cl::Buffer（正在持续扩充），出于以下背景原因考虑
  1. 不同的设备采用 cl::Image2D 和 cl::Buffer 性能优势不同。
  2. 设备本身对 cl::Image2D 的 CL_DEVICE_IMAGE2D_MAX_HEIGHT 和 CL_DEVICE_IMAGE2D_MAX_WIDTH 有限制，导致部分 op 尺寸过大时会报错：malloc image is out of max image size(w,h)。
  3. 部分 op 采用 cl::Buffer 内存对象会有很好的性能，比如 reshape，transpose，keep_dims 为 false 的 argmax，reduce 等。
  支持两种内存对象可配置，通过环境变量 `OPENCL_MEMORY_CONFIG_FILE` 设置『OpenCL 内存对象配置文件』，实现人为指定部分 op使用 cl::Buffer 实现；

- 设置 OpenCL 与 CPU 异构推理
  对于 cl::Image2D 和 cl::Buffer 均无法支持或者性能差的算子，可以人为指定部分 op 跑 CPU 的实现，可通过环境变量 `OPENCL_MEMORY_CONFIG_FILE` 设置『OpenCL 内存对象配置文件』实现。
  如下的例子使用 benchmark 工具，输入为 PaddlePaddle 的部署模型格式，网络模型为 ch_PP-OCRv3_rec_infer，其中 conv2d，depthwise_conv2d 和 pool2d 三个 op 指定为跑 CPU 实现，剩余 op 跑 OpenCL 后端默认实现(大部分为 cl::Image2D)。

  ```shell
  $ cd /data/local/tmp/opencl
  $ cat ./ch_PP-OCRv3_rec_infer_buffer.txt
  device:cpu
  conv2d:elementwise_mul_2:batch_norm_51.tmp_4
  depthwise_conv2d:batch_norm_51.tmp_4:batch_norm_52.tmp_4
  pool2d:batch_norm_52.tmp_4:pool2d_4.tmp_0
  $ export OPENCL_MEMORY_CONFIG_FILE=./ch_PP-OCRv3_rec_infer_buffer.txt
  $ ./benchmark_bin  --model_file=./ch_PP-OCRv3_rec_infer/inference.pdmodel \
    --param_file=./ch_PP-OCRv3_rec_infer/inference.pdiparams \
    --input_shape=1,3,48,320 --backend=opencl --repeats=20 --warmup=2
  ```

  如下的例子为基于 OpenCL 与 CPU 异构推理将 PaddlePaddle 的部署模型格式转化为 Paddle Lite 支持的模型格式，网络模型和 OpenCL 内存对象配置文件同上, 使用 opt 工具方法如下:

  ```shell
  $ export OPENCL_MEMORY_CONFIG_FILE=./ch_PP-OCRv3_rec_infer_buffer.txt
  $ ./opt --model_file=./ch_PP-OCRv3_rec_infer/inference.pdmodel --param_file=./ch_PP-OCRv3_rec_infer/inference.pdiparams --optimize_out=./ch_PP-OCRv3_rec_infer/opt.nb --valid_targets=opencl
  ```


## 其他说明

1. OpenCL 计算过程中大多以 `cl::Image2D` 的数据排布进行计算，不同 gpu 支持的最大 `cl::Image2D` 的宽度和高度有限制，模型输入的数据格式是 buffer 形式的 `NCHW` 数据排布方式。要计算你的模型是否超出最大支持（大部分手机支持的 `cl::Image2D` 最大宽度和高度均为 16384），可以通过公式 `image_h = tensor_n * tensor_h, image_w=tensor_w * (tensor_c + 3) / 4` 计算当前层 `NCHW` 排布的 Tensor 所需的 `cl::Image2D` 的宽度和高度。如果某一层的 Tensor 维度大于如上限制，则会在日志中输出超限提示。
2. 当前版本的 Paddle Lite OpenCL 后端不支持量化模型作为输入；支持 fp32 精度的模型作为输入，在运行时会根据运行时精度配置 API `config.set_opencl_precision()` 来设定运行时精度（fp32 或 fp16）。
3. 部署时需考虑不支持 OpenCL 的情况，可预先使用 API `bool ::IsOpenCLBackendValid()` 判断，对于不支持的情况加载 CPU 模型，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
4. 对性能不满足需求的场景，可以考虑使用调优 API `config.set_opencl_tune(CL_TUNE_NORMAL)`，首次会有一定的初始化耗时，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
5. 对精度要求较高的场景，可以考虑通过 API `config.set_opencl_precision(CL_PRECISION_FP32)` 强制使用 `FP32` 精度，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
6. 对首次加载耗时慢的问题，可以考虑使用 API `config.set_opencl_binary_path_name(bin_path, bin_name)`，提高首次推理时，详见[ ./lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc ](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)。
7. Paddle Lite OpenCL 后端代码尚未完全支持动态 shape，因此在运行动态 shape 的模型时可能会报错。
8. 使用 OpenCL 后端进行部署时，模型推理速度并不一定会比在 CPU 上执行快。GPU 适合运行较大计算强度的负载任务，如果模型本身的单位算子计算密度较低，则有可能出现 GPU 推理速度不及 CPU 的情况。在面向 GPU 设计模型结构时，需要尽量减少低计算密度算子的数量，比如 slice、concat 等，具体可参见[使用 GPU 获取最佳性能](../performance/gpu.md)中的【优化建议】章节。
