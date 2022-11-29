# 昆仑芯 XTCL

Paddle Lite 已支持昆仑芯系列芯片及板卡 在 X86 和 ARM 服务器上进行预测部署。 目前支持子图接入方式，其接入原理是在线分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 XTCL 组网 API 进行网络构建，在线生成并执行模型。

XPU Tensor Compilation Library (XTCL)，即昆仑芯针对机器学习领域实践而提供的图编译引擎库，可提供基于昆仑芯硬件相关的图层分析框架和加速优化能力。

## 支持现状

### 已支持的芯片

- 昆仑芯1代 AI 芯片 CK10 / CK20
- 昆仑芯2代 AI 芯片 CR20

### 已支持的 AI 加速卡

- 昆仑芯 AI 加速卡 K100 / K200
- 昆仑芯 AI 加速卡 R200

### 已验证支持的 Paddle 模型

#### 模型

- 图像分类
  - [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz)
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
  - [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz)
  - [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz)
- 目标检测
  - [PP-YOLO_mbv3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_mbv3_large_coco.tar.gz)
  - [PPYOLO_tiny](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_tiny_650e_coco.tar.gz)
  - [SSD-MobileNetV1(1.8)](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)
  - [SSDLite-MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_large.tar.gz)
  - [SSDLite-MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_small.tar.gz)
  - [YOLOv3-DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz)
  - [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz)
  - [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v3_large_270e_coco.tgz)
  - [YOLOv4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov4_cspdarknet.tgz)
- 人脸检测
  - [FaceBoxes](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/faceboxes.tgz)
- 文本检测 & 文本识别 & 端到端检测识别
  - [ch_ppocr_server_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz)
  - [ch_ppocr_server_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz)
  - [ch_PP-OCRv2_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_det_infer.tar.gz)
  - [ch_PP-OCRv2_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz)
  - [CRNN-mv3-CTC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz)
- 推荐系统
  - [NCF](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/ncf.tar.gz)
- 图像分割
  - [PP-HumanSeg-Lite](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/pphumanseg_lite_generic_192x192_with_softmax.tar.gz)
  - [PP-HumanSeg-Server(DeepLabV3+)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.tar.gz)
- 视频分类
  - [PP-TSN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz)


- [开源模型支持列表](../quick_start/support_model_list.md)

**Note: 以上全部模型目前只在 R200 上测试验证通过，部分模型支持 K100 / K200.**



#### 性能

性能仅供参考,以实际运行效果为准。

| 模型                                                         | Intel CPU 性能 (ms) | x86 + R200 性能 (ms） |
| ------------------------------------------------------------ | ------------------ | -------------------------- |
| [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz) | 37.777400              | 0.689400                     |
| [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz) | 76.767599             | 4.015600                      |
| [EfficientNetB0](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/EfficientNetB0.tgz) | 96.174400             | 1.564200                      |
| [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz) | 46.062800             | 1.464400                       |
| [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz) | 77.464400             | 2.183400                       |
| [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz) | 151.721399            | 3.128800                       |
| [MobileNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV1.tgz) | 23.442400              | 0.500800                       |
| [MobileNet-v2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV2.tgz) | 19.889200              | 1.411200                       |
| [ResNet-18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet18.tgz) | 47.691199             | 0.429400                       |
| [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz) | 108.112999            | 0.815200  |
| [ResNet-101](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz) | 200.134998             |  1.392200                       |
| [ResNeXt50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz) | 110.684799             | 1.648600                       |
| [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz) | 33.140800              | 0.902400  |
| [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz) | 491.237994            | 1.542200                      |
| [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz) | 613.287396            | 1.675200                      |
| [DPN68](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DPN/DPN68.tar.gz) | 88.027000            | 2.805000                     |
| [DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DarkNet/DarkNet53.tar.gz) | 11197.110547            | 1.139400          |
| [GhostNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/GhostNet/GhostNet_x1_0.tar.gz) | 19.538400            | 5.726200            |
| [Res2Net50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/Res2Net/Res2Net50_26w_4s.tar.gz) | 371.824005            | 2.664400          |
| [SE_ResNet50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/SENet/SE_ResNet50_vd.tar.gz) | 140.231198            | 1.521600         |
| [ch_PP-OCRv2_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_det_infer.tar.gz) | 111.911003            | 3.690000        |
| [ch_PP-OCRv2_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz) | 40.988998             | 5.419000        |
| [ch_ppocr_server_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz) | 987.729980  | 6.303000 |
| [ch_ppocr_server_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz) | 370.881012  | 16.434999|
| [CRNN-mv3-CTC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz) | 13.041000             | 4.146000                      |
| [FaceBoxes](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/faceboxes.tgz) | 336.138000            | 43.075001             |
| [NCF](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/ncf.tar.gz) | 0.010000              | 0.147000                      |
| [PP-TSN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz) | 29140.355469            | 90.681000            |
| [SSD-MobileNetV1(1.8)](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz) | 33.758300 | 2.078400 |
| [YOLOv3-DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz) |     1828.807058        | 18.015895|
| [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz) | 541.669899        | 9.348368 |
| [YOLOv4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov4_cspdarknet.tgz) | 6887.933979            | 25.008631|

### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

### 测试设备(昆仑芯 AI 加速卡 R200)

<img src="https://baidu-kunlun-public.su.bcebos.com/paddle_lite/R200.jpg" alt="kunlunxin_xtcl" style="zoom: 100%;" />

### 准备设备环境

- 昆仑芯 AI 加速卡 R200 [产品手册](https://baidu-kunlun-public.su.bcebos.com/paddle_lite/R200%20%E4%BA%A7%E5%93%81%E6%89%8B%E5%86%8C%E5%A4%96%E9%83%A8%E7%89%88_0923.pdf)；
- R200 为全高全长 PCI-E 卡，要求使用 PCIe4.0 x16 插槽，且需要单独的 8 针供电线进行供电；
- 安装 [R200 XRE 驱动](https://baidu-kunlun-public.su.bcebos.com/paddle_lite/XRE%20%E5%AE%89%E8%A3%85%E6%89%8B%E5%86%8C_v1.0.pdf)，目前支持 Ubuntu 和 CentOS 系统，由于驱动依赖 Linux kernel 版本，请正确安装对应版本的驱动安装包。

### 准备本地编译环境
- 为了保证编译环境一致，建议根据机器的实际情况参考[ Linux x86 环境下编译适用于 Linux x86 的库](../source_compile/linux_x86_compile_linux_x86)或[ ARM Linux 环境下编译适用于 ARM Linux 的库](../source_compile/arm_linux_compile_arm_linux)中的``准备编译环境``进行环境配置

### 运行图像分类示例程序

- 下载示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz),解压后清单如下：

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
            - amd64
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - cpu
                  - libiomp5.so # Intel OpenMP 库
                  - libmklml_intel.so # Intel MKL 库
                  - libmklml_gnu.so # GNU MKL 库
                - kunlunxin_xtcl # 昆仑芯 XTCL 库、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libkunlunxin_xtcl.so # NNAdapter device HAL 库
                  - libxtcl.so # 昆仑芯 XTCL 库
                  ...
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - arm64
              - include
              - lib
            - armhf
              ...
        - OpenCV # OpenCV 预编译库
      - object_detection_demo # 目标检测示例程序
  ```

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；

- 执行以下命令比较 mobilenet_v1_fp32_224 模型的性能和结果；
  ```shell
  运行 mobilenet_v1_fp32_224 模型

  For amd64
  (intel x86 cpu only)
  $ ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64

    Top1 Egyptian cat - 0.482870
    Top2 tabby, tabby cat - 0.471594
    Top3 tiger cat - 0.039779
    Top4 lynx, catamount - 0.002430
    Top5 ping-pong ball - 0.000508
    Preprocess time: 3.133000 ms, avg 3.133000 ms, max 3.133000 ms, min 3.133000 ms
    Prediction time: 12.594000 ms, avg 12.594000 ms, max 12.594000 ms, min 12.594000 ms
    Postprocess time: 4.235000 ms, avg 4.235000 ms, max 4.235000 ms, min 4.235000 ms

  (intel x86 cpu + kunlunxin xtcl)
  $ ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 kunlunxin_xtcl

    Top1 Egyptian cat - 0.482607
    Top2 tabby, tabby cat - 0.471841
    Top3 tiger cat - 0.039819
    Top4 lynx, catamount - 0.002419
    Top5 ping-pong ball - 0.000505
    Preprocess time: 2.653000 ms, avg 2.653000 ms, max 2.653000 ms, min 2.653000 ms
    Prediction time: 0.524000 ms, avg 0.524000 ms, max 0.524000 ms, min 0.524000 ms
    Postprocess time: 4.077000 ms, avg 4.077000 ms, max 4.077000 ms, min 4.077000 ms

   For arm64
  (arm cpu only)
  $ ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64

    Top1 Egyptian cat - 0.482871
    Top2 tabby, tabby cat - 0.471594
    Top3 tiger cat - 0.039779
    Top4 lynx, catamount - 0.002430
    Top5 ping-pong ball - 0.000508
    Preprocess time: 8.241000 ms, avg 8.241000 ms, max 8.241000 ms, min 8.241000 ms
    Prediction time: 78.550000 ms, avg 78.550000 ms, max 78.550000 ms, min 78.550000 ms
    Postprocess time: 8.621000 ms, avg 8.621000 ms, max 8.621000 ms, min 8.621000 ms

  (arm cpu + kunlunxin xtcl)
  $ ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 kunlunxin_xtcl
  ```

- 如果需要更改测试模型为 resnet50，可以将 `run.sh` 里的 MODEL_NAME 改成 resnet50_fp32_224，或执行命令：

  ```shell
  (intel x86 cpu + kunlunxin xtcl)
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 kunlunxin_xtcl

  (arm cpu + kunlunxin xtcl)
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 kunlunxin_xtcl
  ```

- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/inputs` 目录下，同时将图片文件名添加到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/list.txt` 中；
- 如果需要重新编译示例程序，直接运行

  ```shell
  For amd64
  $ ./build.sh linux amd64
  
  For arm64
  $ ./build.sh linux arm64
  ```

### 更新支持昆仑芯 XTCL 的 Paddle Lite 库

- 下载 Paddle Lite 源码
  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 PaddleLite + NNAdapter + kunlunxin_xtcl for amd64 and arm64 的部署库
	- For amd64
	    - full_publish 编译
      ```shell
      默认自动从云上下载 kunlunxin_xtcl_sdk，如需指定，请使用参数--nnadapter_kunlunxin_xtcl_sdk_root
      $ ./lite/tools/build_linux.sh --arch=x86 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_kunlunxin_xtcl=ON full_publish
      ```

	    - 替换头文件和库
      ```shell
      清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      
      替换 include 目录
      $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      
      替换 NNAdapter 运行时库
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/kunlunxin_xtcl/
      
      替换 NNAdapter device HAL 库
      $ cp build.lite.linux.x86.gcc/lite/backends/nnadapter/nnadapter/src/driver/kunlunxin_xtcl/*.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/kunlunxin_xtcl/
      
      替换 libpaddle_full_api_shared.so
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      
      替换 libpaddle_light_api_shared.so
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      ```

  - For arm64
	  - full_publish 编译
    ```shell
    arm 环境下需要设置环境变量 CC 和 CXX，分别指定 C 编译器和 C++ 编译器的路径
    默认自动从云上下载 kunlunxin_xtcl_sdk，如需指定，请使用参数--nnadapter_kunlunxin_xtcl_sdk_root
    $ export CC=<path_to_your_c_compiler>
    $ export CXX=<path_to_your_c++_compiler>
    $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_kunlunxin_xtcl=ON full_publish
    ```

	  - 替换头文件和库
    ```shell
    清理原有 include 目录
    $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
    
    替换 include 目录
    $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
    
    替换 NNAdapter 运行时库
    $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/kunlunxin_xtcl/
    
    替换 NNAdapter device HAL 库
    $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libkunlunxin_xtcl.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/kunlunxin_xtcl/
    
    替换 libpaddle_full_api_shared.so
    $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
    
    替换 libpaddle_light_api_shared.so
    $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
    ```

## 高级特性
本节主要说明在不同的昆仑芯AI加速卡上如何设置不同的参数。以下列出了 Paddle Lite 下支持的两种高级参数。

- 高级参数

  - KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS

    指定昆仑芯产品的 ID 号。例如 KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS=0 或 KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS=0,1,2,3

  - KUNLUNXIN_XTCL_DEVICE_TARGET

    指定昆仑芯的不同类型的 AI 加速卡。例如 KUNLUNXIN_XTCL_DEVICE_TARGET=xpu -libs=xdnn -device-type=xpu1 或者 KUNLUNXIN_XTCL_DEVICE_TARGET=xpu -libs=xdnn -device-type=xpu2。XPU 代指昆仑芯自主研发的芯片硬件架构，XPU1 用在昆仑芯 1 代系列产品，包括 K100 和 K200；XPU2 用在昆仑芯 2 代系列产品，包括 R200 等。


- 使用方式
  -  c++ 代码示例
  ```c++
  // Run inference by using light api with MobileConfig
  paddle::lite_api::MobileConfig mobile_config;
  // nnadapter_context_properties, 多个参数之间使用;进行分割
  std::string nnadapter_context_properties = "KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS=0;KUNLUNXIN_XTCL_DEVICE_TARGET=xpu -libs=xdnn -device-type=xpu1"
  mobile_config.set_nnadapter_context_properties(nnadapter_context_properties);
  ```

  - shell 脚本示例
  ```shell
  export KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS=0
  export KUNLUNXIN_XTCL_DEVICE_TARGET="xpu -libs=xdnn -device-type=xpu1"
  ```

## 其他说明

- 昆仑芯的研发团队正在持续适配更多的 Paddle 算子，以便支持更多的 Paddle 模型。
