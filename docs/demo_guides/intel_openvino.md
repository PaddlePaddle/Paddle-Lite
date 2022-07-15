# 英特尔 OpenVINO 部署示例

Paddle Lite 已支持英特尔 OpenVINO 在 X86 服务器上进行预测部署。 目前支持子图接入方式，其接入原理是在线分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 OpenVINO 组网 API (API 2.0) 进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的设备

-  Intel® CPU 

### 已支持的操作系统平台

- Linux
  - Ubuntu 18.04 long-term support (LTS), 64-bit
  - Ubuntu 20.04 long-term support (LTS), 64-bit

### 已支持 OpenVINO 版本

- OpenVINO 2022.1

  注: OpenVINO 2022.1 对于操作系统以及硬件的相关约束可查看: [https://docs.openvino.ai/2022.1/openvino_docs_install_guides_installing_openvino_linux.html#system-requirement](https://docs.openvino.ai/2022.1/openvino_docs_install_guides_installing_openvino_linux.html#system-requirements)

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

### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

测试设备（Intel® CPU X86 服务器）

### 准备设备环境（如 ubuntu18.04-x86_64）

- OpenVINO Runtime 安装 
  
  - 安装链接：https://docs.openvino.ai/2022.1/openvino_docs_install_guides_install_runtime.html

  - 安装方式： 推荐使用 Installer 工具以『安静模式』安装。

    安装命令示例: 
    
     `l_openvino_toolkit_p_2022.1.0.643_offline.sh -a -s --eula accept`
  

  安装结束后请确认运行 OpenVINO Runtime 所需的环境变量已正确设置。

### 运行图像分类示例程序

- 下载示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后清单如下：

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
          - build.linux.amd64 # 已编译好的，适用于 amd64
            - image_classification_demo # 已编译好的，适用于 amd64 的示例程序
          - build.linux.arm64 # 已编译好的，适用于 arm64
            - image_classification_demo # 已编译好的，适用于 arm64 的示例程序
            ...
          ...
          - image_classification_demo.cc # 示例程序源码
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
                - intel_openvino # NNAdapter 运行时库、device HAL 库
                	- libnnadapter.so # NNAdapter 运行时库
                	- libintel_openvino.so # NNAdapter device HAL 库
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
        - OpenCV # OpenCV 预编译库
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
  ```

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；

- 执行以下命令比较 resnet50_fp32_224 模型的性能和结果；

  ```shell
  运行 resnet50_fp32_224 模型
  	
  For amd64
  (intel x86 cpu only)
  $ ./run.sh resnet50_fp32_224 linux amd64
      warmup: 1 repeat: 5, average: 195.586401 ms, max: 203.028000 ms, min: 189.692001 ms
      results: 3
      Top0  tabby, tabby cat - 0.739791
      Top1  tiger cat - 0.130985
      Top2  Egyptian cat - 0.101033
      Preprocess time: 1.504000 ms
      Prediction time: 195.586401 ms
      Postprocess time: 0.287000 ms
  (intel x86 cpu + OpenVINO)
  $ ./run.sh resnet50_fp32_224 linux amd64 intel_openvino
      warmup: 1 repeat: 5, average: 24.080800 ms, max: 31.004000 ms, min: 19.587999 ms
      results: 3
      Top0  tabby, tabby cat - 0.739792
      Top1  tiger cat - 0.130985
      Top2  Egyptian cat - 0.101032
      Preprocess time: 0.994000 ms
      Prediction time: 24.080800 ms
      Postprocess time: 0.146000 ms
  
  ```

- 如果需要更改测试图片，请将图片拷贝到 **`PaddleLite-generic-demo/image_classification_demo/assets/images`** 目录下，修改并执行 **`convert_to_raw_image.py`** 生成相应的 RGB Raw 图像，最后修改 `run.sh` 的 IMAGE_NAME 即可；

- 如果需要重新编译示例程序，直接运行

  ```shell
  $ ./build.sh linux amd64
  ```

  ### 更新支持英特尔 OpenVINO 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 PaddleLite + NNAdapter + OpenVINO Runtime 的部署库

  - full_publish 编译
    ```shell
    $ ./lite/tools/build_linux.sh --arch=x86 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_intel_openvino=ON --nnadapter_intel_openvino_sdk_root=/opt/intel/openvino_2022 full_publish
    ```

  - 替换头文件和库
    ```shell
    # 清理原有 include 目录
    $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
    # 替换 include 目录
    $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
    # 替换 NNAdapter 运行时库
    $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/intel_openvino/
    # 替换 NNAdapter device HAL 库
    $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libintel_openvino.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/intel_openvino/
    # 替换 libpaddle_full_api_shared.so
    $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
    # 替换 libpaddle_light_api_shared.so
    $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
    ```

- 替换头文件后需要重新编译示例程序
