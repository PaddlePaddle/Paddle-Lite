# 昆仑芯 XPU

Paddle Lite 已支持昆仑芯 XPU 在 X86 和 ARM 服务器（例如飞腾 FT-2000+/64）上进行预测部署, 支持 Kernel 接入方式。

## 支持现状

### 已支持的芯片

- 昆仑 818-100（推理芯片）
- 昆仑 818-300（训练芯片）

### 已支持的设备

- K100/K200 昆仑 AI 加速卡
- R200 昆仑芯 AI 加速卡

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
- 百度内部业务模型（由于涉密，不方便透露具体细节）

#### 性能

性能仅供参考,以实际运行效果为准。

| 模型                                                         | x86 + R200 性能 (ms） | arm64 + R200 性能 (ms） |
| ------------------------------------------------------------ | ------------------ | -------------------------- |
| [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz) | 0.5762              | 1.27252                     |
| [DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DarkNet/DarkNet53.tar.gz) | 1.04676              | 3.15686                     |
| [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz) | 3.4908             | 10.35096                      |
| [DPN68](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DPN/DPN68.tar.gz) | 2.5757             | 8.11432                      |
| [EfficientNetB0](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/EfficientNetB0.tgz) | 1.51762             | 4.92276                      |
| [GhostNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/GhostNet/GhostNet_x1_0.tar.gz) | 2.21846             | 7.01912                      |
| [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz) | 1.25748             | 4.20796                       |
| [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz) | 1.86128             | 5.69168                       |
| [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz) | 2.84598            | 8.1627                       |
| [MobileNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV1.tgz) | 0.48536              | 1.72218                       |
| [MobileNet-v2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV2.tgz) | 0.71952              | 2.5511                       |
| [Res2Net50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/Res2Net/Res2Net50_26w_4s.tar.gz) | 2.4858            | 7.46974          |
| [ResNet-101](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz) | 1.55288             |  3.48088                       |
| [ResNet-18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet18.tgz) | 0.41304             | 1.35338                       |
| [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz) | 0.90894            | 2.27986  |
| [ResNeXt50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz) | 1.0345             | 2.47522                       |
| [SE_ResNet50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/SENet/SE_ResNet50_vd.tar.gz) | 1.44298            | 5.42808         |
| [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz) | 0.5519              | 1.97804  |
| [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz) | 1.4011            | 1.94392                      |
| [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz) | 1.51684            | 2.02728                      |
| [ch_PP-OCRv2_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_det_infer.tar.gz) | 2.563            | 12.648        |
| [ch_PP-OCRv2_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz) | 2.851             | 7.069        |
| [ch_ppocr_server_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz) | 4.21  | 12.643 |
| [ch_ppocr_server_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz) | 23.843  | 25.78|
| [CRNN-mv3-CTC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz) | 1.606             | 5.363                      |
| [NCF](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/ncf.tar.gz) | 0.125              | 0.493                      |
| [PP-TSN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz) | 145.632            | 179.704            |
| [YOLOv3-DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz) |     7.1474        | 18.8239|
| [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz) | 5.0864        | 18.2639 |

### 已支持（或部分支持）的 Paddle 算子（ Kernel 接入方式）

- [算子支持列表](../quick_start/support_operation_list)

## 参考示例演示

### 测试设备( K100 昆仑 AI 加速卡)

![kunlunxin_xpu](https://paddlelite-demo.bj.bcebos.com/devices/baidu/baidu_xpu.jpg)

### 准备设备环境

- K100/200 昆仑 AI 加速卡[规格说明书](https://paddlelite-demo.bj.bcebos.com/devices/baidu/K100_K200_spec.pdf)，如需更详细的规格说明书或购买产品，请联系欧阳剑ouyangjian@baidu.com；
- K100 为全长半高 PCI-E 卡，K200 为全长全高 PCI-E 卡，要求使用 PCI-E x16 插槽，且需要单独的 8 针供电线进行供电；
- 安装 K100/K200 驱动，目前支持 Ubuntu 和 CentOS 系统，由于驱动依赖 Linux kernel 版本，请正确安装对应版本的驱动安装包。

### 准备编译环境

- 为了保证编译环境一致，建议根据机器的实际情况参考[ linux(x86) 编译](../source_compile/linux_x86_compile_linux_x86)或[ linux(ARM) 编译](../source_compile/arm_linux_compile_arm_linux)中的``准备编译环境``进行环境配置

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
                - kunlunxin_xtcl # 昆仑芯 XPU API 库、XPU runtime 库
                  - libxpuapi.so # XPU API 库，提供设备管理和算子实现。
                  - libxpurt.so # XPU runtime 库
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

- 执行以下命令观察 mobilenet_v1_int8_224_per_layer 模型的性能和结果；
  ```shell
  运行 mobilenet_v1_int8_224_per_layer 模型

  For amd64
  (intel x86 cpu only)
  $ ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux amd64

    Top1 Egyptian cat - 0.500662
    Top2 tabby, tabby cat - 0.407661
    Top3 tiger cat - 0.074697
    Top4 lynx, catamount - 0.013188
    Top5 ping-pong ball - 0.000638
    [0] Preprocess time: 10.196000 ms Prediction time: 54.002000 ms Postprocess time: 5.197000 ms
    Preprocess time: avg 10.196000 ms, max 10.196000 ms, min 10.196000 ms
    Prediction time: avg 54.002000 ms, max 54.002000 ms, min 54.002000 ms
    Postprocess time: avg 5.197000 ms, max 5.197000 ms, min 5.197000 ms

  (intel x86 cpu + kunlunxin xpu)
  $ ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux amd64 xpu

    Top1 Egyptian cat - 0.471169
    Top2 tabby, tabby cat - 0.445745
    Top3 tiger cat - 0.070651
    Top4 lynx, catamount - 0.008626
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000590
    [0] Preprocess time: 54.025000 ms Prediction time: 2.832000 ms Postprocess time: 32.408000 ms
    Preprocess time: avg 54.025000 ms, max 54.025000 ms, min 54.025000 ms
    Prediction time: avg 2.832000 ms, max 2.832000 ms, min 2.832000 ms
    Postprocess time: avg 32.408000 ms, max 32.408000 ms, min 32.408000 ms

  For arm64
  (arm cpu only)
  $ ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64

    Top1 Egyptian cat - 0.503239
    Top2 tabby, tabby cat - 0.419854
    Top3 tiger cat - 0.065506
    Top4 lynx, catamount - 0.007992
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000494
    [0] Preprocess time: 10.734000 ms Prediction time: 65.614000 ms Postprocess time: 8.718000 ms
    Preprocess time: avg 10.734000 ms, max 10.734000 ms, min 10.734000 ms
    Prediction time: avg 65.614000 ms, max 65.614000 ms, min 65.614000 ms
    Postprocess time: avg 8.718000 ms, max 8.718000 ms, min 8.718000 ms

  (arm cpu + kunlunxin xpu)
  $ ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 xpu

    Top1 Egyptian cat - 0.471169
    Top2 tabby, tabby cat - 0.445745
    Top3 tiger cat - 0.070651
    Top4 lynx, catamount - 0.008626
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000590
    [0] Preprocess time: 9.742000 ms Prediction time: 4.063000 ms Postprocess time: 8.097000 ms
    Preprocess time: avg 9.742000 ms, max 9.742000 ms, min 9.742000 ms
    Prediction time: avg 4.063000 ms, max 4.063000 ms, min 4.063000 ms
    Postprocess time: avg 8.097000 ms, max 8.097000 ms, min 8.097000 ms
  ```

- 如果需要更改测试模型为 resnet50，执行命令修改为如下：

  ```shell
  For amd64
  (intel x86 cpu only)
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64
  (intel x86 cpu + kunlunxin xpu)
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 xpu

  For arm64
  (arm cpu only)
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux arm64
  (arm cpu + kunlunxin xpu)
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 xpu
  ```

- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/inputs` 目录下，同时将图片文件名添加到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/list.txt` 中；

- 如果需要重新编译示例程序，直接运行

  ```shell
  For amd64
  $ ./build.sh linux amd64
  
  For arm64
  $ ./build.sh linux arm64
  ```

### 更新支持昆仑芯 XPU 的 Paddle Lite 库

- 下载 Paddle Lite 源码
  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 PaddleLite + NNAdapter + kunlunxin_xpu for amd64 and arm64 的部署库
	- For amd64
    如果报找不到 cxx11:: 符号的编译错误，请将 gcc 切换到 4.8 版本。
    - tiny_publish 编译
    ```shell
    $ ./lite/tools/build_linux.sh --arch=x86 --with_kunlunxin_xpu=ON
    ```
    
	  - full_publish 编译
    ```shell
    $ ./lite/tools/build_linux.sh --arch=x86 --with_kunlunxin_xpu=ON full_publish
    ```

	  - 替换头文件和库
    ```shell
    清理原有 include 目录
    $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      
    替换 include 目录
    $ cp -rf build.lite.linux.x86.gcc.kunlunxin_xpu/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/

    替换 XPU API 库
    $ cp build.lite.linux.x86.gcc.kunlunxin_xpu/inference_lite_lib/cxx/lib/libxpuapi.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/kunlunxin_xtcl/

    替换 XPU runtime 库
    $ cp build.lite.linux.x86.gcc.kunlunxin_xpu/inference_lite_lib/cxx/lib/libxpurt.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/kunlunxin_xtcl/
      
    替换 libpaddle_light_api_shared.so
    $ cp build.lite.linux.x86.gcc.kunlunxin_xpu/inference_lite_lib/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/

    替换 libpaddle_full_api_shared.so(仅在 full_publish 编译方式下)
    $ cp build.lite.linux.x86.gcc.kunlunxin_xpu/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      ```

  - For arm64
    arm 环境下需要设置环境变量 CC 和 CXX，分别指定 C 编译器和 C++ 编译器的路径
	  - tiny_publish 编译
    ```shell
    $ export CC=<path_to_your_c_compiler>
    $ export CXX=<path_to_your_c++_compiler>
    $ ./lite/tools/build_linux.sh --arch=armv8 --with_kunlunxin_xpu=ON
    ```
    
	  - full_publish 编译
    ```shell
    $ export CC=<path_to_your_c_compiler>
    $ export CXX=<path_to_your_c++_compiler>
    $ ./lite/tools/build_linux.sh --arch=armv8 --with_kunlunxin_xpu=ON full_publish
    ```

	  - 替换头文件和库
    ```shell
    清理原有 include 目录
    $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
    
    替换 include 目录
    $ cp -rf build.lite.linux.armv8.gcc.kunlunxin_xpu/inference_lite_lib.armlinux.armv8.xpu/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
    
    替换 XPU API 库
    $ cp build.lite.linux.armv8.gcc.kunlunxin_xpu/inference_lite_lib.armlinux.armv8.xpu/cxx/lib/libxpuapi.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/kunlunxin_xtcl/

    替换 XPU runtime 库
    $ cp build.lite.linux.armv8.gcc.kunlunxin_xpu/inference_lite_lib.armlinux.armv8.xpu/cxx/lib/libxpurt.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/kunlunxin_xtcl/
    
    替换 libpaddle_light_api_shared.so
    $ cp build.lite.linux.armv8.gcc.kunlunxin_xpu/inference_lite_lib.armlinux.armv8.xpu/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/

    替换 libpaddle_full_api_shared.so(仅在 full_publish 编译方式下)
    $ cp build.lite.linux.armv8.gcc.kunlunxin_xpu/inference_lite_lib.armlinux.armv8.xpu/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
    ```

- 替换头文件后需要重新编译示例程序

## 高级特性

- windows 版本的编译适配

  - Paddle Lite 使用 XPU kernel 的方案

  ```shell
  $ cd Paddle-Lite
  $ lite\\tools\\build_windows.bat with_extra without_python use_vs2017 with_dynamic_crt  with_kunlunxin_xpu kunlunxin_xpu_sdk_root D:\\xpu_toolchain_windows\\output
  ```

  编译脚本 `build_windows.bat` 使用可参考[Windows 环境下编译适用于 Windows 的库](../source_compile/windows_compile_windows)进行环境配置和查找相应编译参数

## 其他说明

- 如需更进一步的了解相关产品的信息，请联系欧阳剑 ouyangjian@baidu.com；
- 昆仑芯的研发同学正在持续适配更多的 Paddle 算子，以便支持更多的 Paddle 模型。
