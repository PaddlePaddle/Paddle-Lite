# 华为昇腾 NPU 部署示例

Paddle Lite 已支持华为昇腾 NPU（Ascend310）在 X86 和 ARM 服务器上进行预测部署。 目前支持子图接入方式，其接入原理是在线分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 Ascend NPU 组网 API 进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- Ascend 310

### 已支持的设备

- Atlas 300I 推理卡（型号：3000/3010)
- Atlas 200 DK 开发者套件
- Atlas 800 推理服务器（型号：3000/3010）

### 已支持的 Paddle 模型

#### 模型
- 图像分类
  - [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz)
  - [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz)
  - [EfficientNetB0](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/EfficientNetB0.tgz)
  - [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz)
  - [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz)
  - [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz)
  - [MobileNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV1.tgz)
  - [MobileNet-v2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV2.tgz)
  - [MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_large_x1_0.tgz)
  - [MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_small_x1_0.tgz)
  - [ResNet-101](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz)
  - [ResNet-18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet18.tgz)
  - [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz)
  - [ResNeXt50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz)
  - [ShuffleNetV2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ShuffleNetV2_x1_0.tgz)
  - [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz)
  - [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz)
  - [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz)
- 目标检测
  - [SSD-MobileNetV1(1.8)](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)
  - [SSD-MobileNetV1(2.0+)](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_mobilenet_v1_300_120e_voc.tgz)
  - [SSD-VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_vgg16_300_240e_voc.tgz)
  - [YOLOv3-DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz)
  - [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz)
  - [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v3_large_270e_coco.tgz)
  - [YOLOv3-ResNet50_vd](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz)
  - [YOLOv4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov4_cspdarknet.tgz)
- 关键点检测
  - [HigherHRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/higherhrnet_hrnet_w32_640.tgz)
  - [HRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/hrnet_w32_384x288.tgz)
- 文本检测 & 文本识别
  - [OCR-DB](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_det_infer.tgz)
  - [OCR-REC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_rec_infer.tgz)
- 语义理解
  - [ERNIE-TINY](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_tiny.tgz)
- 生成网络
  - [ESRGAN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleGAN/esrgan_psnr_x4_div2k.tgz)


- [开源模型支持列表](../quick_start/support_model_list)

#### 性能


|模型    |Intel CPU性能 (ms)    |x86 + Ascend310 性能 (ms）    |鲲鹏 920 CPU 性能 (ms)    |鲲鹏920 + Ascend310 性能 (ms)    |
|---|---|---|---|---|
|AlexNet    |81.16     |4.32     |67.39     |3.64     |
|DenseNet121    |529.45     |7.81     |157.62     |6.23     |
|EfficientNet_b0    |167.59     |21.08     |85.98     |7.14     |
|ERNIE-TINY    |-    |12.31     |347.47     |10.72     |
|ESRGAN    |41774.81     |103.75     |9755.14     |103.06     |
|GoogLeNet    |104.55     |2.65     |71.24     |2.40     |
|HigherHRNet    |6760.83     |386.02     |6482.55     |396.17     |
|HRNet    |1245.54     |12.58     |752.93     |12.08     |
|Inception-v3    |415.00     |5.91     |159.40     |3.46     |
|Inception-v4    |1082.00     |9.00     |706.72     |6.72     |
|MobileNet-v1    |33.88     |3.06     |35.08     |1.43     |
|MobileNet-v2    |25.67     |3.82     |25.27     |1.74     |
|MobileNetV3_large_x1_0    |43.53     |8.96    |20.46     |2.83     |
|MobileNetV3_small_x1_0    |29.38     |6.38     |8.66     |2.24     |
|OCR-DB    |359.16     |10.75     |129.01     |10.78     |
|OCR-REC    |32.39     |8.75     |15.05     |5.87     |
|ResNet-101    |528.43     |7.33     |401.14     |4.27     |
|ResNet-18    |136.10     |2.84     |89.41     |1.59     |
|ResNet-50    |258.21     |5.66     |229.04     |2.87     |
|ResNeXt50    |353.43     |5.32     |261.28     |3.10     |
|ShuffleNetv2    |37.93     |5.41     |14.12     |4.06     |
|SqueezeNet-v1    |51.12     |3.06     |40.50     |1.49     |
|SSD-MobileNetV1(1.8)    |64.82     |6.56     |70.40     |6.19     |
|SSD-MobileNetV1(2.0+)    |70.72     |8.79     |73.16     |8.17     |
|SSD-VGG16    |1999.32     |27.35     |1001.06     |25.93     |
|VGG16    |1204.23     |10.53     |443.20     |9.43     |
|VGG19    |1512.82     |11.50     |667.20     |10.23     |
|YOLOv3-DarkNet53    |4241.35     |38.46     |2806.42     |34.47     |
|YOLOv3-MobileNetV1    |1181.56     |27.20     |941.62     |23.13     |
|YOLOv3-MobileNetV3    |1396.57     |27.97     |789.02     |27.06     |
|YOLOv3-ResNet50_vd    |15521.84     |55.41     |2444.08     |48.96     |
|YOLOv4    |7389.54     |77.80     |5503.77     |71.78     |

### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

测试设备（Atlas300I 推理卡）

<img src="https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/atlas300I.jpg" alt="Huawei_Ascend_NPU" style="zoom: 33%;" />

### 准备设备环境（如 ubuntu18.04-x86_64）

- Atlas 300I 推理卡[规格说明书](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-300-ai)

- 安装Atlas 300I 推理卡的驱动和固件包（Driver 和 Firmware)

- 驱动和固件包下载：https://www.hiascend.com/hardware/firmware-drivers?tag=commercial

  - 驱动：A300-3010-npu-driver_21.0.1_ubuntu18.04-x86_64.run（x86）

  - 固件：A300-3000-3010-npu-firmware_1.77.22.6.220.run

- 安装驱动和固件包：

```shell
# 增加可执行权限
$ chmod +x *.run
# 安装驱动和固件包
$ ./A300-3010-npu-driver_21.0.1_ubuntu18.04-x86_64.run --full
$ ./A300-3000-3010-npu-firmware_1.77.22.6.220.run --full
# 重启服务器
$ reboot
# 查看驱动信息，确认安装成功
$ npu-smi info
```

- 更多系统和详细信息见[昇腾硬件产品文档](https://www.hiascend.com/document?tag=hardware)

### 准备本地编译环境

- 为了保证编译环境一致，建议使用Docker开发环境进行配置；

- for arm64

  ```shell
  # 下载 Dockerfile
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/kunpeng920_arm/Ascend_ubuntu18.04_aarch64.Dockerfile
  # 通过 Dockerfile 生成镜像
  $ docker build --network=host -f Ascend_ubuntu18.04_aarch64.Dockerfile -t paddlelite/ascend_aarch64:cann_3.3.0 .
  # 创建容器
  $ docker run -itd --name=ascend-aarch64 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_aarch64:cann_3.3.0 /bin/bash
  # 进入容器
  $ docker exec -it ascend-aarch64 /bin/bash
  # 确认容器的 Ascend 环境是否创建成功
  $ npu-smi info
  ```

- for amd64

  ```shell
  # 下载 Dockerfile
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/intel_x86/Ascend_ubuntu18.04_x86.Dockerfile
  # 通过 Dockerfile 生成镜像
  $ docker build --network=host -f Ascend_ubuntu18.04_x86.Dockerfile -t paddlelite/ascend_x86:cann_3.3.0 .
  # 创建容器
  $ docker run -itd --name=ascend-x86 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_x86:cann_3.3.0 /bin/bash
  # 进入容器
  $ docker exec -it ascend-x86 /bin/bash
  # 确认容器的 Ascend 环境是否创建成功
  $ npu-smi info
  ```

  

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
                - huawei_ascend_npu # 华为昇腾 NPU CANN 库、NNAdapter 运行时库、device HAL 库
                	- libnnadapter.so # NNAdapter 运行时库
                	- libhuawei_ascend_npu.so # NNAdapter device HAL 库
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

- 执行以下命令比较 mobilenet_v1_fp32_224 模型的性能和结果；

  ```shell
  运行 mobilenet_v1_fp32_224 模型
  	
  For amd64
  (intel x86 cpu only)
  $ ./run.sh mobilenet_v1_fp32_224 linux amd64
      warmup: 1 repeat: 1, average: 44.949001 ms, max: 44.949001 ms, min: 44.949001 ms
      results: 3
      Top0  tabby, tabby cat - 0.529132
      Top1  Egyptian cat - 0.419680
      Top2  tiger cat - 0.045172
      Preprocess time: 1.017000 ms
      Prediction time: 44.949001 ms
      Postprocess time: 0.171000 ms
  (intel x86 cpu + ascend npu)
  $ ./run.sh mobilenet_v1_fp32_224 linux amd64 huawei_ascend_npu
      warmup: 1 repeat: 1, average: 2.079000 ms, max: 2.079000 ms, min: 2.079000 ms
      results: 3
      Top0  tabby, tabby cat - 0.529785
      Top1  Egyptian cat - 0.418945
      Top2  tiger cat - 0.045227
      Preprocess time: 1.132000 ms
      Prediction time: 2.079000 ms
      Postprocess time: 0.251000 ms
  
  For arm64
  (鲲鹏 920 cpu only)
  $ ./run.sh mobilenet_v1_fp32_224 linux arm64
      warmup: 1 repeat: 1, average: 34.160999 ms, max: 34.160999 ms, min: 34.160999 ms
      results: 3
      Top0  tabby, tabby cat - 0.529131
      Top1  Egyptian cat - 0.419681
      Top2  tiger cat - 0.045173
      Preprocess time: 0.571000 ms
      Prediction time: 34.160999 ms
      Postprocess time: 0.081000 ms
  (鲲鹏 920 cpu + ascend npu)
  $ ./run.sh mobilenet_v1_fp32_224 linux arm64 huawei_ascend_npu
      warmup: 1 repeat: 1, average: 1.555000 ms, max: 1.555000 ms, min: 1.555000 ms
      results: 3
      Top0  tabby, tabby cat - 0.529785
      Top1  Egyptian cat - 0.418945
      Top2  tiger cat - 0.045227
      Preprocess time: 0.605000 ms
      Prediction time: 1.555000 ms
      Postprocess time: 0.093000 ms
  ```

- 如果需要更改测试模型为 resnet50，可以将 `run.sh` 里的 MODEL_NAME 改成 resnet50_fp32_224，或执行命令：

  ```shell
  (intel x86 cpu + ascend npu)
  $ ./run.sh resnet50_fp32_224 linux amd64 huawei_ascend_npu
  (鲲鹏 920 cpu + ascend npu)
  $ ./run.sh resnet50_fp32_224 linux arm64 huawei_ascend_npu
  ```

- 如果需要更改测试图片，请将图片拷贝到 **`PaddleLite-generic-demo/image_classification_demo/assets/images`** 目录下，修改并执行 **`convert_to_raw_image.py`** 生成相应的 RGB Raw 图像，最后修改 `run.sh` 的 IMAGE_NAME 即可；

- 如果需要重新编译示例程序，直接运行

  ```shell
  # amd64
  $ ./build.sh linux amd64
  # arm64
  $ ./build.sh linux arm64
  ```

### 更新支持华为昇腾 NPU 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 PaddleLite+NNAdapter+HuaweiAscendNPU for amd64 and arm64 的部署库

  - For amd64
    - full_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=x86 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_huawei_ascend_npu=ON --nnadapter_huawei_ascend_npu_sdk_root=/usr/local/Ascend/ascend-toolkit/latest full_publish
      ```

    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/huawei_ascend_npu/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libhuawei_ascend_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/huawei_ascend_npu/
      # 替换 libpaddle_full_api_shared.so
      $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      ```

  - For arm64
    - full_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv8 --toolchain=gcc --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_huawei_ascend_npu=ON --nnadapter_huawei_ascend_npu_sdk_root=/usr/local/Ascend/ascend-toolkit/latest full_publish
      ```

    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/huawei_ascend_npu/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib/cxx/lib/libhuawei_ascend_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/huawei_ascend_npu/
      # 替换 libpaddle_full_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

- 替换头文件后需要重新编译示例程序

## 其他说明

- 华为达芬奇架构的 NPU 内部大量采用 `float16` 进行运算，因此，预测结果会存在偏差，但大部分情况下精度不会有较大损失。
- 我们正在持续增加能够适配 Ascend IR 的 Paddle 算子 `bridge/converter`，以便适配更多 Paddle 模型，同时华为研发同学也在持续对 Ascend IR 性能进行优化。
