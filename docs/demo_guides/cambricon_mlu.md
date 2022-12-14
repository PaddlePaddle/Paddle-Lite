# 寒武纪 MLU

Paddle Lite 已支持寒武纪 MLU（MLU370-X4 MLU370-S4）在 X86 服务器上进行预测部署。 目前支持子图接入方式，其接入原理是在线分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 MagicMind 组网 API 进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的设备

- 370-X4 （CNToolkit Version ≥  3.0.2）
- 370-S4 （CNToolkit Version ≥  3.0.2)

### 已验证支持的版本

- CNToolkit 版本  ≥  3.0.2
- MagicMind 版本 ≥  0.13.0
- 固件与驱动版本  ≥  4.20.16
- 设备的版本配套关系见 [寒武纪开发者社区](https://developer.cambricon.com/index/document/index/classid/3.html)

### 已验证支持的 Paddle 模型

#### 模型

- 图像分类
  - [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz)
  - [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz)
  - [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz)
  - [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz)
  - [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz)
  - [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz)
  - [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz)
  - [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz)
  - [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz)
- 目标检测
  - [PP-YOLO_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_r50vd_dcn_1x_coco.tar.gz)
  - [YOLOv3-ResNet50_vd](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz)

- [开源模型支持列表](../quick_start/support_model_list)


#### 性能

性能仅供参考,以实际运行效果为准。

| 模型                                                         | MLU370-S4 性能 (ms) |
| ------------------------------------------------------------ | ------------------ |
| [AlexNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz) | 2.08              |
| [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz) | 5.02             |
| [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz) | 2.35            |
| [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz) | 4.45             |
| [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz) | 7.72            |
| [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz) | 5.35             |
| [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz) | 1.23              |
| [VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz) | 8.94            |
| [VGG19](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz) | 9.99            |
| [PP-YOLO_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_r50vd_dcn_1x_coco.tar.gz) | 28.72      |
| [YOLOv3-ResNet50_vd](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz) | 19.91           |


### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

测试设备（MLU370-S4 推理卡）

<img src="https://paddlelite-demo.bj.bcebos.com/devices/cambricon_mlu/cambricon_device.jpg" alt="Cambricon_MLU370-S4" style="zoom: 33%;" />

### 准备设备环境（如 ubuntu18.04-x86_64）

- MLU370S4/X4 推理卡[规格说明书](https://developer.cambricon.com/index/document/index/classid/3.html)
- 从寒武纪开发者社区申请驱动和相关组件（Driver CNToolkit 和 MagicMind)

- 安装MLU370 推理卡的驱动(Driver)

- 驱动包：

  - CentOS7：neuware-mlu370-driver-4.15.16-1.x86_64.rpm
  - Ubuntu1604/1804/2004：neuware-mlu370-driver-dkms_4.15.16_all.deb

- 安装驱动和固件包：

```shell
增加可执行权限
$ chmod +x *.run

安装驱动
For CentOS7
$ yum install -y epel-release && yum makecache && yum install -y dkms
$ rpm -i neuware-mlu370-driver-4.15.16-1.x86_64.rpm

For Ubuntu1604/1804/2004
$ apt-get install -y dkms
$ dpkg -i neuware-mlu370-driver-dkms_4.15.16_all.deb

重启服务器
$ reboot

查看驱动信息，确认安装成功
$ cnmon 
```

- 按照文档，安装CNToolkit、MagicMind套件
- 更多系统和详细信息见[寒武纪开发者社区](https://developer.cambricon.com/index/document/index/classid/3.html)

### 准备本地编译环境

- 为了保证编译环境一致，建议参考 [Docker 统一编译环境搭建](../source_compile/docker_env) 中的 Docker 开发环境进行配置；

- for amd64
  ```shell
  下载 Dockerfile
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/cambricon_mlu/MagicMind_ubuntu18.04_x86.Dockerfile
  
  获取 cntoolkit、magicmind和cnnl/cnnlextra 等寒武纪MLU370的SDK，放在当前路径下
  通过 Dockerfile 生成镜像
  $ docker build --network=host -f MagicMind_ubuntu18.04_x86.Dockerfile -t paddlelite/mlu370_x86_magicmind .
  
  创建容器
  $ docker run -itd --name=mlu370-x86 --net=host \
    -v $PWD:/home/share -w /home/share -it --network=host --privileged \
    --device /dev/cambricon_ipcm0:/dev/cambricon_ipcm0 \
    --device /dev/cambricon_dev0:/dev/cambricon_dev0 \
    --device /dev/cambricon_ctl \
    -v /dev/cambricon:/dev/cambricon \
    -v /usr/bin/cnmon:/usr/bin/cnmon \
    paddlelite/mlu370_x86_magicmind /bin/bash
  
  进入容器
  $ docker exec -it mlu370-x86 /bin/bash
  
  确认容器的 MLU370 环境是否创建成功
  $ cnmon info
  ```
  

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
                - cambricon_mlu # 寒武纪MLU neuware 库、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libcambricon_mlu.so # NNAdapter device HAL 库
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

- 执行以下命令比较 resnet50_fp32_224 模型的性能和结果；

  ```shell
  运行 resnet50_fp32_224 模型
    
  For amd64
  (intel x86 cpu only)
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64

    Top1 tabby, tabby cat - 0.705223
    Top2 tiger cat - 0.134570
    Top3 Egyptian cat - 0.121521
    Top4 lynx, catamount - 0.028652
    Top5 ping-pong ball - 0.001043
    Preprocess time: 3.711000 ms, avg 3.711000 ms, max 3.711000 ms, min 3.711000 ms
    Prediction time: 174.218000 ms, avg 174.218000 ms, max 174.218000 ms, min 174.218000 ms
    Postprocess time: 4.920000 ms, avg 4.920000 ms, max 4.920000 ms, min 4.920000 ms

  (intel x86 cpu + cambricon mlu)
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 cambricon_mlu

    Top1 tabby, tabby cat - 0.705225
    Top2 tiger cat - 0.134570
    Top3 Egyptian cat - 0.121520
    Top4 lynx, catamount - 0.028652
    Top5 ping-pong ball - 0.001043
    Preprocess time: 4.269000 ms, avg 4.269000 ms, max 4.269000 ms, min 4.269000 ms
    Prediction time: 6.133000 ms, avg 6.133000 ms, max 6.133000 ms, min 6.133000 ms
    Postprocess time: 5.005000 ms, avg 5.005000 ms, max 5.005000 ms, min 5.005000 ms
  
- 如果需要更改测试模型，可以将 `run.sh` 里的 MODEL_NAME 改成比如 mobilenet_v1_fp32_224，或执行命令：

  ```shell
  (intel x86 cpu + cambricon mlu)
  $ ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 cambricon_mlu
  ```

- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/inputs` 目录下，同时将图片文件名添加到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/list.txt` 中；
- 如果需要重新编译示例程序，直接运行

  ```shell
  For amd64
  $ ./build.sh linux amd64
  ```

### 更新支持寒武纪 MLU 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 PaddleLite+NNAdapter+CambriconMLU for amd64 的部署库

  - For amd64

    - full_publish 编译

      ```shell
      $ ./lite/tools/build_linux.sh --arch=x86 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_cambricon_mlu=ON --nnadapter_cambricon_mlu_sdk_root=/usr/local/neuware full_publish
      ```

    - 替换头文件和库

      ```shell
      清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/

      替换 include 目录
      $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      
      替换 NNAdapter 运行时库
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/cambricon_mlu/
      
      替换 NNAdapter device HAL 库
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libcambricon_mlu.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/cambricon_mlu/
      
      替换 libpaddle_full_api_shared.so
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      
      替换 libpaddle_light_api_shared.so
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      ```


- 替换头文件后需要重新编译示例程序

## 高级特性

- 混合精度

  支持量化模型的推理，要求模型必须是由 PaddleSlim 产出的量化模型，例如：[resnet50_int8_per_layer](https://paddlelite-demo.bj.bcebos.com/devices/generic/models/resnet50_int8_224_per_layer.tar.gz)模型。

  

  **使用方式：**

  ```c++
  // Run inference by using full api with CxxConfig
  paddle::lite_api::CxxConfig cxx_config;
  std::string nnadapter_mixed_precision_quantization_config_path{"nnadapter_mixed_precision_quantization_config_path.txt"};
  // nnadapter_mixed_precision
  cxx_config.set_nnadapter_mixed_precision_quantization_config_path(nnadapter_mixed_precision_quantization_config_path);
  ```

  **nnadapter_mixed_precision_quantization_config_path.txt：** 该文件表示在全量化模型里，MLU无法支持量化的算子。

  目前MLU上运行全量化模型时，除 conv2d 和 fc 算子可运行在 INT8 精度下外，其余算子均需运行在 FP16 或 FP32 精度上。

  ```shell
  softmax
  pool2d
  elementwise_add
  relu
  ```

- 高级参数

  - CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH：

    指定模型编译参数配置文件的路径

  配置文件示例：

  ```shell
  {
      "archs": ["mtp_372"],
      "graph_shape_mutable": false,
      "precision_config":{
        "precision_mode":"qint8_mixed_float16"
      },
      "debug_config":{
        "fusion_enable": true
      },
      "opt_config":{
        "conv_scale_fold": true,
        "clustering_launch_enable": false,
        "type64to32_conversion": true
      }
  }
  ```


## 其他说明

- MLU 使用 `int8/int16/float16` 进行运算时，预测结果会存在偏差，但大部分情况下精度不会有较大损失。
