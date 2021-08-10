# PaddleLite使用华为昇腾NPU预测部署

Paddle Lite已支持华为昇腾NPU（Ascend310）在x86和Arm服务器上进行预测部署。 目前支持子图接入方式，其接入原理是在线分析Paddle模型，将Paddle算子先转为统一的NNAdapter标准算子，再通过Ascend NPU组网API进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- Ascend 310

### 已支持的设备

- Atlas 300I推理卡（型号：3000/3010)
- Atlas 200 DK开发者套件
- Atlas 800推理服务器（型号：3000/3010）

### 已支持的Paddle模型

#### 模型

- [MobileNetV1](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)
- [ResNet50](https://paddlelite-demo.bj.bcebos.com/models/resnet50_fp32_224_fluid.tar.gz)
- [SSD-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_fluid.tar.gz)

- [开源模型支持列表](https://paddle-lite.readthedocs.io/zh/latest/introduction/support_model_list.html)

#### 性能

| 模型                               | Intel CPU性能(ms) | x86+Ascend310性能(ms） | 鲲鹏920 CPU性能(ms) | 鲲鹏920+Ascend310(ms) |
| ---------------------------------- | ----------------- | ---------------------- | ------------------- | --------------------- |
| mobilenet_v1_fp32_224              | 44.949            | 2.079                  | 34.161              | 1.555                 |
| resnet50_fp32_224                  | 266.570           | 1.828                  | 200.603             | 1.668                 |
| ssd_mobilenet_v1_relu_voc_fp32_300 | 87.061            | 7.016                  | 69.263              | 5.644                 |

### 已支持（或部分支持）的Paddle算子

可以通过访问[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/bridges/paddle_use_bridges.h]获得最新的算子支持列表。

## 参考示例演示

测试设备（Atlas300I推理卡）

<img src="https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/atlas300I.jpg" alt="Huawei_Ascend_NPU" style="zoom: 33%;" />

### 准备设备环境（如ubuntu18.04-x86_64）

- Atlas 300I推理卡[规格说明书](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-300-ai)

- 安装Atlas 300I推理卡的驱动和固件包（Driver和Firmware)

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

- 更多系统和详细信息见昇腾硬件产品文档（https://www.hiascend.com/document?tag=hardware）

### 准备本地编译环境

- 为了保证编译环境一致，建议使用Docker开发环境进行配置；

- for arm64

  ```shell
  # 下载Dockerfile
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/kunpeng920_arm/Ascend_ubuntu18.04_aarch64.Dockerfile
  # 通过Dockerfile生成镜像
  $ docker build --network=host -f Ascend_ubuntu18.04_aarch64.Dockerfile -t paddlelite/ascend_aarch64:cann_3.3.0 .
  # 创建容器
  $ docker run -itd --name=ascend-aarch64 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_aarch64:cann_3.3.0 /bin/bash
  # 进入容器
  $ docker exec -it ascend-aarch64 /bin/bash
  # 确认容器的Ascend环境是否创建成功
  $ npu-smi info
  ```

- for amd64

  ```shell
  # 下载Dockerfile
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/intel_x86/Ascend_ubuntu18.04_x86.Dockerfile
  # 通过Dockerfile生成镜像
  $ docker build --network=host -f Ascend_ubuntu18.04_x86.Dockerfile -t paddlelite/ascend_x86:cann_3.3.0 .
  # 创建容器
  $ docker run -itd --name=ascend-x86 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_x86:cann_3.3.0 /bin/bash
  # 进入容器
  $ docker exec -it ascend-x86 /bin/bash
  # 确认容器的Ascend环境是否创建成功
  $ npu-smi info
  ```

  

### 运行图像分类示例程序

- 下载示例程序[PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后清单如下：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - images
            - tabby_cat.jpg # 测试图片
            - tabby_cat.raw # 经过convert_to_raw_image.py处理后的RGB Raw图像
          - labels
            - synset_words.txt # 1000分类label文件
          - models
            - resnet50_fp32_224 # Paddle non-combined格式的resnet50 float32模型
              - __model__ # Paddle fluid模型组网文件，可拖入https://lutzroeder.github.io/netron/进行可视化显示网络结构
              - bn2a_branch1_mean # Paddle fluid模型参数文件
              - bn2a_branch1_scale
              ...
        - shell
          - CMakeLists.txt # 示例程序CMake脚本
          - build.linux.amd64 # 已编译好的，适用于amd64
            - image_classification_demo # 已编译好的，适用于amd64的示例程序
          - build.linux.arm64 # 已编译好的，适用于arm64
            - image_classification_demo # 已编译好的，适用于arm64的示例程序
            ...
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序ssh运行脚本
          - run_with_adb.sh # 示例程序adb运行脚本
      - libs
        - PaddleLite
          - android
            - arm64-v8a
            - armeabi-v7a
          - linux
            - amd64
              - include # PaddleLite头文件
              - lib # PaddleLite库文件
                - huawei_ascend_npu # 华为昇腾NPU NNAdapter API运行时库和Driver HAL库
                	- libnnadapter.so # NNAdapter API运行时库
                	- libnnadapter_driver_huawei_ascend_npu.so # 华为昇腾NPU NNAdapter driver HAL库
                - libiomp5.so # Intel OpenMP库
                - libmklml_intel.so # Intel MKL库
                - libmklml_gnu.so # GNU MKL库
                - libpaddle_full_api_shared.so # 预编译PaddleLite full api库
            - arm64
              - include # PaddleLite头文件
              - lib
            - armhf
            	...
        - OpenCV # OpenCV预编译库
      - ssd_detection_demo # 基于ssd的目标检测示例程序
  ```

  

- 进入PaddleLite-generic-demo/image_classification_demo/shell/；

- 按照以下命令比较mobilenet_v1_fp32_224模型的性能和结果；

  ```shell
  运行mobilenet_v1_fp32_224模型
  $ vi run.sh
  	将MODEL_NAME设置为mobilenet_v1_fp32_224
  	
  For amd64
  (intel x86 cpu only)
  $ ./run.sh linux amd64
      warmup: 1 repeat: 1, average: 44.949001 ms, max: 44.949001 ms, min: 44.949001 ms
      results: 3
      Top0  tabby, tabby cat - 0.529132
      Top1  Egyptian cat - 0.419680
      Top2  tiger cat - 0.045172
      Preprocess time: 1.017000 ms
      Prediction time: 44.949001 ms
      Postprocess time: 0.171000 ms
  (intel x86 cpu + ascend npu)
  $ ./run.sh linux amd64 huawei_ascend_npu
      warmup: 1 repeat: 1, average: 2.079000 ms, max: 2.079000 ms, min: 2.079000 ms
      results: 3
      Top0  tabby, tabby cat - 0.529785
      Top1  Egyptian cat - 0.418945
      Top2  tiger cat - 0.045227
      Preprocess time: 1.132000 ms
      Prediction time: 2.079000 ms
      Postprocess time: 0.251000 ms
  
  For arm64
  (鲲鹏920 cpu only)
  $ ./run.sh linux arm64
      warmup: 1 repeat: 1, average: 34.160999 ms, max: 34.160999 ms, min: 34.160999 ms
      results: 3
      Top0  tabby, tabby cat - 0.529131
      Top1  Egyptian cat - 0.419681
      Top2  tiger cat - 0.045173
      Preprocess time: 0.571000 ms
      Prediction time: 34.160999 ms
      Postprocess time: 0.081000 ms
  (鲲鹏920 cpu + ascend npu)
  $ ./run.sh linux arm64 huawei_ascend_npu
      warmup: 1 repeat: 1, average: 1.555000 ms, max: 1.555000 ms, min: 1.555000 ms
      results: 3
      Top0  tabby, tabby cat - 0.529785
      Top1  Egyptian cat - 0.418945
      Top2  tiger cat - 0.045227
      Preprocess time: 0.605000 ms
      Prediction time: 1.555000 ms
      Postprocess time: 0.093000 ms
  ```

- 如果需要更改测试模型为resnet50，可以改成**MODEL_NAME=resnet50_fp32_224**；

- 如果需要更改测试图片，请将图片拷贝到**PaddleLite-linux-demo/image_classification_demo/assets/images**目录下，修改并执行**convert_to_raw_image.py**生成相应的RGB Raw图像，最后修改run.sh的IMAGE_NAME即可；

- 如果需要重新编译示例程序，直接运行

  ```shell
  # amd64
  $ ./build.sh linux amd64
  # arm64
  $ ./build.sh linux arm64
  ```

### 更新支持华为昇腾NPU的Paddle Lite库

- 下载PaddleLite源码：

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译full_publish for amd64 or arm64；

  ```shell
  # amd64
  $ ./lite/tools/build_linux.sh --arch=x86 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_huawei_ascend_npu=ON --nnadapter_huawei_ascend_npu_sdk_root=/usr/local/Ascend/ascend-toolkit/latest full_publish
  
  # arm64
  $ ./lite/tools/build_linux.sh --arch=armv8 --toolchain=gcc --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_huawei_ascend_npu=ON --nnadapter_huawei_ascend_npu_sdk_root=/usr/local/Ascend/ascend-toolkit/latest full_publish
  ```

- 替换库文件和头文件（for amd64）

  ```shell
  # 替换 include 目录：
  $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/include/* PaddleLite-generic-demo/libs/PaddleLite/amd64/include/
  # 替换 NNAdapter相关so：
  $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libnnadapter* PaddleLite-generic-demo/libs/PaddleLite/amd64/lib/huawei_ascend_npu/
  # 替换 libpaddle_full_api_shared.so
  $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/amd64/lib/
  ```

- 替换库文件和头文件（for arm64）

  ```shell
  # 替换 include 目录：
  $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib/cxx/include/* PaddleLite-generic-demo/libs/PaddleLite/arm64/include/
  # 替换 NNAdapter相关so：
  $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib/cxx/lib/libnnadapter* PaddleLite-generic-demo/libs/PaddleLite/arm64/lib/huawei_ascend_npu/
  # 替换 libpaddle_full_api_shared.so
  $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/arm64/lib/
  ```

  备注：替换库文件和头文件后需要重新编译示例程序

## 其他说明

- 华为达芬奇架构的NPU内部大量采用float16进行运算，因此，预测结果会存在偏差，但大部分情况下精度不会有较大损失。
- 我们正在持续增加能够适配Ascend IR的Paddle算子bridge/converter，以便适配更多Paddle模型，同时华为研发同学也在持续对Ascend IR性能进行优化。
