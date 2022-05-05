# 英伟达 TensorRT 部署示例

Paddle Lite 已支持 NVIDIA TensorRT 预测部署。 其接入原理是在线分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 TensorRT 组网 API 进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的 GPU 类型

- Jetson 全系列
- Pascal/Volt/Turning 架构的 GPU, 即将支持 Ampere 架构 GPU

### 已支持的英伟达软件栈

- Jetson
  - Jetpack 4.3 以上
- Tesla
  - CUDA 10.2/CUDA 11.0/CUDA 11.1
- cuDNN
  - 8.0.x
- TensorRT 
  - 7.1.3.x


### 已支持模型

- 分类模型
    - [resnet50_fp32_224](http://paddlelite-demo.bj.bcebos.com/devices/generic/models/resnet50_fp32_224.tar.gz)

- 检测模型
    - [yolov3_darknet53_270e_coco_fp32_608](http://paddlelite-demo.bj.bcebos.com/devices/generic/models/yolov3_darknet53_270e_coco_fp32_608.tar.gz)

### 性能

- 测试环境
  - 设备环境
    - NVIDIA Jetson AGX Xavier [16GB]
      - Jetpack 4.4.1 [L4T 32.4.4]
      - NV Power Mode: MAXN - Type: 0
    - Board info:
      - Type: AGX Xavier [16GB]
      - CUDA GPU architecture (ARCH_BIN): 7.2
    - Libraries:
      - CUDA: 10.2.89
      - cuDNN: 8.0.0.180
      - TensorRT: 7.1.3.0
      - Visionworks: 1.6.0.501
      - OpenCV: 4.1.1 compiled CUDA: NO
      - VPI: 0.4.4
      - Vulkan: 1.2.70
  - 编译环境
    - 操作系统: Ubuntu 18.04.4 LTS aarch64
    - gcc: 7.5.0
    - cmake: 3.23.0-rc4

- 测试结果

| Model | Input| Batch | Dataset | GPU FP16 Latency(ms) | GPU INT8 Latency(ms) | DLA FP16 Latency(ms)  | DLA INT8 Latency(ms) |
|---|---|---|---|---|---|---|---|
|resnet50_fp32_224| 1,3,224,224 | 1 | ImageNet 2012 | 3.15104 | 2.26772 | 6.65585 | 3.95726 |
|yolov3_darknet53_270e_coco_fp32_608| 1,3,608,608 | 1 | COCO | 37.327 | 24.584800 | 54.19 | 31.959|


### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 准备设备环境

- 如需安装 TensorRT 环境, 请参考 [NVIDIA TENSORRT DOCUMENTATION](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html)

## 运行示例程序

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
                - nvidia_tensorrt # NNAdapter 运行时库、device HAL 库
                	- libnnadapter.so # NNAdapter 运行时库
                	- libnvdia_tensorrt.so # NNAdapter device HAL 库
                - libiomp5.so # Intel OpenMP 库
                - libmklml_intel.so # Intel MKL 库
                - libmklml_gnu.so # GNU MKL 库
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - arm64
              - include # Paddle Lite 头文件
              - lib  # Paddle Lite 库文件
               - nvidia_tensorrt # NNAdapter 运行时库、device HAL 库
                	- libnnadapter.so # NNAdapter 运行时库
                	- libnvdia_tensorrt.so # NNAdapter device HAL 库
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - armhf
            	...
        - OpenCV # OpenCV 预编译库
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
      - yolo_detection_demo # 基于 yolo 的目标检测示例程序
  ```

- PaddleLite + TensorRT 运行时 context 相关选项设置
  - 设备选择
    - NVIDIA_TENSORRT_DEVICE_TYPE # 设置 device 类型, 默认 GPU
      - NVIDIA_TENSORRT_DEVICE_TYPE=GPU # 选择 GPU 设备进行推理
      - NVIDIA_TENSORRT_DEVICE_TYPE=DLA # 选择 DLA 设备进行推理
  - 设备号选择
    - NVIDIA_TENSORRT_DEVICE_ID # 设置 device id, 默认 0
      - NVIDIA_TENSORRT_DEVICE_ID=0 # 选择 DEVICE_TYPE 第 0 个设备
  - 精度选择
    - NVIDIA_TENSORRT_PRECISION # 设置精度, 默认 float32
      - NVIDIA_TENSORRT_PRECISION=int8 # int8 精度进行推理
      - NVIDIA_TENSORRT_PRECISION=float16 # float16 精度进行推理
      - NVIDIA_TENSORRT_PRECISION=float32 # float32 精度进行推理
  - Int8 精度校准
    - NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH # 设置 calibration 后生成 calibration table 的路径
    - NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH # 设置 calibration 所需的数据集路径 
          
      该路径下文件已如下方式进行组织:
      ```shell
        Image1.raw # 第 1 张图片预处理后的 raw 数据
        Image2.raw # 第 2 张图片预处理后的 raw 数据
        Image3.raw # 第 3 张图片预处理后的 raw 数据
        ...
        ...
        ImageN.raw # 第 N 张图片预处理后的 raw 数据
        lists.txt # 校准所需 raw 数据列表
      ```
      
      其中 lists.txt 内容格式如下:
      ```shell
        Image1.raw # 第 1 个 batch 校准所需的 raw 数据文件名称
        Image2.raw # 第 2 个 batch 校准所需的 raw 数据文件名称
        Image3.raw # 第 3 个 batch 校准所需的 raw 数据文件名称
        ...
        ...
        ImageN.raw # 第 N 个 batch 校准所需的 raw 数据文件名称
      ```

- Demo 中编译库版本须知：
  - Demo 中的 libs/PaddleLite/linux/arm64/lib/nvidia_tensorrt/libnvidia_tensorrt.so 是基于 Jetpack 4.4 + CUDA 10.2 + cuDNN 8.0 + TensorRT 7.1.3.0 在 Jetson AGX Xavier 环境设备上编译的。
  
  - Demo 中的 libs/PaddleLite/linux/amd64/lib/nvidia_tensorrt/libnvidia_tensorrt.so 是基于 CUDA 10.2 + cuDNN 8.0 + TensorRT 7.1.3.4 在 Quadro RTX 4000 环境设备上编译的。

  **如果用户环境不同，请参考『更新支持英伟达 TensorRT 的 Paddle Lite 库』章节进行编译和替换。**

### 运行图像分类示例程序
- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`
- 执行以下命令比较 ResNet50 模型的性能和结果；
    - float32 精度推理（默认）
      ```shell
      # For Jetson AGX Xavier arm64
      (Arm cpu only)
      $ ./run.sh resnet50_fp32_224 linux arm64 cpu
        warmup: 1 repeat: 5, average: 289.136005 ms, max: 295.342010 ms, min: 285.328003 ms
        results: 3
        Top0  tabby, tabby cat - 0.739791
        Top1  tiger cat - 0.130986
        Top2  Egyptian cat - 0.101033
        Preprocess time: 0.706000 ms
        Prediction time: 289.136005 ms
        Postprocess time: 0.315000 ms

      (Arm cpu + TensorRT) # CUDA 10.2 | cuDNN 8.0 | TensorRT 7.1.3.0
      # 注: 如果软件包版本和 Demo 中使用不一致需要重新编译 Paddle Lite 库, 请参考章节 "更新支持英伟达 TensorRT 的 Paddle Lite 库"
      # 默认使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 float32 的精度进行推理
      $ ./run.sh resnet50_fp32_224 linux arm64 nvidia_tensorrt
        warmup: 1 repeat: 5, average: 9.197000 ms, max: 9.224000 ms, min: 9.147000 ms
        results: 3
        Top0  tabby, tabby cat - 0.739792
        Top1  tiger cat - 0.130985
        Top2  Egyptian cat - 0.101032
        Preprocess time: 0.698000 ms
        Prediction time: 9.197000 ms
        Postprocess time: 0.313000 ms
      
      # For RTX4000 amd64
      (Intel cpu only)
      $ ./run.sh resnet50_fp32_224 linux amd64 cpu
        warmup: 1 repeat: 5, average: 192.425604 ms, max: 215.518005 ms, min: 176.852005 ms
        results: 3
        Top0  tabby, tabby cat - 0.739791
        Top1  tiger cat - 0.130985
        Top2  Egyptian cat - 0.101033
        Preprocess time: 0.947000 ms
        Prediction time: 192.425604 ms
        Postprocess time: 0.245000 ms
        
      (Intel cpu + TensorRT) # CUDA 10.2 | cuDNN 8.0 | TensorRT 7.1.3.4
      # 注: 如果软件包版本和 Demo 中使用不一致需要重新编译 Paddle Lite 库, 请参考章节 "更新支持英伟达 TensorRT 的 Paddle Lite 库"
      # 默认使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 float32 的精度进行推理
      $ ./run.sh resnet50_fp32_224 linux amd64 nvidia_tensorrt
          warmup: 1 repeat: 5, average: 4.760800 ms, max: 4.800000 ms, min: 4.717000 ms
          results: 3
          Top0  tabby, tabby cat - 0.739792
          Top1  tiger cat - 0.130985
          Top2  Egyptian cat - 0.101033
          Preprocess time: 1.022000 ms
          Prediction time: 4.760800 ms
          Postprocess time: 0.261000 ms
      ```
          
    - float16 精度推理:
      ```shell
        # 使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 float16 的精度进行推理 
        # 执行结果:
        # For Jetson AGX Xavier arm64
        $ ./run.sh resnet50_fp32_224 linux arm64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=GPU;NVIDIA_TENSORRT_DEVICE_ID=0;NVIDIA_TENSORRT_PRECISION=float16;"
        warmup: 1 repeat: 5, average: 3.530800 ms, max: 3.578000 ms, min: 3.390000 ms
        results: 3
        Top0  tabby, tabby cat - 0.740723
        Top1  tiger cat - 0.129761
        Top2  Egyptian cat - 0.101074
        Preprocess time: 0.704000 ms
        Prediction time: 3.530800 ms
        Postprocess time: 0.302000 ms

        # For RTX4000 amd64
        $ ./run.sh resnet50_fp32_224 linux amd64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=GPU;NVIDIA_TENSORRT_DEVICE_ID=0;NVIDIA_TENSORRT_PRECISION=float16;"
        warmup: 1 repeat: 5, average: 1.952600 ms, max: 2.087000 ms, min: 1.858000 ms
        results: 3
        Top0  tabby, tabby cat - 0.741211
        Top1  tiger cat - 0.129883
        Top2  Egyptian cat - 0.100342
        Preprocess time: 0.979000 ms
        Prediction time: 1.952600 ms
        Postprocess time: 0.251000 ms

      ```
    - int8 精度推理：
        ```shell
        # 使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 int8 的精度进行推理
        # 下载 calibration 所需数据集
        $ curl https://paddlelite-demo.bj.bcebos.com/devices/nvidia_tensorrt/datasets/imagenet_raw_1000.tar.gz -o -| tar -xz -C ../assets/ # ImageNet 验证数据集前 1000 张图片经过预处理后的 raw 数据   
        # 执行结果:
        # For Jetson AGX Xavier arm64
        $ ./run.sh resnet50_fp32_224 linux arm64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=GPU;NVIDIA_TENSORRT_DEVICE_ID=0;NVIDIA_TENSORRT_PRECISION=int8;NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=../assets/imagenet_raw_1000;NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=../assets/models/resnet50_fp32_224/calibration_table;"
        warmup: 1 repeat: 5, average: 2.674600 ms, max: 2.846000 ms, min: 2.587000 ms
        results: 3
        Top0  tabby, tabby cat - 0.742700
        Top1  tiger cat - 0.150497
        Top2  Egyptian cat - 0.078266
        Preprocess time: 0.655000 ms
        Prediction time: 2.674600 ms
        Postprocess time: 0.304000 ms

        # For RTX4000 amd64
        $ ./run.sh resnet50_fp32_224 linux amd64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=GPU;NVIDIA_TENSORRT_DEVICE_ID=0;NVIDIA_TENSORRT_PRECISION=int8;NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=../assets/imagenet_raw_1000;NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=../assets/models/resnet50_fp32_224/calibration_table;"
        warmup: 1 repeat: 5, average: 1.646200 ms, max: 1.724000 ms, min: 1.582000 ms
        results: 3
        Top0  tabby, tabby cat - 0.735309
        Top1  tiger cat - 0.153960
        Top2  Egyptian cat - 0.080652
        Preprocess time: 0.943000 ms
        Prediction time: 1.646200 ms
        Postprocess time: 0.246000 ms

        ```
      
- 测试图片设置
    
    如需更改测试图片，请将图片拷贝到 **`PaddleLite-generic-demo/image_classification_demo/assets/images`** 目录下，修改并执行 **`convert_to_raw_image.py`** 生成相应的 RGB Raw 图像，最后修改 `run.sh` 的 IMAGE_NAME 即可

### 运行目标检测示例程序

- 进入 `PaddleLite-generic-demo/yolo_detection_demo/shell/`
- 执行以下命令比较 yolov3_darknet53_270e_coco_fp32_608 模型的性能和结果
  - float32 精度推理（默认）
    ```shell
      # For Jetson AGX Xavier arm64
      (Arm cpu only)
      $ ./run.sh yolov3_darknet53_270e_coco_fp32_608 linux arm64 cpu
        warmup: 1 repeat: 5, average: 3340.893359 ms, max: 3370.016113 ms, min: 3315.264893 ms
        results: 3
        [0] bicycle - 0.994217 96.105804,134.768875,452.231476,429.883850
        [1] truck - 0.822486 371.336975,82.459366,542.959106,179.119370
        [2] dog - 0.987700 101.067924,238.172989,249.511230,566.703308
        Preprocess time: 3.220000 ms
        Prediction time: 3340.893359 ms
        Postprocess time: 0.012000 ms

      (Arm cpu + TensorRT) # CUDA 10.2 | cuDNN 8.0 | TensorRT 7.1.3.0
      # 注: 如果软件包版本和 Demo 中使用不一致需要重新编译 Paddle Lite 库, 请参考章节 "更新支持英伟达 TensorRT 的 Paddle Lite 库"
      # 默认使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 float32 的精度进行推理
      $ ./run.sh yolov3_darknet53_270e_coco_fp32_608 linux arm64 nvidia_tensorrt
        warmup: 1 repeat: 5, average: 108.054800 ms, max: 109.084999 ms, min: 106.007004 ms
        results: 3
        [0] bicycle - 0.994217 96.105850,134.768967,452.231445,429.883728
        [1] truck - 0.822484 371.336914,82.459389,542.959045,179.119354
        [2] dog - 0.987700 101.067947,238.173141,249.511246,566.703125
        Preprocess time: 4.250000 ms
        Prediction time: 108.054800 ms
        Postprocess time: 0.009000 ms
      
      # For RTX4000 amd64
      (Intel cpu only)
      $ ./run.sh yolov3_darknet53_270e_coco_fp32_608 linux amd64 cpu
        warmup: 1 repeat: 5, average: 2063.369971 ms, max: 2313.992920 ms, min: 1978.823975 ms
        results: 3
        [0] bicycle - 0.994217 96.105835,134.768921,452.231445,429.883789
        [1] truck - 0.822484 371.336914,82.459381,542.959045,179.119354
        [2] dog - 0.987700 101.067955,238.173111,249.511230,566.703186
        Preprocess time: 8.212000 ms
        Prediction time: 2063.369971 ms
        Postprocess time: 0.014000 ms
        
      (Intel cpu + TensorRT) # CUDA 10.2 | cuDNN 8.0 | TensorRT 7.1.3.4
      # 注: 如果软件包版本和 Demo 中使用不一致需要重新编译 Paddle Lite 库, 请参考章节 "更新支持英伟达 TensorRT 的 Paddle Lite 库"
      # 默认使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 float32 的精度进行推理
      $ ./run.sh yolov3_darknet53_270e_coco_fp32_608 linux amd64 nvidia_tensorrt
        warmup: 1 repeat: 5, average: 20.190400 ms, max: 20.415001 ms, min: 20.035000 ms
        results: 3
        [0] bicycle - 0.994138 96.253372,134.615204,451.998718,429.995758
        [1] truck - 0.824422 371.321472,82.480606,542.967773,179.083664
        [2] dog - 0.987714 101.088051,238.092957,249.499786,566.778992
        Preprocess time: 7.380000 ms
        Prediction time: 20.190400 ms
        Postprocess time: 0.009000 ms
      ```
          
  - float16 精度推理:
    ```shell
      # 使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 float16 的精度进行推理 
      # 执行结果:
      # For Jetson AGX Xavier arm64
      $ ./run.sh yolov3_darknet53_270e_coco_fp32_608 linux arm64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=GPU;NVIDIA_TENSORRT_DEVICE_ID=0;NVIDIA_TENSORRT_PRECISION=float16;"
      warmup: 1 repeat: 5, average: 38.339001 ms, max: 38.834999 ms, min: 36.581001 ms
      results: 3
      [0] bicycle - 0.994222 95.958069,134.931198,452.224915,429.843384
      [1] truck - 0.819042 371.327850,82.497841,542.974121,179.136292
      [2] dog - 0.987907 101.132500,238.003235,249.381256,566.689270
      Preprocess time: 3.162000 ms
      Prediction time: 38.339001 ms
      Postprocess time: 0.011000 ms

      # For RTX4000 amd64
      $ ./run.sh yolov3_darknet53_270e_coco_fp32_608 linux amd64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=GPU;NVIDIA_TENSORRT_DEVICE_ID=0;NVIDIA_TENSORRT_PRECISION=float16;"
      warmup: 1 repeat: 5, average: 21.748800 ms, max: 22.000000 ms, min: 21.627001 ms
      results: 3
      [0] bicycle - 0.994138 96.253372,134.615204,451.998718,429.995758
      [1] truck - 0.824422 371.321472,82.480606,542.967773,179.083664
      [2] dog - 0.987714 101.088051,238.092957,249.499786,566.778992
      Preprocess time: 6.977000 ms
      Prediction time: 21.748800 ms
      Postprocess time: 0.010000 ms

    ```
  - int8 精度推理：
      ```shell
      # 使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 int8 的精度进行推理
      # 下载 calibration 所需数据集
      $ curl https://paddlelite-demo.bj.bcebos.com/devices/nvidia_tensorrt/datasets/coco_raw_1000.tar.gz -o -| tar -xz -C ../assets/ # coco 数据集前 1000 张图片经过预处理后的 raw 数据
      # 执行结果:
      # For Jetson AGX Xavier arm64
      $ ./run.sh yolov3_darknet53_270e_coco_fp32_608 linux arm64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=GPU;NVIDIA_TENSORRT_DEVICE_ID=0;NVIDIA_TENSORRT_PRECISION=int8;NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=../assets/coco_raw_1000;NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=../assets/models/yolov3_darknet53_270e_coco_fp32_608/calibration_table;"
      warmup: 1 repeat: 5, average: 24.788400 ms, max: 24.900999 ms, min: 24.667999 ms
      results: 3
      [0] bicycle - 0.980123 96.815704,132.817017,452.862457,429.673828
      [1] truck - 0.563585 372.201050,83.083687,546.585938,179.713409
      [2] dog - 0.961471 99.111099,239.197708,252.214218,566.569702
      Preprocess time: 1.998000 ms
      Prediction time: 24.788400 ms
      Postprocess time: 0.010000 ms

      # For RTX4000 amd64
      $ ./run.sh yolov3_darknet53_270e_coco_fp32_608 linux amd64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=GPU;NVIDIA_TENSORRT_DEVICE_ID=0;NVIDIA_TENSORRT_PRECISION=int8;NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=../assets/coco_raw_1000;NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=../assets/models/yolov3_darknet53_270e_coco_fp32_608/calibration_table;"
      warmup: 1 repeat: 5, average: 17.169000 ms, max: 17.409000 ms, min: 16.837000 ms
      results: 3
      [0] bicycle - 0.980123 96.815704,132.817017,452.862457,429.673828
      [1] truck - 0.563585 372.201050,83.083687,546.585938,179.713409
      [2] dog - 0.961471 99.111099,239.197708,252.214218,566.569702
      Preprocess time: 7.698000 ms
      Prediction time: 17.169000 ms
      Postprocess time: 0.007000 ms

      ```

- 测试图片设置
    
    如需更改测试图片，请将图片拷贝到 **`PaddleLite-generic-demo/yolo_detection_demo/assets/images`** 目录下，修改并执行 **`convert_to_raw_image.py`** 生成相应的 RGB Raw 图像，最后修改 `run.sh` 的 IMAGE_NAME 即可

## 更新支持英伟达 TensorRT 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 PaddleLite+NNAdapter+TensorRT for amd64 and arm64 的部署库

  - For amd64
    - full_publish 编译
      ```shell
      $ export NNADAPTER_NVIDIA_CUDA_ROOT="/usr/local/cuda" # 替换成自己环境的 cuda 路径
      $ export NNADAPTER_NVIDIA_TENSORRT_ROOT="/usr/local/tensorrt" # 替换成自己环境的 tensorrt 路径
      $ ./lite/tools/build_linux.sh --arch=x86 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_nvidia_tensorrt=ON --nnadapter_nvidia_cuda_root=$NNADAPTER_NVIDIA_CUDA_ROOT --nnadapter_nvidia_tensorrt_root=$NNADAPTER_NVIDIA_TENSORRT_ROOT full_publish
      ```

    - 替换头文件和库
      ```shell
      # 清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      # 替换 include 目录
      $ cp -rf build.lite.linux.x86.gcc/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/include/
      # 替换 NNAdapter 运行时库
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/nvidia_tensorrt/
      # 替换 NNAdapter device HAL 库
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libnvidia_tensorrt.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/nvidia_tensorrt/
      # 替换 libpaddle_full_api_shared.so
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      # 替换 libpaddle_light_api_shared.so
      $ cp build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/amd64/lib/
      ```

  - For Jetson
    - full_publish 编译
      ```shell
      $ export NNADAPTER_NVIDIA_CUDA_ROOT="/usr/local/cuda" # 替换成自己环境的 cuda 路径
      $ export NNADAPTER_NVIDIA_TENSORRT_ROOT="/usr/local/tensorrt" # 替换成自己环境的 tensorrt 路径
      $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_nvidia_tensorrt=ON --nnadapter_nvidia_cuda_root=$NNADAPTER_NVIDIA_CUDA_ROOT --nnadapter_nvidia_tensorrt_root=$NNADAPTER_NVIDIA_TENSORRT_ROOT full_publish
      ```

    - 替换头文件和库
      ```shell
      # 清理原有 include 目录
      $ rm -rf PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/nvidia_tensorrt/
      # 替换 NNAdapter device HAL 库
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnvidia_tensorrt.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/nvidia_tensorrt/
      # 替换 libpaddle_full_api_shared.so
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_light_api_shared.so
      $ cp build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

- 替换头文件后需要重新编译示例程序
