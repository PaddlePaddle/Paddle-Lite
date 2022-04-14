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
    - [ResNet50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz)

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

| Model | Input| Batch | Dataset | GPU FP16 Latency(ms) | DLA FP16 Latency(ms)  |
|---|---|---|---|---|---|
|ResNet50| 1,3,224,224 | 1 | ImageNet 2012 | 3.574 | 6.9214 |


### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 准备设备环境

- 如需安装 TensorRT 环境, 请参考 [NVIDIA TENSORRT DOCUMENTATION](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html)

## 运行图像分类示例程序

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
  ```

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；
    - 须知：
        - Demo 中的 libs/PaddleLite/linux/arm64/lib/nvidia_tensorrt/libnvidia_tensorrt.so 是基于 Jetpack 4.4 + CUDA 10.2 + cuDNN 8.0 + TensorRT 7.1.3.0 在 Jetson AGX Xavier 环境设备上编译的。
        - Demo 中的 libs/PaddleLite/linux/amd64/lib/nvidia_tensorrt/libnvidia_tensorrt.so 是基于 CUDA 10.2 + cuDNN 8.0 + TensorRT 7.1.3.4 在 Quadro RTX 4000 环境设备上编译的。
    
       **如果用户环境不同，请参考『更新支持英伟达 TensorRT 的 Paddle Lite 库』章节进行编译和替换。**

- 执行以下命令比较 ResNet50 模型的性能和结果；

  ```shell
  运行 ResNet50 模型
  	
  # For Jetson AGX Xavier arm64
  (Arm cpu only)
  $ ./build.sh linux arm64
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
  $ ./run.sh resnet50_fp32_224 linux arm64 nvidia_tensorrt # 默认 fp32 的精度进行推理
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
  $ ./build.sh linux amd64
  $ ./run.sh resnet50_fp32_224 linux amd64 cpu
    warmup: 1 repeat: 5, average: 250.463202 ms, max: 278.670990 ms, min: 221.733002 ms
    results: 3
    Top0  tabby, tabby cat - 0.739791
    Top1  tiger cat - 0.130985
    Top2  Egyptian cat - 0.101033
    Preprocess time: 1.034000 ms
    Prediction time: 250.463202 ms
    Postprocess time: 0.142000 ms
     
  (Intel cpu + TensorRT) # CUDA 10.2 | cuDNN 8.0 | TensorRT 7.1.3.4
  # 注: 如果软件包版本和 Demo 中使用不一致需要重新编译 Paddle Lite 库, 请参考章节 "更新支持英伟达 TensorRT 的 Paddle Lite 库"
  $ ./run.sh resnet50_fp32_224 linux amd64 nvidia_tensorrt # 默认 fp32 的精度进行推理
      warmup: 1 repeat: 5, average: 4.760800 ms, max: 4.800000 ms, min: 4.717000 ms
      results: 3
      Top0  tabby, tabby cat - 0.739792
      Top1  tiger cat - 0.130985
      Top2  Egyptian cat - 0.101033
      Preprocess time: 1.022000 ms
      Prediction time: 4.760800 ms
      Postprocess time: 0.261000 ms
  ```

- 设备和精度选择设置:

    目前 Paddle Lite 支持 TensorRT 选择设备和不同精度进行推理。
    - 支持设备
        - GPU, DLA
    - 支持精度
        - float16, float32
    - 设置方法:
    ```
        $ export NVIDIA_TENSORRT_DEVICE_TYPE=GPU # 设置 device 类型
        $ export NVIDIA_TENSORRT_DEVICE_ID=0 # 设置 device id
        $ export NVIDIA_TENSORRT_PRECISION=float16 # 设置精度
        
        # 设置如上信息后，执行如下命令:
        $ ./run.sh resnet50_fp32_224 linux arm64 nvidia_tensorrt "NVIDIA_TENSORRT_DEVICE_TYPE=$NVIDIA_TENSORRT_DEVICE_TYPE;NVIDIA_TENSORRT_DEVICE_ID=$NVIDIA_TENSORRT_DEVICE_ID;NVIDIA_TENSORRT_PRECISION=$NVIDIA_TENSORRT_PRECISION;"
        
        # 执行结果:
            # For Jetson AGX Xavier arm64
            warmup: 1 repeat: 5, average: 3.530800 ms, max: 3.578000 ms, min: 3.390000 ms
            results: 3
            Top0  tabby, tabby cat - 0.740723
            Top1  tiger cat - 0.129761
            Top2  Egyptian cat - 0.101074
            Preprocess time: 0.704000 ms
            Prediction time: 3.530800 ms
            Postprocess time: 0.302000 ms

            # For RTX4000 amd64
            warmup: 1 repeat: 5, average: 1.952600 ms, max: 2.087000 ms, min: 1.858000 ms
            results: 3
            Top0  tabby, tabby cat - 0.741211
            Top1  tiger cat - 0.129883
            Top2  Egyptian cat - 0.100342
            Preprocess time: 0.979000 ms
            Prediction time: 1.952600 ms
            Postprocess time: 0.251000 ms

    ```
    
    上述命令表示：**使用 nvidia_tensorrt 在 GPU 的 第 0 个设备上以 float16 的精度进行推理。**

- 测试图片设置
    
    如需更改测试图片，请将图片拷贝到 **`PaddleLite-generic-demo/image_classification_demo/assets/images`** 目录下，修改并执行 **`convert_to_raw_image.py`** 生成相应的 RGB Raw 图像，最后修改 `run.sh` 的 IMAGE_NAME 即可


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
      $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_log=ON --with_exception=ON --with_nnadapter=ON --nnadapter_with_nvidia_tensorrt=ON -- nnadapter_nvidia_cuda_root=$NNADAPTER_NVIDIA_CUDA_ROOT --nnadapter_nvidia_tensorrt_root=$NNADAPTER_NVIDIA_TENSORRT_ROOT full_publish
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
