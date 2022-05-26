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

### 已支持的 Paddle 模型

#### 模型

- 图像分类
  - [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/resnet50_fp32_224.tar.gz)

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
