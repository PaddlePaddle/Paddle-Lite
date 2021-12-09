# 英特尔 FPGA 部署示例

Paddle Lite 已支持英特尔 FPGA 平台的预测部署，Paddle Lite 通过调用底层驱动实现对 FPGA 硬件的调度。

## Paddle Lite 实现英特尔 FPGA 简介

Paddle Lite 支持英特尔 FPGA 作为后端硬件进行模型推理，其主要特性如下：

- Paddle Lite 中英特尔 FPGA 的 kernel，`weights` 和 `bias` 仍为 `FP32`、`NCHW` 的格式，在提升计算速度的同时能做到用户对数据格式无感知
- 对于英特尔 FPGA 暂不支持的 kernel，均会切回 ARM 端运行，实现 ARM+FPGA 混合布署运行
- 目前英特尔 FPGA 成本功耗都较低，可作为边缘设备首选硬件

## 支持现状

### 已支持的芯片

- 英特尔 FPGA Cyclone V 系列芯片

### 已支持的设备

- 海运捷讯 C5MB（英特尔 FPGA Cyclone V）开发板
- 海运捷讯 C5CB（英特尔 FPGA Cyclone V）开发板
- 海运捷讯 C5TB（英特尔 FPGA Cyclone V）开发板

### 已支持的 Paddle 模型

- [SSD-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_fluid.tar.gz)

### 已支持（或部分支持）的 Paddle 算子

- conv2d
- depthwise_conv2d

## 准备工作

开发板 `C5MB` 可以通过串口线进行连接，也可以通过 `ssh` 进行连接，初次使用请参考[文档](https://paddlelite-demo.bj.bcebos.com/devices/intel/AIGO_C5MB_UG.pdf)
可以通过串口完成 C5MB 开发板的 IP 修改：
  ```
  $ vi /etc/network/interfaces # 设备网络配置文件，将对应的 address，netmask，和 gateway 设置为指定的地址即可。
  ```

## 参考示例演示

### 测试设备( C5MB 开发板)

![c5mb_front](https://paddlelite-demo.bj.bcebos.com/devices/intel/c5mb_front.jpg)

![c5mb_back](https://paddlelite-demo.bj.bcebos.com/devices/intel/c5mb_back.jpg)

### 准备设备环境

- 提前准备带有 intelfpgadrv.ko 的英特尔 FPGA 开发板（如 C5MB）；
- 确定能够通过 `SSH` 方式远程登录 C5MB 开发板；
- 由于 C5MB 的 ARM 能力较弱，示例程序和 Paddle Lite 库的编译均采用交叉编译方式。

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[编译环境准备](../source_compile/docker_env)中的 Docker 开发环境进行配置；
- 由于需要通过 `scp` 和 `ssh` 命令将交叉编译生成的 Paddle Lite 库和示例程序传输到设备上执行，因此，在进入 Docker 容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像检测示例程序

- 下载示例程序[ PaddleLite-linux-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/intel/PaddleLite-linux-demo.tar.gz)，解压后清单如下：

  ```shell
  - PaddleLite-linux-demo
    - ssd_detection
      - assets
        - images 
          - dog.jpg # 测试图片
          - dog.raw # 已处理成 raw 数据的测试图片
        - labels
          - pascalvoc_label_list # 检测 label 文件
        - models
          - ssd_mobilenet_v1_fp32_300_fluid # Paddle fluid non-combined 格式的SSD-MobileNetV1 float32 模型
          - ssd_mobilenet_v1_fp32_300_for_intel_fpga
            - model.nb # 已通过 opt 转好的、适合英特尔 FPGA 的 SSD-MobileNetV1 float32 模型
      - shell
        - CMakeLists.txt # 示例程序 CMake 脚本
        - build
          - ssd_detection # 已编译好的示例程序
        - ssd_detection.cc # 示例程序源码
        - convert_to_raw_image.py # 将测试图片保存为 raw 数据的 python 脚本
        - build.sh # 示例程序编译脚本
        - run.sh # 示例程序运行脚本
        - intelfpgadrv.ko # 英特尔 FPGA 内核驱动程序
    - libs
      - PaddleLite
        - armhf
          - include # Paddle Lite 头文件
          - lib
            - libvnna.so # 英特尔 FPGA 推理运行时库
            - libpaddle_light_api_shared.so # 用于最终移动端部署的预编译 Paddle Lite 库（ tiny publish 模式下编译生成的库）
            - libpaddle_full_api_shared.so # 用于直接加载 Paddle 模型进行测试和 Debug 的预编译Paddle Lite 库（ full publish 模式下编译生成的库）
  ```

- 按照以下命令运行转换后的ARM+FPGA模型

  ```shell
  注意：
  1）`run.sh` 必须在 Host 机器上运行，且执行前需要配置目标设备的 IP 地址、SSH 账号和密码；
  2）`build.sh` 建议在 Docker 环境中执行，目前英特尔 FPGA 在 Paddle Lite 上只支持 armhf。

  运行适用于英特尔 FPGA 的 ssd_mobilenet_v1 量化模型
  $ cd PaddleLite-linux-demo/ssd_detection/assets/models
  $ cp ssd_mobilenet_v1_fp32_300_for_intel_fpga/model.nb ssd_mobilenet_v1_fp32_300_fluid.nb
  $ cd ../../shell
  $ vim ./run.sh
    MODEL_NAME 设置为 ssd_mobilenet_v1_fp32_300_fluid
  $ ./run.sh
    iter 0 cost: 3079.443115 ms
    iter 1 cost: 3072.508057 ms
    iter 2 cost: 3063.342041 ms
    warmup: 1 repeat: 3, average: 3071.764404 ms, max: 3079.443115 ms, min: 3063.342041 ms
    results: 3
    [0] bicycle - 0.997817 0.163673,0.217786,0.721802,0.786120
    [1] car - 0.943994 0.597238,0.131665,0.905698,0.297017
    [2] dog - 0.959329 0.157911,0.334807,0.431497,0.920035
    Preprocess time: 114.061000 ms
    Prediction time: 3071.764404 ms
    Postprocess time: 13.166000 ms
  ```

- 如果需要更改测试图片，可通过 `convert_to_raw_image.py` 工具生成；
- 如果需要重新编译示例程序，直接运行 `./build.sh` 即可，注意：`build.sh` 的执行建议在 Docker 环境中，否则可能编译出错。

### 更新模型

- 通过 Paddle 训练，或 X2Paddle 转换得到 SSD-MobileNetV1 float32 模型[ ssd_mobilenet_v1_fp32_300_fluid ](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_fluid.tar.gz)；
- 参考[模型转化方法](../user_guides/model_optimize_tool)，利用 opt 工具转换生成英特尔 FPGA 模型，仅需要将 `valid_targets` 设置为 intel_fpga, arm 即可。
  ```shell
  $ ./opt --model_dir=ssd_mobilenet_v1_fp32_300_fluid \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=intel_fpga,arm
  
  替换自带的英特尔 FPGA 模型
  $ cp opt_model.nb ssd_mobilenet_v1_fp32_300_for_intel_fpga/model.nb
  ```

- 注意：opt 生成的模型只是标记了英特尔 FPGA 支持的 Paddle 算子，并没有真正生成英特尔 FPGA 模型，只有在执行时才会将标记的 Paddle 算子转成英特尔 FPGA 的 APIs，最终生成并执行模型。

### 更新支持英特尔 FPGA 的 Paddle Lite 库

- 下载 Paddle Lite 源码和英特尔 FPGA 的 SDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  $ curl -L https://paddlelite-demo.bj.bcebos.com/devices/intel/intel_fpga_sdk_1.0.0.tar.gz -o - | tar -zx
  ```

- 编译并生成 Paddle Lite + Intel FPGA 的部署库

  For C5MB
  - tiny_publish 编译方式
  ```shell
  $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_log=ON --with_intel_fpga=ON --intel_fpga_sdk_root=./intel_fpga_sdk
  
  将 tiny_publish 模式下编译生成的 `build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.intel_fpga/cxx/lib/libpaddle_light_api_shared.so` 替换 `PaddleLite-linux-demo/libs/PaddleLite/armhf/lib/libpaddle_light_api_shared.so` 文件；
	```
  - full_publish 编译方式
  ```shell
  $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_log=ON --with_intel_fpga=ON --intel_fpga_sdk_root=./intel_fpga_sdk full_publish
  
  将 full_publis h模式下编译生成的 `build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.intel_fpga/cxx/lib/libpaddle_full_api_shared.so` 替换 `PaddleLite-linux-demo/libs/PaddleLite/armhf/lib/libpaddle_full_api_shared.so` 文件。
  ```

  - 将编译生成的 `build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.intel_fpga/cxx/include`替换 `PaddleLite-linux-demo/libs/PaddleLite/armhf/include` 目录；

## 其它说明
