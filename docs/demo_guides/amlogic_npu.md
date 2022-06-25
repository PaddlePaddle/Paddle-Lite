# 晶晨 NPU 部署示例

Paddle Lite 已支持晶晨 NPU 的预测部署。
其接入原理是与之前华为 Kirin NPU、瑞芯微 Rockchip NPU 等类似，即加载并分析 Paddle 模型，首先将 Paddle 算子转成 NNAdapter 标准算子，其次再转换为 Amlogic NPU 组网 API 进行网络构建，在线生成并执行模型。
- **请注意**：本文介绍的是 Paddle Lite 基于 AmlogicNPU DDK 来调用晶晨 SoC 的 NPU 算力，考虑到算子以及模型支持的广度，如果需要在晶晨 SoC 上部署较为复杂的模型，我们强烈建议您参考[芯原 TIM-VX 部署示例](./verisilicon_timvx)，同样能调用晶晨 SoC 的 NPU 算力，且支持场景更多。

## 支持现状

### 已支持的芯片

- C308X
- A311D
- S905D3(Android 版本)

### 已支持的 Paddle 模型

#### 模型

- [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)
- [resnet50_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/resnet50_int8_224_per_layer.tar.gz)
- [ssd_mobilenet_v1_relu_voc_int8_300_per_layer](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_int8_300_per_layer.tar.gz)

#### 性能

- 测试环境
  - 编译环境
    - Ubuntu 16.04，GCC 5.4 for ARMLinux armhf and aarch64

  - 硬件环境
    - C308X
      - CPU：2 x ARM Cortex-55
      - NPU：4 TOPs for INT8

    - A311D
      - CPU：4 x ARM Cortex-A73 \+  2 x ARM Cortex-A53
      - NPU：5 TOPs for INT8
    - S905D3(Android 版本)
      - CPU：2 x ARM Cortex-55
      - NPU：1.2 TOPs for INT8
  
- 测试方法
  - warmup=1, repeats=5，统计平均时间，单位是 ms
  - 线程数为1，`paddle::lite_api::PowerMode CPU_POWER_MODE`设置为` paddle::lite_api::PowerMode::LITE_POWER_HIGH `
  - 分类模型的输入图像维度是{1, 3, 224, 224}
  
- 测试结果

  |模型 |C308X||A311D||S905D3(Android 版本)||
  |---|---|---|---|---|---|---|
  |  |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |
  |mobilenet_v1_int8_224_per_layer| 167.6996 | 6.982800| 81.632133 | 5.607733 | 280.465997 | 13.411600 |
  |resnet50_int8_224_per_layer| 695.527405| 20.288600| 390.498300| 18.002560| 787.532340 | 42.858800|
  |ssd_mobilenet_v1_relu_voc_int8_300_per_layer| 281.442310| 18.015800| 134.991560| 15.978300| 295.48919| 41.035610|

### 已支持（或部分支持）NNAdapter 的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

### 测试设备

- C308X开发板

  <img src="https://paddlelite-demo.bj.bcebos.com/devices/amlogic/C308X.jpg" alt="C380X" style="zoom: 20%;" />

  

- A311D开发板

   <img src="https://paddlelite-demo.bj.bcebos.com/devices/amlogic/A311D.jpg" alt="A311D" style="zoom: 20%;" />

  

- S905D3开发板

   <img src="https://paddlelite-demo.bj.bcebos.com/devices/amlogic/S905D3.jpg" alt="A311D" style="zoom: 22%;" />

### 准备设备环境

- 确定开发板 NPU 驱动版本
  - 由于晶晨 SoC 使用芯原 NPU IP，因此，部署前要保证芯原 Linux Kernel NPU 驱动—— galcore.so 版本及所适用的芯片型号与依赖库保持一致。
  - 可通过命令行输入 `dmesg | grep Galcore` 查询 NPU 驱动版本。请注意，建议版本为 6.4.4.3。如果当前版本就是 6.4.4.3 ，可以跳过本环节。
  - 有两种方式可以修改当前的 NPU 驱动版本及其依赖库：
    - 方法一 ：刷机，根据具体的开发板型号，向开发板卖家或官网客服索要 6.4.4.3 版本 NPU 驱动对应的固件和刷机方法。
      - 在此额外提供 khadas 开发板 VIM3|VIM3L 的 6.4.4.3 固件以及官方教程链接：
        - 刷机镜像（包含 NPU 驱动文件和芯原相关依赖库，分别提供 khadas 官方服务器下载地址，和飞桨服务器的下载地址，均可下载使用）：
          - VIM3 Android：VIM3_Pie_V210908：[官方链接](https://dl.khadas.com/Firmware/VIM3/Android/VIM3_Pie_V210908.7z)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3/VIM3_Pie_V210908.7z)
          - VIM3 Linux：VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625：[官方链接](http://dl.khadas.com/firmware/VIM3/Ubuntu/EMMC/VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3/VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)
          - VIM3L Android：VIM3L_Pie_V210906：[官方链接](https://dl.khadas.com/Firmware/VIM3L/Android/VIM3L_Pie_V210906.7z)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3l/VIM3L_Pie_V210906.7z)
          - VIM3L Linux：VIM3L_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625：[官方链接](https://dl.khadas.com/Firmware/VIM3L/Ubuntu/EMMC/VIM3L_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3l/VIM3L_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)
        - 官方刷机教程：[VIM3/3L Android 文档](https://docs.khadas.com/android/zh-cn/vim3/) , [VIM3/3L Linux 文档](https://docs.khadas.com/linux/zh-cn/vim3)，其中有详细描述刷机方法。
      - 其余开发板用户可向开发板卖家或官网客服索要 6.4.4.3 版本 NPU 驱动对应的固件和刷机方法。。
    - 方法二：手动替换驱动文件和依赖库，在[PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)中的指定目录下找到不同版本、不同芯片型号的 Linux Kernel 驱动和预编译库：（详细目录树结构可以参考后文『运行图像分类示例程序』）：
      - 如果您的开发板是 Linux 系统，驱动和预编译库存放在 PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/amlogic_npu 目录。
      - 如果您的开发板是 Android 系统，驱动和预编译库存放在 PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/amlogic_npu 目录。
      - 第一步，执行 ./switch_amlnpu_ddk.sh 6_4_4_3 {SoC型号}，以s905d3芯片为例：./switch_amlnpu_ddk.sh 6_4_4_3 s905d3。请注意当前我们提供的是 linux 系统下 A311D、S905D3、C308X，以及 Android 系统下 S905D3 的 NPU 驱动和相关依赖库。
      - 第二步，amlnpu_ddk_6_4_4_3/lib/{SoC型号}/{系统版本号}/ 目录下，提供了不同芯片型号、不同 Linux Kernel 版本的 NPU 驱动—— galcore.ko 。比如，用户使用 Android S905D3， Linux Kernel 版本 4.9.113（可通过 uname -a 命令查看 Linux Kernel 版本），则在PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/amlogic_npu/amlnpu_ddk_6_4_4_3/lib/s905d3/4.9.113 下找到 NPU 驱动文件 galcore.ko。注意，不同设备的操作系统版本号不同，如果与我们提供的操作系统版本号不一致，则无法直接使用，此时请参考上文提到的方法『方法一 ：刷机』。
      - 第三步，将 galcore.ko 传到设备上，登录设备，命令行输入 `sudo rmmod galcore` 来卸载原始驱动，输入 `sudo insmod galcore.ko` 来加载传上设备的驱动
      - 第四部，输入 `dmesg | grep Galcore` 查询 NPU 驱动版本，确定为 6.4.4.3


- C308X

  - 需要驱动版本为 6.4.4.3（修改驱动方法请参考前一个小节『确定开发板 NPU 驱动版本』）。
  - 注意是 64 位系统。
  - 将 MicroUSB 线插入到设备的 MicroUSB OTG 口，就可以使用 Android 的 `adb` 命令进行设备的交互，当然也提供了网络连接 SSH 登录的方式。

    - 可通过 `dmesg | grep Galcore` 查询系统版本：

      ```shell
      $ dmesg | grep  Galcore
      [   23.599566] Galcore version 6.4.4.3.310723AAA
      ```

- A311D

  - 需要驱动版本为 6.4.4.3（修改驱动方法请参考前一个小节『确定开发板 NPU 驱动版本』）。

  - 注意是 64 位系统。

  - 将 MicroUSB 线插入到设备的 MicroUSB OTG 口，就可以使用 Android 的 `adb` 命令进行设备的交互，当然也提供了网络连接 SSH 登录的方式。

    - 可通过 `dmesg | grep Galcore` 查询系统版本：

      ```shell
      $ dmesg | grep Galcore
      [   24.140820] Galcore version 6.4.4.3.310723AAA
      ```

- S905D3(Android 版本)

   - 需要驱动版本为 6.4.4.3（修改驱动方法请参考前一个小节『确定开发板 NPU 驱动版本』）：
   - `adb root + adb remount` 以获得修改系统库的权限。

      ```shell
      $ dmesg | grep Galcore
      [    9.020168] <6>[    9.020168@0] Galcore version 6.4.4.3.310723a
      ```

- 示例程序和 Paddle Lite 库的编译建议采用交叉编译方式，通过 `adb` 进行设备的交互和示例程序的运行。


### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[ Docker 环境准备](../source_compile/docker_environment)中的 Docker 开发环境进行配置；
- 由于有些设备只提供网络访问方式（根据开发版的实际情况），需要通过 `scp` 和 `ssh` 命令将交叉编译生成的Paddle Lite 库和示例程序传输到设备上执行，因此，在进入 Docker 容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像分类示例程序

- 下载 Paddle Lite 通用示例程序[PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下（注意其中软链接为 switch_amlnpu_ddk.sh 根据芯片型号和 NPU 驱动版本创建依赖库的软链接）：

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
            - mobilenet_v1_int8_224_per_layer
              - __model__ # Paddle fluid 模型组网文件，可使用 netron 查看网络结构
              — conv1_weights # Paddle fluid 模型参数文件
              - batch_norm_0.tmp_2.quant_dequant.scale # Paddle fluid 模型量化参数文件
              — subgraph_partition_config_file.txt # 自定义子图分割配置文件
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.linux.arm64 # arm64 编译工作目录
            - image_classification_demo # 已编译好的，适用于 arm64 的示例程序
          - build.linux.armhf # armhf编译工作目录
            - image_classification_demo # 已编译好的，适用于 armhf 的示例程序
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序ssh运行脚本
          - run_with_adb.sh # 示例程序adb运行脚本
      - libs
        - PaddleLite
          - linux
            - arm64 # Linux 64 位系统
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - amlogic_npu # Amlogic NPU DDK、NNAdapter 运行时库、device HAL 库
                  - libArchModelSw.so -> ./amlnpu_ddk_6_4_4_3/lib/libArchModelSw.so
                  - libCLC.so -> ./amlnpu_ddk_6_4_4_3/lib/libCLC.so
                  - libGAL.so -> ./amlnpu_ddk_6_4_4_3/lib/libGAL.so
                  - libNNArchPerf.so -> ./amlnpu_ddk_6_4_4_3/lib/libNNArchPerf.so
                  - libNNGPUBinary.so -> ./amlnpu_ddk_6_4_4_3/lib/a311d/libNNGPUBinary.so
                  - libNNVXCBinary.so -> ./amlnpu_ddk_6_4_4_3/lib/a311d/libNNVXCBinary.so
                  - libOpenCL.so -> ./amlnpu_ddk_6_4_4_3/lib/libOpenCL.so
                  - libOpenVX.so -> ./amlnpu_ddk_6_4_4_3/lib/libOpenVX.so
                  - libOpenVXU.so -> ./amlnpu_ddk_6_4_4_3/lib/libOpenVXU.so
                  - libOvx12VXCBinary.so -> ./amlnpu_ddk_6_4_4_3/lib/a311d/libOvx12VXCBinary.so
                  - libVSC.so -> ./amlnpu_ddk_6_4_4_3/lib/libVSC.so
                  - libamlnpu_ddk.so -> ./amlnpu_ddk_6_4_4_3/lib/libamlnpu_ddk.so
                  - libamlogic_npu.so # NNAdapter device HAL 库
                  - libnnadapter.so  # NNAdapter 运行时库
                  - libnnsdk_lite.so -> ./amlnpu_ddk_6_4_4_3/lib/libnnsdk_lite.so
                  - switch_amlnpu_ddk.sh # 根据芯片型号和 NPU 驱动版本创建依赖库的软链接
                  - amlnpu_ddk_6_4_4_3
                    - include
                    - lib
                    - a311d # 针对 a311d 平台
                      - 4.9.241
                        - galcore.ko # NPU 驱动文件
                      - libNNGPUBinary.so # 芯原 DDK
                      - libNNVXCBinary.so # 芯原 DDK
                      - libOvx12VXCBinary.so # 芯原 DDK
                    - c308x # 针对 c308x 平台
                       - 4.19.81
                         - galcore.ko
                       - libNNGPUBinary.so
                       - libNNVXCBinary.so
                       - libOvx12VXCBinary.so
                    - libArchModelSw.so # 芯原 DDK
                    - libCLC.so # 芯原 DDK
                    - libGAL.so # 芯原 DDK
                    - libNNArchPerf.so # 芯原 DDK
                    - libOpenCL.so # 芯原 DDK
                    - libOpenVX.so # 芯原 DDK
                    - libOpenVXU.so # 芯原 DDK
                    - libVSC.so # 芯原 DDK
                    - libamlnpu_ddk.so # 晶晨 NPU DDK
                    - libnnsdk_lite.so # 晶晨 NPU DDK
                    - libovxlib.so
                    - s905d3 # 针对 s905d3 平台
                        - 4.9.241
                          - galcore.ko
                        - libNNGPUBinary.so
                        - libNNVXCBinary.so
                        - libOvx12VXCBinary.so
                - libpaddle_full_api_shared.so # 预编译 PaddleLite full api 库
                - libpaddle_light_api_shared.so # 预编译 PaddleLite light api 库
            ...
          - android
           - armeabi-v7a # Android 32 位系统
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - amlogic_npu # Amlogic NPU DDK、NNAdapter 运行时库、device HAL 库
                  - libCLC.so -> ./amlnpu_ddk_6_4_4_3/lib/libCLC.so
                  - libGAL.so -> ./amlnpu_ddk_6_4_4_3/lib/libGAL.so
                  - libNNArchPerf.so -> ./amlnpu_ddk_6_4_4_3/lib/libNNArchPerf.so
                  - libNNGPUBinary.so -> ./amlnpu_ddk_6_4_4_3/lib/s905d3/libNNGPUBinary.so
                  - libNNVXCBinary.so -> ./amlnpu_ddk_6_4_4_3/lib/s905d3/libNNVXCBinary.so
                  - libOpenCL.so -> ./amlnpu_ddk_6_4_4_3/lib/libOpenCL.so
                  - libOpenVX.so -> ./amlnpu_ddk_6_4_4_3/lib/libOpenVX.so
                  - libOpenVXU.so -> ./amlnpu_ddk_6_4_4_3/lib/libOpenVXU.so
                  - libOvx12VXCBinary.so -> ./amlnpu_ddk_6_4_4_3/lib/s905d3/libOvx12VXCBinary.so
                  - libVSC.so -> ./amlnpu_ddk_6_4_4_3/lib/libVSC.so
                  - libamlnpu_ddk.so -> ./amlnpu_ddk_6_4_4_3/lib/libamlnpu_ddk.so
                  - libamlogic_npu.so # NNAdapter device HAL 库
                  - libarchmodelSw.so -> ./amlnpu_ddk_6_4_4_3/lib/libarchmodelSw.so
                  - libnnadapter.so # NNAdapter 运行时库
                  - libnnrt.so -> ./amlnpu_ddk_6_4_4_3/lib/libnnrt.so
                  - libnnsdk_lite.so -> ./amlnpu_ddk_6_4_4_3/lib/libnnsdk_lite.so
                  - libovxlib.so -> ./amlnpu_ddk_6_4_4_3/lib/libovxlib.so
                  - switch_amlnpu_ddk.sh # 根据芯片型号和 NPU 驱动版本创建依赖库的软链接
                  - amlnpu_ddk_6_4_4_3
                    - include
                    - lib
                      - libCLC.so # 芯原 DDK
                      - libGAL.so # 芯原 DDK
                      - libNNArchPerf.so # 芯原 DDK
                      - libOpenCL.so 
                      - libOpenVX.so # 芯原 DDK
                      - libOpenVXU.so # 芯原 DDK
                      - libVSC.so # 芯原 DDK
                      - libamlnpu_ddk.so # 晶晨 NPU DDK
                      - libarchmodelSw.so # 芯原 DDK
                      - libnnrt.so # amlogic DDK 依赖库
                      - libnnsdk_lite.so # amlogic DDK 依赖库
                      - libovxlib.so # 芯原 DDK
                      - s905d3 # 针对 s905d3 平台
                        - 4.9.113
                            - VERSION
                            - galcore.ko # NPU驱动
                        - libNNGPUBinary.so # 芯原 DDK
                        - libNNVXCBinary.so # 芯原 DDK
                        - libOvx12VXCBinary.so # 芯原 DDK
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
        - OpenCV # OpenCV 预编译库
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
  ```

- 按照以下命令分别运行转换后的ARM CPU模型和Amlogic NPU模型，比较它们的性能和结果；

  ```shell
  注意：
  1）`run_with_adb.sh` 不能在 Docker 环境执行，否则可能无法找到设备，也不能在设备上运行。
  2）`run_with_ssh.sh` 不能在设备上运行，且执行前需要配置目标设备的 IP 地址、SSH 账号和密码。
  3）`build.sh` 根据入参生成针对不同操作系统、体系结构的二进制程序，需查阅注释信息配置正确的参数值。
  4）`run_with_adb.sh` 入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。
  5）`run_with_ssh.sh` 入参包括模型名称、操作系统、体系结构、目标设备、ip地址、用户名、用户密码等，需查阅注释信息配置正确的参数值。
  6）下述命令行示例中涉及的具体IP、SSH账号密码、设备序列号等均为示例环境，请用户根据自身实际设备环境修改。
  
  在 ARM CPU 上运行 mobilenet_v1_int8_224_per_layer 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  
  For C308X
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu 192.168.100.244 22 root 123456
    (C308X)
    warmup: 1 repeat: 5, average: 167.6916 ms, max: 207.458000 ms, min: 159.823239 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.423000 ms
    Prediction time: 167.6996 ms
    Postprocess time: 0.542000 ms
  
  For A311D
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu 0123456789ABCDEF
    (A311D)
    warmup: 1 repeat: 15, average: 81.678067 ms, max: 81.945999 ms, min: 81.591003 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 1.352000 ms
    Prediction time: 81.678067 ms
    Postprocess time: 0.407000 ms
  
  For S905D3(Android版)
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a cpu c8631471d5cd
    (S905D3(Android版))
    warmup: 1 repeat: 5, average: 280.465997 ms, max: 358.815002 ms, min: 268.549812 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.199000 ms
    Prediction time: 280.465997 ms
    Postprocess time: 0.596000 ms
  
  ------------------------------
  
  在 Amlogic NPU 上运行 mobilenet_v1_int8_224_per_layer 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  
  For C308X
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 amlogic_npu 192.168.100.244 22 root 123456
    (C308X)
    warmup: 1 repeat: 5, average: 6.982800 ms, max: 7.045000 ms, min: 6.951000 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 2.417000 ms
    Prediction time: 6.982800 ms
    Postprocess time: 0.509000 ms
  
  For A311D
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 amlogic_npu 0123456789ABCDEF
    ( A311D)
    warmup: 1 repeat: 15, average: 5.567867 ms, max: 5.723000 ms, min: 5.461000 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 1.356000 ms
    Prediction time: 5.567867 ms
    Postprocess time: 0.411000 ms
  
  For S905D3(Android版)
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a amlogic_npu c8631471d5cd
    (S905D3(Android版))
    warmup: 1 repeat: 5, average: 13.4116 ms, max: 15.751210 ms, min: 12.433400 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 3.170000 ms
    Prediction time: 13.4116 ms
    Postprocess time: 0.634000 ms
  ```
  
- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/images` 目录下，然后调用 `convert_to_raw_image.py` 生成相应的 RGB Raw 图像，最后修改 `run_with_adb.sh`、`run_with_ssh.sh` 的 IMAGE_NAME 变量即可；
- 重新编译示例程序：  
  ```shell
  注意：
  1）请根据 `buid.sh` 配置正确的参数值。
  2）需在 Docker 环境中编译。
  
  # 对于C308X，A311D
  ./build.sh linux arm64
  
  # 对于S905D3(Android版)
  ./build.sh android armeabi-v7a
  ```

### 更新模型
- 通过 Paddle 训练或 X2Paddle 转换得到 MobileNetv1 foat32 模型[ mobilenet_v1_fp32_224 ](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)
- 通过 Paddle+PaddleSlim 后量化方式，生成[ mobilenet_v1_int8_224_per_layer 量化模型](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)
- 下载[ PaddleSlim-quant-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/tools/PaddleSlim-quant-demo.tar.gz)，解压后清单如下：
    ```shell
    - PaddleSlim-quant-demo
      - image_classification_demo
        - quant_post # 后量化
          - quant_post_rockchip_npu.sh # Rockchip NPU 一键量化脚本，Amlogic 和瑞芯微底层都使用芯原的 NPU，所以通用
          - README.md # 环境配置说明，涉及 PaddlePaddle、PaddleSlim 的版本选择、编译和安装步骤
          - datasets # 量化所需要的校准数据集合
            - ILSVRC2012_val_100 # 从 ImageNet2012 验证集挑选的 100 张图片
          - inputs # 待量化的 fp32 模型
            - mobilenet_v1
            - resnet50
          - outputs # 产出的全量化模型
          - scripts # 后量化内置脚本
    ```
- 查看 `README.md` 完成 PaddlePaddle 和 PaddleSlim 的安装
- 直接执行 `./quant_post_rockchip_npu.sh` 即可在 `outputs` 目录下生成mobilenet_v1_int8_224_per_layer 量化模型
  ```shell
  -----------  Configuration Arguments -----------
  activation_bits: 8
  activation_quantize_type: moving_average_abs_max
  algo: KL
  batch_nums: 10
  batch_size: 10
  data_dir: ../dataset/ILSVRC2012_val_100
  is_full_quantize: 1
  is_use_cache_file: 0
  model_path: ../models/mobilenet_v1
  optimize_model: 1
  output_path: ../outputs/mobilenet_v1
  quantizable_op_type: conv2d,depthwise_conv2d,mul
  use_gpu: 0
  use_slim: 1
  weight_bits: 8
  weight_quantize_type: abs_max
  ------------------------------------------------
  quantizable_op_type:['conv2d', 'depthwise_conv2d', 'mul']
  2021-08-30 05:52:10,048-INFO: Load model and set data loader ...
  2021-08-30 05:52:10,129-INFO: Optimize FP32 model ...
  I0830 05:52:10.139564 14447 graph_pattern_detector.cc:91] ---  detected 14 subgraphs
  I0830 05:52:10.148236 14447 graph_pattern_detector.cc:91] ---  detected 13 subgraphs
  2021-08-30 05:52:10,167-INFO: Collect quantized variable names ...
  2021-08-30 05:52:10,168-WARNING: feed is not supported for quantization.
  2021-08-30 05:52:10,169-WARNING: fetch is not supported for quantization.
  2021-08-30 05:52:10,170-INFO: Preparation stage ...
  2021-08-30 05:52:11,853-INFO: Run batch: 0
  2021-08-30 05:52:16,963-INFO: Run batch: 5
  2021-08-30 05:52:21,037-INFO: Finish preparation stage, all batch:10
  2021-08-30 05:52:21,048-INFO: Sampling stage ...
  2021-08-30 05:52:31,800-INFO: Run batch: 0
  2021-08-30 05:53:23,443-INFO: Run batch: 5
  2021-08-30 05:54:03,773-INFO: Finish sampling stage, all batch: 10
  2021-08-30 05:54:03,774-INFO: Calculate KL threshold ...
  2021-08-30 05:54:28,580-INFO: Update the program ...
  2021-08-30 05:54:29,194-INFO: The quantized model is saved in ../outputs/mobilenet_v1
  post training quantization finish, and it takes 139.42292165756226.
  
  -----------  Configuration Arguments -----------
  batch_size: 20
  class_dim: 1000
  data_dir: ../dataset/ILSVRC2012_val_100
  image_shape: 3,224,224
  inference_model: ../outputs/mobilenet_v1
  input_img_save_path: ./img_txt
  save_input_img: False
  test_samples: -1
  use_gpu: 0
  ------------------------------------------------
  Testbatch 0, acc1 0.8, acc5 1.0, time 1.63 sec
  End test: test_acc1 0.76, test_acc5 0.92
  --------finish eval int8 model: mobilenet_v1-------------
  ```
- 参考[模型转化方法](../user_guides/model_optimize_tool)，利用 opt 工具转换生成 Amlogic NPU 模型，仅需要将 `valid_targets` 设置为 `amlogic_npu`, `arm` 即可。
  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_per_layer \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=amlogic_npu,arm
  ```
### 更新支持 Amlogic NPU 的 Paddle Lite 库

- 下载 Paddle Lite 源码和 Amlogic NPU DDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  # C308X、A311D Linux 版本 ddk
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/amlogic/linux/amlnpu_ddk.tar.gz
  # S905D3 Android 版本 ddk
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/amlogic/android/amlnpu_ddk.tar.gz
  $ tar -xvf amlnpu_ddk.tar.gz
  ```

- 编译并生成 `Paddle Lite+Amlogic NPU` 的部署库

  - For C308X and A311D
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_amlogic_npu=ON --nnadapter_amlogic_npu_sdk_root=$(pwd)/amlnpu_ddk
      
      ```
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_amlogic_npu=ON --nnadapter_amlogic_npu_sdk_root=$(pwd)/amlnpu_ddk full_publish
      
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/amlogic_npu/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libamlogic_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/amlogic_npu/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

  - S905D3(Android 版)
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_amlogic_npu=ON --nnadapter_amlogic_npu_sdk_root=$(pwd)/amlnpu_ddk
      ```

    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_amlogic_npu=ON --nnadapter_amlogic_npu_sdk_root=$(pwd)/amlnpu_ddk full_publish
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/amlogic_npu/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libamlogic_npu.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/amlogic_npu/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      # 替换 libpaddle_full_api_shared.so(仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```
  
- 替换头文件后需要重新编译示例程序

## 其它说明

- Amlogic 和 Paddle Lite 研发团队正在持续增加用于适配 Paddle 算子的 `bridge/converter`，以便适配更多 Paddle 模型。
