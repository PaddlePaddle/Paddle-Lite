# 芯原 TIM-VX 部署示例

Paddle Lite 已支持通过 TIM-VX 的方式调用芯原 NPU 算力的预测部署。
其接入原理是与其他接入 Paddle Lite 的新硬件类似，即加载并分析 Paddle 模型，首先将 Paddle 算子转成 NNAdapter 标准算子，其次再通过 TIM-VX 的组网 API 进行网络构建，在线编译模型并执行模型。

需要注意的是，芯原（verisilicon）作为 IP 设计厂商，本身并不提供实体SoC产品，而是授权其 IP 给芯片厂商，如：晶晨（Amlogic），瑞芯微（Rockchip）等。因此本文是适用于被芯原授权了 NPU IP 的芯片产品。只要芯片产品没有大副修改芯原的底层库，则该芯片就可以使用本文档作为 Paddle Lite 推理部署的参考和教程。在本文中，晶晨 SoC 中的 NPU 和 瑞芯微 SoC 中的 NPU 统称为芯原 NPU。

本文档与[ 晶晨 NPU 部署示例 ](./amlogic_npu)和[ 瑞芯微 NPU 部署示例 ](./rockchip_npu)中所描述的部署示例相比，虽然涉及的部分芯片产品相同，但前者是通过 IP 厂商芯原的 TIM-VX 框架接入 Paddle Lite，后二者是通过各自芯片 DDK 接入 Paddle Lite。接入方式不同，支持的算子和模型范围也有所区别。

## 支持现状

### 已支持的芯片

- Amlogic A311D

- Amlogic S905D3

  注意：理论上支持所有经过芯原授权了 NPU IP 的 SoC（须有匹配版本的 NPU 驱动，下文描述），上述为经过测试的部分。

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
    - Amlogic A311D
      - CPU：4 x ARM Cortex-A73 \+  2 x ARM Cortex-A53
      - NPU：5 TOPs for INT8
    - Amlogic S905D3(Android 版本)
      - CPU：2 x ARM Cortex-55
      - NPU：1.2 TOPs for INT8

- 测试方法
  - warmup=1, repeats=5，统计平均时间，单位是 ms
  - 线程数为1，`paddle::lite_api::PowerMode CPU_POWER_MODE`设置为` paddle::lite_api::PowerMode::LITE_POWER_HIGH `
  - 分类模型的输入图像维度是{1, 3, 224, 224}
  
- 测试结果

  |模型 |A311D||S905D3(Android 版本)||
  |---|---|---|---|---|
  |  |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |
  |mobilenet_v1_int8_224_per_layer| 81.632133 | 5.112500 | 280.465997 | 12.808100 |
  |resnet50_int8_224_per_layer| 390.498300| 17.583200 | 787.532340 | 41.313999 |
  |ssd_mobilenet_v1_relu_voc_int8_300_per_layer| 134.991560| 15.216700 | 295.48919| 40.108970 |

### 已支持（或部分支持）NNAdapter 的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

### 测试设备

- Khadas VIM3 开发板（SoC 为 Amlogic A311D）

   <img src="https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/khadas_vim3.jpg" alt="A311D" style="zoom: 20%;" />

  

- Khadas VIM3L 开发板（SoC 为 Amlogic S905D3)

   <img src="https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/khadas_vim3l_android.jpg" alt="A311D" style="zoom: 20%;" />

### 准备设备环境

- A311D

  - 需要驱动版本为 6.4.4.3（下载驱动请联系开发板厂商）。

  - 注意是 64 位系统。

  - 提供了网络连接 SSH 登录的方式，部分系统提供了adb连接的方式。

    - 可通过 `dmesg | grep Galcore` 查询系统版本：

    ```shell
    $ dmesg | grep Galcore
    [   24.140820] Galcore version 6.4.4.3.310723AAA
    ```

- S905D3(Android 版本)

   - 需要驱动版本为 6.4.4.3（下载驱动请联系开发板厂商）。
   - 注意是 32 位系统。
   - `adb root + adb remount` 以获得修改系统库的权限。
   
    ```shell
    $ dmesg | grep Galcore
    [    9.020168] <6>[    9.020168@0] Galcore version 6.4.4.3.310723a
    ```
   
   - 示例程序和 Paddle Lite 库的编译需要采用交叉编译方式，通过 `adb`或`ssh` 进行设备的交互和示例程序的运行。
   

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[ Docker 环境准备](../source_compile/docker_environment)中的 Docker 开发环境进行配置；
- 由于有些设备只提供网络访问方式（根据开发版的实际情况），需要通过 `scp` 和 `ssh` 命令将交叉编译生成的Paddle Lite 库和示例程序传输到设备上执行，因此，在进入 Docker 容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像分类示例程序

- 下载 Paddle Lite 通用示例程序[PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下：

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
          - build.android.armeabi-v7a # Android armv7编译工作目录
            - image_classification_demo # 已编译好的，适用于 Android armv7 的示例程序
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
                - verisilicon_timvx # 芯原 TIM-VX DDK、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libGAL.so # 芯原 DDK
                  - libVSC.so # 芯原 DDK
                  - libOpenVX.so # 芯原 DDK
                  - libarchmodelSw.so # 芯原 DDK
                  - libNNArchPerf.so # 芯原 DDK
                  - libOvx12VXCBinary.so # 芯原 DDK
                  - libNNVXCBinary.so # 芯原 DDK
                  - libOpenVXU.so # 芯原 DDK
                  - libNNGPUBinary.so # 芯原 DDK
                  - libovxlib.so # 芯原 DDK
                  - libOpenCL.so # OpenCL
                  - libverisilicon_timvx.so # # NNAdapter device HAL 库
                  - libtim-vx.so # 芯原 TIM-VX
                  - libgomp.so.1 # gnuomp 库
                - libpaddle_full_api_shared.so # 预编译 PaddleLite full api 库
                - libpaddle_light_api_shared.so # 预编译 PaddleLite light api 库
            ...
          - android
           - armeabi-v7a # Android 32 位系统
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - verisilicon_timvx # 芯原 TIM-VX DDK、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libGAL.so # 芯原 DDK
                  - libVSC.so # 芯原 DDK
                  - libOpenVX.so # 芯原 DDK
                  - libarchmodelSw.so # 芯原 DDK
                  - libNNArchPerf.so # 芯原 DDK
                  - libOvx12VXCBinary.so # 芯原 DDK
                  - libNNVXCBinary.so # 芯原 DDK
                  - libOpenVXU.so # 芯原 DDK
                  - libNNGPUBinary.so # 芯原 DDK
                  - libovxlib.so # 芯原 DDK
                  - libOpenCL.so # OpenCL
                  - libverisilicon_timvx.so # # NNAdapter device HAL 库
                  - libtim-vx.so # 芯原 TIM-VX
                  - libgomp.so.1 # gnuomp 库
                  - libc++_shared.so
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
        - OpenCV # OpenCV 预编译库
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
  ```

- 按照以下命令分别运行转换后的ARM CPU模型和 芯原 TIM-VX 模型，比较它们的性能和结果；

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
  
  For A311D
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu 192.168.100.30 22 khadas khadas
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
  
  在 芯原 NPU 上运行 mobilenet_v1_int8_224_per_layer 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  
  For A311D
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 verisilicon_timvx 192.168.100.30 22 khadas khadas
    (A311D)
    warmup: 1 repeat: 15, average: 5.112500 ms, max: 5.223000 ms, min: 5.009130 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 1.356000 ms
    Prediction time: 5.112500 ms
    Postprocess time: 0.411000 ms
  
  For S905D3(Android版)
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a verisilicon_timvx c8631471d5cd
    (S905D3(Android版))
    warmup: 1 repeat: 5, average: 13.4116 ms, max: 14.7615 ms, min: 12.80810 ms
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
  
  # 对于 A311D
  ./build.sh linux arm64
  
  # 对于 S905D3(Android版)
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
          - quant_post_rockchip_npu.sh # 一键量化脚本，Amlogic 和瑞芯微底层都使用芯原的 NPU，所以通用
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
  - 参考[模型转化方法](../user_guides/model_optimize_tool)，利用 opt 工具转换生成 TIM-VX 模型，仅需要将 `valid_targets` 设置为 `verisilicon_timvx`, `arm` 即可。
  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_per_layer \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=verisilicon_timvx,arm
  ```
### 更新支持 TIM-VX 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  # 注意：编译中依赖的 verisilicon_timvx 相关代码和依赖项会在后续编译脚本中自动下载，无需用户手动下载。
  ```
  
- 编译并生成 `Paddle Lite+Verisilicon_TIMVX` 的部署库

  - For A311D
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_4_3_generic.tgz
      
      ```
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_4_3_generic.tgz full_publish
      
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libverisilicon_timvx.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 芯原 TIM-VX 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libtim-vx.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```
    
  - S905D3(Android 版)
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_exception=ON --with_cv=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_android_9_armeabi_v7a_6_4_4_3_generic.tgz
      ```
  
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_exception=ON --with_cv=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_verisilicon_timvx=ON --nnadapter_verisilicon_timvx_src_git_tag=main --nnadapter_verisilicon_timvx_viv_sdk_url=http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_android_9_armeabi_v7a_6_4_4_3_generic.tgz full_publish
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libverisilicon_timvx.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx/
      # 替换 芯原 TIM-VX 库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libtim-vx.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/verisilicon_timvx/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      # 替换 libpaddle_full_api_shared.so(仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```
  
- 替换头文件后需要重新编译示例程序

## 其它说明

- Paddle Lite 研发团队正在持续扩展基于TIM-VX的算子和模型。
