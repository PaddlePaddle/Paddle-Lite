# 亿智 NPU 部署示例

Paddle Lite 已支持 亿智 NPU (eeasytech NPU) 的预测部署。
其接入原理是使用亿智 NPU DDK(EEASYTECH DDK)。首先加载并分析 Paddle 模型，首先将 Paddle 算子转成 NNAdapter 标准算子，其次再调用 EEASYTECH DDK 组网 API 进行网络构建，在线生成并执行模型。
- 请注意，亿智 NPU 所使用的量化方式与其他芯片不同，scale 需要符合 power(2) 的限制，会在后续量化小节中详细描述

## 支持现状

### 已支持的芯片

- SH506
- SH510
- SV806
- SV810

### 已支持的设备

- SH506/510 开发板
- SV810/806 开发板

### 已支持的 Paddle 模型

#### 模型
- [mobilenet_v1_int8_per_layer_log2](https://paddlelite-demo.bj.bcebos.com/devices/generic/models/mobilenet_v1_int8_per_layer_log2.tar.gz)
- [mobilenet_v2_int8_per_layer_log2](https://paddlelite-demo.bj.bcebos.com/devices/generic/models/mobilenet_v2_int8_per_layer_log2.tar.gz)

#### 性能
- 测试环境
  - 编译环境
    - Ubuntu 16.04，GCC 5.4 for ARMLinux armhf and aarch64

  - 硬件环境
    - SH506 开发板
      - CPU：2 x Cortex-A7
      - NPU：1.2 TOPs for INT8

- 测试方法
  - warmup=1, repeats=5，统计平均时间，单位是 ms
  - 线程数为 1，`paddle::lite_api::PowerMode CPU_POWER_MODE` 设置为 ` paddle::lite_api::PowerMode::LITE_POWER_HIGH`
  - 分类模型的输入图像维度是{1, 3, 224, 224}，检测模型的维度是{1, 3, 300, 300}

- 测试结果

  |模型 |SH506||
  |---|---|---|
  |  |CPU(ms) | NPU(ms) |
  |mobilenet_v1_int8_per_layer_log2|  672.450012|  47.832000|
  |mobilenet_v2_int8_per_layer_log2|  518.518982|  53.127300|


### 已支持（或部分支持）NNAdapter 的 Paddle 算子
您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

## 参考示例演示

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[编译环境准备](../source_compile/compile_env)中的 Docker 开发环境进行配置；
- 由于有些设备只提供网络访问方式(具体看开发板的实际情况)，需要通过 `scp` 和 `ssh` 命令将交叉编译生成的 Paddle Lite 库和示例程序传输到设备上执行，因此，在进入 Docker 容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像分类示例程序

- 下载 Paddle Lite 通用示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下：

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
            - mobilenet_v1_int8_per_layer_log2
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
            - armhf # Linux 32 位系统
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - eeasytech_npu # 亿智 NPU DDK、NNAdapter 运行时库、device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libeeasytech_npu.so.so # NNAdapter device HAL 库
                  - libeznpu_ddk.so.so # 亿智 NPU DDK
                  - libnn.so # 亿智 DDK
                  - libopenvx-nn.so # 亿智 DDK
                  - libopenvx.so # 亿智 DDK
                  - libsoft-nn.so # 亿智 DDK
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            ...
          - android
        - OpenCV # OpenCV 预编译库
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
  ```

- 按照以下命令分别运行转换后的 ARM CPU 模型和 亿智 NPU 模型，比较它们的性能和结果；

  ```shell
  注意：
  1）`run_with_adb.sh` 不能在 Docker 环境执行，否则可能无法找到设备，也不能在设备上运行。
  2）`run_with_ssh.sh` 不能在设备上运行，且执行前需要配置目标设备的IP地址、SSH账号和密码。
  3）`build.sh` 根据入参生成针对不同操作系统、体系结构的二进制程序，需查阅注释信息配置正确的参数值。
  4）`run_with_adb.sh` 入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。
  5）`run_with_ssh.sh` 入参包括模型名称、操作系统、体系结构、目标设备、ip 地址、用户名、用户密码等，需查阅注释信息配置正确的参数值。
  6）下述命令行示例中涉及的具体IP、SSH账号密码、设备序列号等均为示例环境，请用户根据自身实际设备环境修改。

  在 ARM CPU 上运行 mobilenet_v1_int8_per_layer_log2 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell

  For SH506 CPU
  $ ./run_with_adb.sh mobilenet_v1_int8_per_layer_log2 linux armhf cpu adb设备号
    (RK1808 EVB)
    warmup: 1 repeat: 5, average: 517.333008 ms, max: 519.331000 ms, min: 516.848999 ms
    results: 3
    Top0  tabby, tabby cat - 0.638649
    Top1  Egyptian cat - 0.289704
    Top2  tiger cat - 0.051178
    Preprocess time: 6.928000 ms
    Prediction time: 517.333008 ms
    Postprocess time: 0.538000 ms

  ------------------------------

  在 EEASYTECH NPU 上运行 mobilenet_v1_int8_per_layer_log2 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell

  For SH506 NPU
  $ ./run_with_adb.sh mobilenet_v1_int8_per_layer_log2 linux armhf eeasytech_npu adb设备号
    (SH596)
    warmup: 1 repeat: 5, average: 52.715000 ms, max: 54.652100 ms, min: 51.233000 ms
    results: 3
    Top0  tabby, tabby cat - 0.708991
    Top1  Egyptian cat - 0.125688
    Top2  tiger cat - 0.051297
    Preprocess time: 6.935000 ms
    Prediction time: 50.715000 ms
    Postprocess time: 0.897000 ms
  ```

- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/images` 目录下，然后调用 `convert_to_raw_image.py` 生成相应的 RGB Raw 图像，最后修改 `run_with_adb.sh`、`run_with_ssh.sh` 的 IMAGE_NAME 变量即可；
- 重新编译示例程序：  
  ```shell
  注意：
  1）请根据 `buid.sh`配置正确的参数值。
  2）需在 Docker 环境中编译。

  ./build.sh linux armhf
  ```

### 更新模型
- 通过 Paddle 训练或 X2Paddle 转换得到 MobileNetv1 foat32 模型[ mobilenet_v1_fp32_224 ](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)
- 下载[ PaddleSlim-quant-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/tools/PaddleSlim-quant-demo.tar.gz)，解压后清单如下：
    ```shell
    - PaddleSlim-quant-demo
      - image_classification_demo
        - quant_post # 后量化
          - quant_post_rockchip_npu.sh # 一键量化脚本
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
- 请注意，亿智 NPU 所使用的量化方式与其他芯片不同，scale 需要符合 power(2) 的限制，因此需要对 PaddleSlim 的 python 包做小幅修改。
 - 在完成 PaddlePaddle 和 PaddleSlim 的安装后，命令行输出 `python -c "import paddle; print(paddle)"` 找到 PaddlePaddle 的 python 包，例如 '/usr/local/lib/python3.7/site-packages/paddle/__init__.py'，既 PaddlePaddle 的 python 包路径为 '/usr/local/lib/python3.7/site-packages/paddle/'，进入该目录，并找到文件fluid/contrib/slim/quantization/post_training_quantization.py，备份
 - 下载符合亿智 NPU 量化限制的 [post_training_quantization.py](https://paddlelite-demo.bj.bcebos.com/tools/PaddleSlim-log2-quant/post_training_quantization.py)，替换原本的post_training_quantization.py
- 回到 PaddleSlim-quant-demo 中，直接执行 `./quant_post_rockchip_npu.sh` 即可在 `outputs` 目录下生成 mobilenet_v1_int8_per_layer_log2 量化模型
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
  - 参考[模型转化方法](../user_guides/model_optimize_tool)，利用 opt 工具转换生成 eeasytech NPU 模型，仅需要将 `valid_targets` 设置为 eeasytech_npu,arm 即可。
  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_per_layer \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=eeasytech_npu,arm
  ```
### 更新支持 EEASYTECH NPU 的 Paddle Lite 库

- 下载 Paddle Lite 源码和 EEASYTECH NPU DDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/eeasytech/eznpu_ddk.tar.gz
  $ tar -zxvf eznpu_ddk.tar.gz
  ```

- 编译并生成 `Paddle Lite + EEASYTECH NPU` for armhf 的部署库

  - For SH506/510 SV810/806 Linux armhf 
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --toolchain=clang --with_extra=ON --with_log=ON --with_exception=ON --arch=armv7hf  --with_nnadapter=ON --nnadapter_with_eeasytech_npu=ON --nnadapter_eeasytech_npu_sdk_root=$(pwd)/eznpu_ddk
      ```

    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --toolchain=clang --with_extra=ON --with_log=ON --with_exception=ON --arch=armv7hf  --with_nnadapter=ON --nnadapter_with_eeasytech_npu=ON --nnadapter_eeasytech_npu_sdk_root=$(pwd)/eznpu_ddk full_publish
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv7hf.clang/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv7hf.clang/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/eeasytech_npu/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv7hf.clang/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libeeasytech_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/eeasytech_npu/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv7hf.clang/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv7hf.clang/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      ```
  
- 替换头文件后需要重新编译示例程序

## 其它说明

- 亿智研发同学正在持续增加用于适配 Paddle Lite NNAdapter 算子，以便适配更多 Paddle 模型。
