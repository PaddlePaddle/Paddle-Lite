# 颖脉 NNA 部署示例

Paddle Lite 已支持 Imagination NNA 的预测部署。
其接入原理是与之前华为 Kirin NPU 类似，即加载并分析 Paddle 模型，将 Paddle 算子转成 Imagination DNN APIs 进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- 紫光展锐虎贲 T7510

### 已支持的设备

- 海信 F50，Roc1 开发板（基于 T7510 的微型电脑主板）
- 酷派 X10（暂未提供 demo）

### 已支持的 Paddle 模型

#### 模型
- [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)

#### 性能
- 测试环境
  - 编译环境
    - Ubuntu 18.04，GCC 5.4 for ARMLinux aarch64

  - 硬件环境
    - 紫光展锐虎贲 T7510
      - Roc1 开发板
      - CPU：4 x Cortex-A75 2.0 GHz + 4 x Cortex-A55 1.8 GHz
      - NNA：4 TOPs @1.0GHz

- 测试方法
  - warmup=1，repeats=5，统计平均时间，单位是 ms
  - 线程数为 1，`paddle::lite_api::PowerMode CPU_POWER_MODE` 设置为 ` paddle::lite_api::PowerMode::LITE_POWER_HIGH`
  - 分类模型的输入图像维度是{1, 3, 224, 224}

- 测试结果

  |模型 |紫光展锐虎贲 T7510||
  |---|---|---|
  |  |CPU(ms) | NPU(ms) |
  |mobilenet_v1_int8_224_per_layer|  61.093601|  3.217800|

### 已支持（或部分支持）的 Paddle 算子

您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

**不经过 NNAdapter 标准算子转换，而是直接将 Paddle 算子转换成 `Imagination NNA IR` 的方案可点击[链接](https://paddle-lite.readthedocs.io/zh/release-v2.9/demo_guides/imagination_nna.html)**。

## 参考示例演示

### 测试设备( Roc1 开发板)

![roc1_front](https://paddlelite-demo.bj.bcebos.com/devices/imagination/Roc1_front.jpg)

![roc1_back](https://paddlelite-demo.bj.bcebos.com/devices/imagination/Roc1_back.jpg)

### 准备设备环境

- 需要依赖特定版本的 firmware，请联系 Imagination 相关研发同学 jason.wang@imgtec.com；
- 确定能够通过 SSH 方式远程登录 Roc 1 开发板；
- 由于 Roc 1 的 ARM CPU 能力较弱，示例程序和 Paddle Lite 库的编译均采用交叉编译方式。

### 准备交叉编译环境

- 按照以下两种方式配置交叉编译环境：
  - Docker 交叉编译环境：由于 Roc1 运行环境为Ubuntu 18.04，且 Imagination NNA DDK 依赖高版本的 glibc，因此不能直接使用[编译环境准备](../source_compile/docker_env)中的 Docker image，而需要按照如下方式在Host机器上手动构建 Ubuntu 18.04 的 Docker image；

    ```
    $ wget https://paddlelite-demo.bj.bcebos.com/devices/imagination/Dockerfile
    $ docker build --network=host -t paddlepaddle/paddle-lite-ubuntu18_04:1.0 .
    $ docker run --name paddle-lite-ubuntu18_04 --net=host -it --privileged -v $PWD:/Work -w /Work paddlepaddle/paddle-lite-ubuntu18_04:1.0 /bin/bash
    ```

  - Ubuntu 交叉编译环境：要求 Host 为 Ubuntu 18.04 系统，参考[编译环境准备](../source_compile/compile_env)中的"交叉编译 ARM Linux "步骤安装交叉编译工具链。
- 由于需要通过 `scp` 和 `ssh` 命令将交叉编译生成的 Paddle Lite 库和示例程序传输到设备上执行，因此，在进入 Docker 容器后还需要安装如下软件：

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
            - mobilenet_v1_int8_224_per_layer # Paddle non-combined 格式的 mobilenet_v1 int8 全量化模型
              - __model__ # Paddle fluid 模型组网文件，可使用 netron 查看网络结构
              — conv1_weights # Paddle fluid 模型参数文件
              - batch_norm_0.tmp_2.quant_dequant.scale # Paddle fluid 模型量化参数文件
              — subgraph_partition_config_file.txt # 自定义子图分割配置文件
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.linux.arm64 # arm64 编译工作目录
            - image_classification_demo # 已编译好的，适用于 arm64 的示例程序
            ...
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run_with_ssh.sh # 示例程序 adb 运行脚本
          - run_with_adb.sh # 示例程序 ssh 运行脚本
          - run.sh # 示例程序运行脚本
      - libs
        - PaddleLite
          - linux
            - arm64
              - include
              - lib
                - imagination_nna # 颖脉 NNA imgdnn DDK、NNAdapter 运行时库、device HAL 库
                  - libimagination_nna.so # NNAdapter device HAL 库
                  - libnnadapter.so # NNAdapter 运行时库
                  - libcrypto.so
                  ...
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
                ...
            ...
          ...
        - OpenCV # OpenCV 预编译库
  ```

- 按照以下命令分别运行转换后的 ARM CPU 模型和 Imagination NNA 模型，比较它们的性能和结果；

  ```shell
  注意：
  1）`run_with_adb.sh` 不能在 Docker 环境执行，否则可能无法找到设备，也不能在设备上运行。
  2）`run_with_ssh.sh` 不能在设备上运行，且执行前需要配置目标设备的IP地址、SSH 账号和密码。
  3）`build.sh` 根据入参生成针对不同操作系统、体系结构的二进制程序，需查阅注释信息配置正确的参数值。
  4）`run_with_adb.sh` 入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。
  5）`run_with_ssh.sh` 入参包括模型名称、操作系统、体系结构、目标设备、ip 地址、用户名、用户密码等，需查阅注释信息配置正确的参数值。

  在 ARM CPU 上运行 mobilenetv1 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu 192.168.100.10 22 img imgroc1
  ...
  iter 0 cost: 61.130001 ms
  iter 1 cost: 61.073002 ms
  iter 2 cost: 61.081001 ms
  iter 3 cost: 61.088001 ms
  iter 4 cost: 61.096001 ms
  warmup: 1 repeat: 5, average: 61.093601 ms, max: 61.130001 ms, min: 61.073002 ms
  results: 3
  Top0  tabby, tabby cat - 0.490191
  Top1  Egyptian cat - 0.441032
  Top2  tiger cat - 0.060051
  Preprocess time: 0.798000 ms
  Prediction time: 61.093601 ms
  Postprocess time: 0.167000 ms


  在 Imagination NNA 上运行 mobilenetv1 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 imagination_nna 192.168.100.10 22 img imgroc1
  ...
  iter 0 cost: 3.288000 ms
  iter 1 cost: 3.220000 ms
  iter 2 cost: 3.167000 ms
  iter 3 cost: 3.268000 ms
  iter 4 cost: 3.146000 ms
  warmup: 1 repeat: 5, average: 3.217800 ms, max: 3.288000 ms, min: 3.146000 ms
  results: 3
  Top0  tabby, tabby cat - 0.514779
  Top1  Egyptian cat - 0.421183
  Top2  tiger cat - 0.052648
  Preprocess time: 0.818000 ms
  Prediction time: 3.217800 ms
  Postprocess time: 0.157000 ms
  ```

  - 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/images` 目录下，然后调用 `convert_to_raw_image.py` 生成相应的 RGB Raw 图像，最后修改 `run_with_adb.sh` 的 IMAGE_NAME 变量即可；
  ```shell
  注意：
  1）请根据 `buid.sh` 配置正确的参数值。
  2）需在 Docker 环境中编译。

  ./build.sh linux arm64
  ```


### 更新模型

- 通过 Paddle Fluid 训练，或 X2Paddle 转换得到 MobileNetv1 foat32 模型[mobilenet_v1_fp32_224_fluid](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)；
- 参考[模型量化-静态离线量化](../user_guides/quant_post_static)使用 PaddleSlim 对  `float32` 模型进行量化（注意：由于 Imagination NNA 只支持 tensor-wise 的全量化模型，在启动量化脚本时请注意相关参数的设置），最终得到全量化 MobileNetV1 模型[mobilenet_v1_int8_224_fluid](https://paddlelite-demo.bj.bcebos.com/devices/imagination/mobilenet_v1_int8_224_fluid.tar.gz)；
- 参考[模型转化方法](../user_guides/model_optimize_tool)，利用 opt 工具转换生成 Imagination NNA 模型，仅需要将 `valid_targets` 设置为 imagination_nna,arm 即可。

  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_per_layer \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=imagination_nna,arm
  
  替换自带的 Imagination NNA 模型
  $ cp opt_model.nb mobilenet_v1_int8_224_per_layer/model.nb
  ```

- 注意：opt 生成的模型只是标记了 Imagination NNA 支持的 Paddle 算子，并没有真正生成 Imagination NNA 模型，只有在执行时才会将标记的 Paddle 算子转成 Imagination DNN APIs，最终生成并执行模型。

### 更新支持 Imagination NNA 的 Paddle Lite 库

- 下载 Paddle Lite 源码和 Imagination NNA DDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  $ curl -L https://paddlelite-demo.bj.bcebos.com/devices/imagination/imagination_nna_sdk.tar.gz -o - | tar -zx
  ```

- 编译并生成 `Paddle Lite + ImaginationNNA` for armv8的部署库

  - For Roc1
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_imagination_nna=ON --nnadapter_imagination_nna_sdk_root=$(pwd)/imagination_nna_sdk
      ```
      
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_imagination_nna=ON --nnadapter_imagination_nna_sdk_root=$(pwd)/imagination_nna_sdk full_publish
      ```

    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/imagination_nna/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libimagination_nna.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/imagination_nna/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

- 替换头文件后需要重新编译示例程序

## 其它说明

- Imagination 研发同学正在持续增加用于适配 Paddle 算子 `bridge/converter`，以便适配更多 Paddle 模型。
