# PaddleLite使用颖脉NNA预测部署

Paddle Lite已支持Imagination NNA的预测部署。
其接入原理是与之前华为Kirin NPU类似，即加载并分析Paddle模型，将Paddle算子转成Imagination DNN APIs进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- 紫光展锐虎贲T7510

### 已支持的设备

- 海信F50，Roc1开发板（基于T7510的微型电脑主板）
- 酷派X10（暂未提供demo）

### 已支持的Paddle模型

- [全量化MobileNetV1](https://paddlelite-demo.bj.bcebos.com/devices/imagination/mobilenet_v1_int8_224_fluid.tar.gz)

### 已支持（或部分支持）的Paddle算子

- relu
- conv2d
- depthwise_conv2d
- pool2d
- fc

## 参考示例演示

### 测试设备(Roc1开发板)

![roc1_front](https://paddlelite-demo.bj.bcebos.com/devices/imagination/Roc1_front.jpg)

![roc1_back](https://paddlelite-demo.bj.bcebos.com/devices/imagination/Roc1_back.jpg)

### 准备设备环境

- 需要依赖特定版本的firmware，请联系Imagination相关研发同学；
- 确定能够通过SSH方式远程登录Roc 1开发板；
- 由于Roc 1的ARM CPU能力较弱，示例程序和PaddleLite库的编译均采用交叉编译方式。

### 准备交叉编译环境

- 按照以下两种方式配置交叉编译环境：
  - Docker交叉编译环境：由于Roc1运行环境为Ubuntu 18.04，且Imagination NNA DDK依赖高版本的glibc，因此不能直接使用[编译环境准备](../source_compile/compile_env)中的docker image，而需要按照如下方式在Host机器上手动构建Ubuntu 18.04的docker image；

    ```
    $ wget https://paddlelite-demo.bj.bcebos.com/devices/imagination/Dockerfile
    $ docker build --network=host -t paddlepaddle/paddle-lite-ubuntu18_04:1.0 .
    $ docker run --name paddle-lite-ubuntu18_04 --net=host -it --privileged -v $PWD:/Work -w /Work paddlepaddle/paddle-lite-ubuntu18_04:1.0 /bin/bash
    ```

  - Ubuntu交叉编译环境：要求Host为Ubuntu 18.04系统，参考[编译环境准备](../source_compile/compile_env)中的"交叉编译ARM Linux"步骤安装交叉编译工具链。
- 由于需要通过scp和ssh命令将交叉编译生成的PaddleLite库和示例程序传输到设备上执行，因此，在进入Docker容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像分类示例程序

- 下载示例程序[PaddleLite-linux-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/imagination/PaddleLite-linux-demo.tar.gz)，解压后清单如下：

  ```shell
  - PaddleLite-linux-demo
    - image_classification_demo
      - assets
        - images 
          - tabby_cat.jpg # 测试图片
          - tabby_cat.raw # 已处理成raw数据的测试图片
        - labels
          - synset_words.txt # 1000分类label文件
        - models
          - mobilenet_v1_int8_224_for_cpu_fluid # Paddle fluid non-combined格式的、适用于ARM CPU的mobilenetv1量化模型
          - mobilenet_v1_int8_224_for_imagination_nna_fluid # Paddle fluid non-combined格式的、适用于Imagination NNA的mobilenetv1全量化模型
          - mobilenet_v1_int8_224_for_cpu
            - model.nb # 已通过opt转好的、适合ARM CPU的mobilenetv1量化模型
          - mobilenet_v1_int8_224_for_imagination_nna
            - model.nb # 已通过opt转好的、适合Imagination NNA的mobilenetv1全量化模型
      - shell
        - CMakeLists.txt # 示例程序CMake脚本
        - build
          - image_classification_demo # 已编译好的示例程序
        - image_classification_demo.cc # 示例程序源码
        - convert_to_raw_image.py # 将测试图片保存为raw数据的python脚本
        - build.sh # 示例程序编译脚本
        - run.sh # 示例程序运行脚本
    - libs
      - PaddleLite
        - arm64
          - include # PaddleLite头文件
          - lib
            - libcrypto.so.1.1
            - libssl.so.1.1
            - libz.so.1.2.11
            - libgomp.so.1 # gnuomp库
            - libimgcustom.so # Imagination NNA的部分layer的软件实现，PaddleLite暂时没有用到
            - libimgdnn.so # Imagination NNA的DNN组网、编译和执行接口库
            - libnnasession.so # Imagination NNA的推理runtime库
            - nna_config # Imagination NNA硬件和模型编译（mapping）配置文件，运行测试程序时，一定要放在可执行程序的同级目录下
            - libpaddle_light_api_shared.so # 用于最终移动端部署的预编译PaddleLite库（tiny publish模式下编译生成的库）
            - libpaddle_full_api_shared.so # 用于直接加载Paddle模型进行测试和Debug的预编译PaddleLite库（full publish模式下编译生成的库）
  ```

- 按照以下命令分别运行转换后的ARM CPU模型和Imagination NNA模型，比较它们的性能和结果；

  ```shell
  注意：
  1）run.sh必须在Host机器上运行，且执行前需要配置目标设备的IP地址、SSH账号和密码；
  2）build.sh建议在docker环境中执行，目前只支持arm64。

  运行适用于ARM CPU的mobilenetv1全量化模型
  $ cd PaddleLite-linux-demo/image_classification_demo/assets/models
  $ cp mobilenet_v1_int8_224_for_cpu/model.nb mobilenet_v1_int8_224_for_cpu_fluid.nb
  $ cd ../../shell
  $ vim ./run.sh
    MODEL_NAME设置为mobilenet_v1_int8_224_for_cpu_fluid
  $ ./run.sh
    warmup: 5 repeat: 10, average: 61.408800 ms, max: 61.472000 ms, min: 61.367001 ms
    results: 3
    Top0  tabby, tabby cat - 0.522023
    Top1  Egyptian cat - 0.395266
    Top2  tiger cat - 0.073605
    Preprocess time: 0.834000 ms
    Prediction time: 61.408800 ms
    Postprocess time: 0.161000 ms

  运行适用于Imagination NNA的mobilenetv1全量化模型
  $ cd PaddleLite-linux-demo/image_classification_demo/assets/models
  $ cp mobilenet_v1_int8_224_for_imagination_nna/model.nb mobilenet_v1_int8_224_for_imagination_nna_fluid.nb
  $ cd ../../shell
  $ vim ./run.sh
    MODEL_NAME设置为mobilenet_v1_int8_224_for_imagination_nna_fluid
  $ ./run.sh
    warmup: 5 repeat: 10, average: 18.024800 ms, max: 19.073000 ms, min: 17.368999 ms
    results: 3
    Top0  Egyptian cat - 0.039642
    Top1  tabby, tabby cat - 0.039642
    Top2  tiger cat - 0.026363
    Preprocess time: 0.815000 ms
    Prediction time: 18.024800 ms
    Postprocess time: 0.169000 ms

  ```

- 如果需要更改测试图片，可通过convert_to_raw_image.py工具生成；
- 如果需要重新编译示例程序，直接运行./build.sh即可，注意：build.sh的执行建议在docker环境中，否则可能编译出错。


### 更新模型

- 通过Paddle Fluid训练，或X2Paddle转换得到MobileNetv1 foat32模型[mobilenet_v1_fp32_224_fluid](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)；
- 参考[模型量化-有校准数据训练后量化](../user_guides/post_quant_with_data)使用PaddleSlim对float32模型进行量化（注意：由于Imagination NNA只支持tensor-wise的全量化模型，在启动量化脚本时请注意相关参数的设置），最终得到全量化MobileNetV1模型[mobilenet_v1_int8_224_fluid](https://paddlelite-demo.bj.bcebos.com/devices/imagination/mobilenet_v1_int8_224_fluid.tar.gz)；
- 参考[模型转化方法](../user_guides/model_optimize_tool)，利用opt工具转换生成Imagination NNA模型，仅需要将valid_targets设置为imagination_nna,arm即可。

  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_for_imagination_nna_fluid \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=imagination_nna,arm
  
  替换自带的Imagination NNA模型
  $ cp opt_model.nb mobilenet_v1_int8_224_for_imagination_nna/model.nb
  ```

- 注意：opt生成的模型只是标记了Imagination NNA支持的Paddle算子，并没有真正生成Imagination NNA模型，只有在执行时才会将标记的Paddle算子转成Imagination DNN APIs，最终生成并执行模型。

### 更新支持Imagination NNA的Paddle Lite库

- 下载PaddleLite源码和Imagination NNA DDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  $ curl -L https://paddlelite-demo.bj.bcebos.com/devices/imagination/imagination_nna_sdk.tar.gz -o - | tar -zx
  ```

- 编译并生成PaddleLite+ImaginationNNA for armv8的部署库

  ```shell
  For Roc1
  tiny_publish
  $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_imagination_nna=ON --imagination_nna_sdk_root=./imagination_nna_sdk
  full_publish
  $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_imagination_nna=ON --imagination_nna_sdk_root=./imagination_nna_sdk full_publish
  ```

- 将编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.nna/cxx/include替换PaddleLite-linux-demo/libs/PaddleLite/arm64/include目录；
- 将tiny_publish模式下编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.nna/cxx/lib/libpaddle_light_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/arm64/lib/libpaddle_light_api_shared.so文件；
- 将full_publish模式下编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.nna/cxx/lib/libpaddle_full_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/arm64/lib/libpaddle_full_api_shared.so文件。

## 其它说明

- Imagination研发同学正在持续增加用于适配Paddle算子bridge/converter，以便适配更多Paddle模型。
