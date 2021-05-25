# PaddleLite使用瑞芯微NPU预测部署

Paddle Lite已支持Rockchip NPU的预测部署。
其接入原理是与之前华为Kirin NPU类似，即加载并分析Paddle模型，将Paddle算子转成Rockchip NPU组网API进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- RK1808/1806
- RV1126/1109
注意：暂时不支持RK3399Pro

### 已支持的设备

- RK1808/1806 EVB
- TB-RK1808S0 AI计算棒
- RV1126/1109 EVB

### 已支持的Paddle模型

#### 模型
- [MobileNetV1-int8全量化模型](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)
- [ResNet50-int8全量化模型](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/resnet50_int8_224_fluid.tar.gz)

#### 性能
- 测试环境
  - 编译环境
    - Ubuntu 16.04，GCC 5.4 for ARMLinux armhf and aarch64

  - 硬件环境
    - RK1808EVB/TB-RK1808S0 AI计算棒
      - CPU：2 x Cortex-A35 1.6 GHz
      - NPU：3 TOPs for INT8 / 300 GOPs for INT16 / 100 GFLOPs for FP16

    - RV1109EVB
      - CPU：2 x Cortex-A7 1.2 GHz
      - NPU：1.2Tops，support INT8/ INT16

- 测试方法
  - warmup=10, repeats=30，统计平均时间，单位是ms
  - 线程数为1，```DeviceInfo::Global().SetRunMode```设置LITE_POWER_HIGH
  - 分类模型的输入图像维度是{1, 3, 224, 224}

- 测试结果

  |模型 |RK1808EVB||TB-RK1808S0 AI计算棒||RV1109EVB||
  |---|---|---|---|---|---|---|
  |  |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |
  |MobileNetV1-int8|  266.623505|  6.139|  359.007996|  9.4335|  335.03993|  6.6995|
  |ResNet50-int8|  1488.346999|  18.19899|  1983.601501|  23.5935|  1960.27252|  29.8895|

### 已支持（或部分支持）的Paddle算子

- relu
- conv2d
- depthwise_conv2d
- pool2d
- fc
- softmax
- batch_norm
- concat
- elementwise_add
- elementwise_sub
- elementwise_mul
- elementwise_div
- transpose2
- reshape2
- sigmoid
- scale
- flatten
- flatten2
- pad2d

可以通过访问[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/rknpu/bridges/paddle_use_bridges.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/rknpu/bridges/paddle_use_bridges.h)获得最新的算子支持列表。

## 参考示例演示

### 测试设备

- RK1808 EVB

  ![rk1808_evb_front](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rk1808_evb_front.jpg)

  ![rk1808_evb_back](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rk1808_evb_back.jpg)

- TB-RK1808S0 AI计算棒

  ![tb-rk1808s0](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/TB-RK1808S0.jpg)

- RV1126 EVB

   ![rk1126_evb](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rv1126_evb.jpg)

### 准备设备环境

- RK1808 EVB

  - 需要依赖特定版本的firmware，请参照[rknpu_ddk](https://github.com/airockchip/rknpu_ddk)的说明对设备进行firmware的更新；
  - 由于RK1808 EVB在刷firmware后，只是一个纯净的Linux系统，无法像Ubuntu那样使用apt-get命令方便的安装软件，因此，示例程序和PaddleLite库的编译均采用交叉编译方式；
  - 将MicroUSB线插入到设备的MicroUSB OTG口，就可以使用Android的adb命令进行设备的交互，再也不用配置网络使用ssh或者通过串口的方式访问设备了，这个设计非常赞！
  - **将rknpu_ddk的lib64目录下除librknpu_ddk.so之外的动态库都拷贝到设备的/usr/lib目录下，更新Rockchip NPU的系统库。**

- TB-RK1808S0 AI计算棒

  - 参考[TB-RK1808S0 wiki教程的](http://t.rock-chips.com/wiki.php?mod=view&pid=28)将计算棒配置为主动模式，完成网络设置和firmware的升级，具体步骤如下：
    - 将计算棒插入Window7/10主机，参考[主动模式开发](http://t.rock-chips.com/wiki.php?mod=view&id=66)配主机的虚拟网卡IP地址，通过ssh toybrick@192.168.180.8验证是否能登录计算棒；
    - 参考[Window7/10系统配置计算棒网络共享](http://t.rock-chips.com/wiki.php?mod=view&id=77)，SSH登录计算棒后通过wget www.baidu.com验证是否能够访问外网；
    - 参考[固件在线升级](http://t.rock-chips.com/wiki.php?mod=view&id=148)，建议通过ssh登录计算棒，在shell下执行sudo dnf update -y命令快速升级到最新版本系统（要求系统版本>=1.4.1-2），可通过rpm -qa | grep toybrick-server查询系统版本：

    ```shell
    $ rpm -qa | grep toybrick-server
    toybrick-server-1.4.1-2.rk1808.fc28.aarch64
    ```
    - **将rknpu_ddk的lib64目录下除librknpu_ddk.so之外的动态库都拷贝到设备的/usr/lib目录下，更新Rockchip NPU的系统库。**

- RV1126 EVB

   - 需要升级1.51的firmware（下载和烧录方法请联系RK相关同学），可通过以下命令确认librknn_runtime.so的版本：

    ```shell
    # strings /usr/lib/librknn_runtime.so | grep build |grep version
    librknn_runtime version 1.5.1 (161f53f build: 2020-11-05 15:12:30 base: 1126)
    ```

   - 示例程序和PaddleLite库的编译需要采用交叉编译方式，通过adb进行设备的交互和示例程序的运行。
   - **将rknpu_ddk的lib目录下除librknpu_ddk.so之外的动态库都拷贝到设备的/usr/lib目录下，更新Rockchip NPU的系统库。**
   

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[编译环境准备](../source_compile/compile_env)中的Docker开发环境进行配置；
- 由于有些设备只提供网络访问方式（例如：TB-RK1808S0 AI计算棒），需要通过scp和ssh命令将交叉编译生成的PaddleLite库和示例程序传输到设备上执行，因此，在进入Docker容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像分类示例程序

- 下载示例程序[PaddleLite-linux-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/PaddleLite-linux-demo_v2_9_0.tar.gz)，解压后清单如下：

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
          - mobilenet_v1_int8_224_for_cpu_fluid # Paddle fluid non-combined格式的、适用于ARM CPU的MobileNetV1-int8量化模型
          - mobilenet_v1_int8_224_for_rockchip_npu_fluid # Paddle fluid non-combined格式的、适用于Rockchip NPU的MobileNetV1-int8全量化模型
          - mobilenet_v1_int8_224_for_cpu
            - model.nb # 已通过opt转好的、适合ARM CPU的obileNetV1-int8量化模型
          - mobilenet_v1_int8_224_for_rockchip_npu
            - model.nb # 已通过opt转好的、适合Rockchip NPU的MobileNetV1-int8全量化模型
          - resnet50_int8_224_for_cpu_fluid # Paddle fluid non-combined格式的、适用于ARM CPU的ResNet50-int8量化模型
          - resnet50_int8_224_for_rockchip_npu_fluid # Paddle fluid non-combined格式的、适用于Rockchip NPU的ResNet50-int8全量化模型
          - resnet50_int8_224_for_cpu
            - model.nb # 已通过opt转好的、适合ARM CPU的ResNet50-int8量化模型
          - resnet50_int8_224_for_rockchip_npu
            - model.nb # 已通过opt转好的、适合Rockchip NPU的ResNet50-int8全量化模型
      - shell
        - CMakeLists.txt # 示例程序CMake脚本
        - build
          - image_classification_demo # 已编译好的示例程序
        - image_classification_demo.cc # 示例程序源码
        - convert_to_raw_image.py # 将测试图片保存为raw数据的python脚本
        - build.sh # 示例程序编译脚本
        - run_with_adb.sh # RK1808/RK1806/RV1126/RV1109 EVB的示例程序运行脚本
        - run_with_ssh.sh # TB-RK1808S0 AI计算棒的示例程序运行脚本
    - libs
      - PaddleLite
        - arm64 # 适用于RK1808 EVB和TB-RK1808S0 AI计算棒的PaddleLite预编译库
          - include # PaddleLite头文件
          - lib
            - librknpu_ddk.so # RK DDK库
            - libgomp.so.1 # gnuomp库
            - libpaddle_light_api_shared.so # 预编译PaddleLite库
        - armhf # 适用于RK1806/RV1126/RV1109 EVB的PaddleLite预编译库
  ```

- 按照以下命令分别运行转换后的ARM CPU模型和Rockchip NPU模型，比较它们的性能和结果；

  ```shell
  注意：
  1）run_with_adb.sh不能在docker环境执行，否则可能无法找到设备，也不能在设备上运行；
  2）run_with_ssh.sh不能在设备上运行，且执行前需要配置目标设备的IP地址、SSH账号和密码；
  2）build.sh需要在docker环境中执行，如果需要测试armhf库，可将build.sh的TARGET_ARCH_ABI修改成armhf后重新生成image_classification_demo。

  运行适用于ARM CPU的mobilenetv1全量化模型
  $ cd PaddleLite-linux-demo/image_classification_demo/assets/models
  $ cp mobilenet_v1_int8_224_for_cpu/model.nb mobilenet_v1_int8_224_for_cpu_fluid.nb
  $ cd ../../shell
  $ vim ./run_with_adb.sh 或 vim ./run_with_ssh.sh
    MODEL_NAME设置为mobilenet_v1_int8_224_for_cpu_fluid

  For RK1808 EVB
  $ ./run_with_adb.sh arm64
    (RK1808 EVB)
    warmup: 5 repeat: 10, average: 266.276001 ms, max: 266.576996 ms, min: 266.158997 ms
    results: 3
    Top0  tabby, tabby cat - 0.522023
    Top1  Egyptian cat - 0.395266
    Top2  tiger cat - 0.073605
    Preprocess time: 2.684000 ms
    Prediction time: 266.276001 ms
    Postprocess time: 0.456000 ms

  For RK1806/RV1126/RV1109 EVB
  $ ./build.sh armhf
  $ ./run_with_adb.sh armhf
    (RV1126 EVB)
    warmup: 5 repeat: 10, average: 338.019904 ms, max: 371.528992 ms, min: 331.010010 ms
    results: 3
    Top0  tabby, tabby cat - 0.522023
    Top1  Egyptian cat - 0.395266
    Top2  tiger cat - 0.073605
    Preprocess time: 3.443000 ms
    Prediction time: 338.019904 ms
    Postprocess time: 0.600000 ms
  
    (RV1109 EVB)
    warmup: 5 repeat: 10, average: 335.438400 ms, max: 346.362000 ms, min: 331.894012 ms
    results: 3
    Top0  tabby, tabby cat - 0.522023
    Top1  Egyptian cat - 0.395266
    Top2  tiger cat - 0.073605
    Preprocess time: 3.420000 ms
    Prediction time: 335.438400 ms
    Postprocess time: 0.582000 ms

  For TB-RK1808S0 AI计算棒
  $ ./run_with_ssh.sh arm64
    (TB-RK1808S0 AI计算棒)
    warmup: 5 repeat: 10, average: 358.836304 ms, max: 361.001007 ms, min: 358.035004 ms
    results: 3
    Top0  tabby, tabby cat - 0.522023
    Top1  Egyptian cat - 0.395266
    Top2  tiger cat - 0.073605
    Preprocess time: 3.670000 ms
    Prediction time: 358.836304 ms
    Postprocess time: 0.542000 ms

  运行适用于Rockchip NPU的mobilenetv1全量化模型
  $ cd PaddleLite-linux-demo/image_classification_demo/assets/models
  $ cp mobilenet_v1_int8_224_for_rockchip_npu/model.nb mobilenet_v1_int8_224_for_rockchip_npu_fluid.nb
  $ cd ../../shell
  $ vim ./run_with_adb.sh 或 vim ./run_with_ssh.sh
    MODEL_NAME设置为mobilenet_v1_int8_224_for_rockchip_npu_fluid

  For RK1808 EVB
  $ ./run_with_adb.sh arm64
    (RK1808 EVB)
    warmup: 5 repeat: 10, average: 6.663300 ms, max: 6.705000 ms, min: 6.590000 ms
    results: 3
    Top0  Egyptian cat - 0.514779
    Top1  tabby, tabby cat - 0.421183
    Top2  tiger cat - 0.052648
    Preprocess time: 2.391000 ms
    Prediction time: 6.663300 ms
    Postprocess time: 0.470000 ms

  For RK1806/RV1126/RV1109 EVB
  $ ./build.sh armhf
  $ ./run_with_adb.sh armhf
    (RV1126 EVB)
    warmup: 5 repeat: 10, average: 5.956600 ms, max: 6.083000 ms, min: 5.860000 ms
    results: 3
    Top0  Egyptian cat - 0.497230
    Top1  tabby, tabby cat - 0.409483
    Top2  tiger cat - 0.081897
    Preprocess time: 3.514000 ms
    Prediction time: 5.956600 ms
    Postprocess time: 0.539000 ms
  
    (RV1109 EVB)
    warmup: 5 repeat: 10, average: 7.163200 ms, max: 7.459000 ms, min: 7.055000 ms
    results: 3
    Top0  Egyptian cat - 0.497230
    Top1  tabby, tabby cat - 0.409483
    Top2  tiger cat - 0.081897
    Preprocess time: 3.465000 ms
    Prediction time: 7.163200 ms
    Postprocess time: 0.595000 ms

  For TB-RK1808S0 AI计算棒
  $ ./run_with_ssh.sh arm64
    (TB-RK1808S0 AI计算棒)
    warmup: 5 repeat: 10, average: 9.819400 ms, max: 9.970000 ms, min: 9.776000 ms
    results: 3
    Top0  Egyptian cat - 0.514779
    Top1  tabby, tabby cat - 0.421183
    Top2  tiger cat - 0.052648
    Preprocess time: 4.277000 ms
    Prediction time: 9.819400 ms
    Postprocess time: 5.776000 ms

  ```

- 如果需要更改测试图片，可通过convert_to_raw_image.py工具生成；
- 如果需要重新编译示例程序，直接运行./build.sh即可，注意：build.sh的执行必须在docker环境中，否则可能编译出错。


### 更新模型

- 通过Paddle训练或X2Paddle转换得到MobileNetv1 foat32模型[mobilenet_v1_fp32_224_fluid](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)；
- 通过Paddle+PaddleSlim后量化方式，生成[MobileNetV1-int8全量化模型](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)
  - 下载[PaddleSlim-quant-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/tools/PaddleSlim-quant-demo.tar.gz)，解压后清单如下：
    ```shell
    - PaddleSlim-quant-demo
      - image_classification_demo
        - quant_post # 后量化
          - quant_post_rockchip_npu.sh # Rockchip NPU 一键量化脚本
          - README.md # 环境配置说明，涉及PaddlePaddle、PaddleSlim的版本选择、编译和安装步骤
          - datasets # 量化所需要的校准数据集合
            - ILSVRC2012_val_100 # 从ImageNet2012验证集挑选的100张图片
          - inputs # 待量化的fp32模型
            - mobilenet_v1
            - resnet50
          - outputs # 产出的全量化模型
          - scripts # 后量化内置脚本
    ```

  - 查看README.md完成PaddlePaddle和PaddleSlim的安装
  - 直接执行./quant_post_rockchip_npu.sh即可在outputs目录下生成MobileNetV1-int8全量化模型
    ```shell
    $ cd PaddleSlim-quant-demo/image_classification_demo/quant_post
    $ ./quant_post_rockchip_npu.sh
    ...
    2021-01-20 14:02:45,671-INFO: Preparation stage ...
    2021-01-20 14:02:47,205-INFO: Run batch: 0
    2021-01-20 14:02:52,124-INFO: Run batch: 5
    2021-01-20 14:02:56,072-INFO: Finish preparation stage, all batch:10
    2021-01-20 14:02:56,079-INFO: Sampling stage ...
    2021-01-20 14:03:08,759-INFO: Run batch: 0
    2021-01-20 14:04:11,117-INFO: Run batch: 5
    2021-01-20 14:05:00,945-INFO: Finish sampling stage, all batch: 10
    2021-01-20 14:05:00,946-INFO: Calculate KL threshold ...
    2021-01-20 14:05:33,664-INFO: Update the program ...
    2021-01-20 14:05:34,400-INFO: The quantized model is saved in ../outputs/mobilenet_v1
    post training quantization finish, and it takes 169.63866925239563.

    --------start eval int8 model: mobilenet_v1-------------
    grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
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
    Testbatch 0, acc1 0.75, acc5 1.0, time 1.38 sec
    End test: test_acc1 0.76, test_acc5 0.93
    --------finish eval int8 model: mobilenet_v1-------------
    ```

- 参考[模型转化方法](../user_guides/model_optimize_tool)，利用opt工具转换生成Rockchip NPU模型，仅需要将valid_targets设置为rknpu,arm即可。

  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_for_rockchip_npu_fluid \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=rknpu,arm

  替换自带的Rockchip NPU模型
  $ cp opt_model.nb mobilenet_v1_int8_224_for_rockchip_npu/model.nb
  ```

- 注意：opt生成的模型只是标记了Rockchip NPU支持的Paddle算子，并没有真正生成Rockchip NPU模型，只有在执行时才会将标记的Paddle算子转成Rockchip NPU组网API，最终生成并执行模型。

### 更新支持Rockchip NPU的Paddle Lite库

- 下载PaddleLite源码和Rockchip NPU DDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  $ git clone https://github.com/airockchip/rknpu_ddk.git
  ```

- 编译并生成PaddleLite+RockchipNPU for armv8 and armv7的部署库

  - For RK1808 EVB and TB-RK1808S0 AI计算棒
    - tiny_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_rockchip_npu=ON --rockchip_npu_sdk_root=./rknpu_ddk

      将tiny_publish模式下编译生成的build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.rknpu/cxx/lib/libpaddle_light_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/arm64/lib/libpaddle_light_api_shared.so文件；
      ```

    - full_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_rockchip_npu=ON --rockchip_npu_sdk_root=./rknpu_ddk full_publish

      将full_publish模式下编译生成的build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.rknpu/cxx/lib/libpaddle_full_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/arm64/lib/libpaddle_full_api_shared.so文件；
      ```
    将编译生成的build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.rknpu/cxx/include替换PaddleLite-linux-demo/libs/PaddleLite/arm64/include目录；

  - For RK1806/RV1126/RV1109 EVB
    - tiny_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_log=ON --with_rockchip_npu=ON --rockchip_npu_sdk_root=./rknpu_ddk

      将tiny_publish模式下编译生成的build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.rknpu/cxx/lib/libpaddle_light_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/armhf/lib/libpaddle_light_api_shared.so文件；
      ```

    - full_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_log=ON --with_rockchip_npu=ON --rockchip_npu_sdk_root=./rknpu_ddk full_publish

      将full_publish模式下编译生成的build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.rknpu/cxx/lib/libpaddle_full_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/armhf/lib/libpaddle_full_api_shared.so文件。
      ```
    将编译生成的build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.rknpu/cxx/include替换PaddleLite-linux-demo/libs/PaddleLite/armhf/include目录；
  
- 替换头文件后需要重新编译示例程序

## 其它说明

- RK研发同学正在持续增加用于适配Paddle算子bridge/converter，以便适配更多Paddle模型。
