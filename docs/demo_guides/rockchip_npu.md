# PaddleLite使用瑞芯微NPU预测部署

Paddle Lite已支持Rockchip NPU的预测部署。
其接入原理是与之前华为Kirin NPU类似，即加载并分析Paddle模型，将Paddle算子转成Rockchip NPU组网API进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- RK1808
- RK1806
注意：暂时不支持RK3399Pro

### 已支持的设备

- RK1808/1806 EVB
- TB-RK1808S0 AI计算棒

### 已支持的Paddle模型

- [全量化MobileNetV1](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)

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

## 参考示例演示

### 测试设备

- RK1808 EVB

  ![rk1808_evb_front](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rk1808_evb_front.jpg)

  ![rk1808_evb_back](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rk1808_evb_back.jpg)

- TB-RK1808S0 AI计算棒

  ![tb-rk1808s0](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/TB-RK1808S0.jpg)

### 准备设备环境

- RK1808 EVB

  - 需要依赖特定版本的firmware，请参照[rknpu_ddk](https://github.com/airockchip/rknpu_ddk)的说明对设备进行firmware的更新；
  - 由于RK1808 EVB在刷firmware后，只是一个纯净的Linux系统，无法像Ubuntu那样使用apt-get命令方便的安装软件，因此，示例程序和PaddleLite库的编译均采用交叉编译方式；
  - 将MicroUSB线插入到设备的MicroUSB OTG口，就可以使用Android的adb命令进行设备的交互，再也不用配置网络使用ssh或者通过串口的方式访问设备了，这个设计非常赞！

- TB-RK1808S0 AI计算棒

  - 参考[TB-RK1808S0 wiki教程的](http://t.rock-chips.com/wiki.php?mod=view&pid=28)将计算棒配置为主动模式，完成网络设置和firmware的升级，具体步骤如下：
    - 将计算棒插入Window7/10主机，参考[主动模式开发](http://t.rock-chips.com/wiki.php?mod=view&id=66)配主机的虚拟网卡IP地址，通过ssh toybrick@192.168.180.8验证是否能登录计算棒；
    - 参考[Window7/10系统配置计算棒网络共享](http://t.rock-chips.com/wiki.php?mod=view&id=77)，SSH登录计算棒后通过wget www.baidu.com验证是否能够访问外网；
    - 参考[固件在线升级](http://t.rock-chips.com/wiki.php?mod=view&id=148)，建议通过ssh登录计算棒，在shell下执行sudo dnf update -y命令快速升级到最新版本系统（要求系统版本>=1.4.1-2），可通过rpm -qa | grep toybrick-server查询系统版本：

    ```shell
    $ rpm -qa | grep toybrick-server
    toybrick-server-1.4.1-2.rk1808.fc28.aarch64
    ```

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[编译环境准备](../source_compile/compile_env)中的Docker开发环境进行配置。

### 运行图像分类示例程序

- 下载示例程序[PaddleLite-linux-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/PaddleLite-linux-demo.tar.gz)，解压后清单如下：

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
          - mobilenet_v1_int8_224_for_rockchip_npu_fluid # Paddle fluid non-combined格式的、适用于Rockchip NPU的mobilenetv1全量化模型
          - mobilenet_v1_int8_224_for_cpu
            - model.nb # 已通过opt转好的、适合ARM CPU的mobilenetv1量化模型
          - mobilenet_v1_int8_224_for_rockchip_npu
            - model.nb # 已通过opt转好的、适合Rockchip NPU的mobilenetv1全量化模型
      - shell
        - CMakeLists.txt # 示例程序CMake脚本
        - build
          - image_classification_demo # 已编译好的示例程序
        - image_classification_demo.cc # 示例程序源码
        - convert_to_raw_image.py # 将测试图片保存为raw数据的python脚本
        - build.sh # 示例程序编译脚本
        - run_with_adb.sh # RK1808 EVB的示例程序运行脚本
        - run_with_ssh.sh # TB-RK1808S0 AI计算棒的示例程序运行脚本
    - libs
      - PaddleLite
        - arm64
          - include # PaddleLite头文件
          - lib
            - libGAL.so # RK DDK库
            - libOpenVX.so
            - libVSC.so
            - librknpu_ddk.so
            - libgomp.so.1 # gnuomp库
            - libpaddle_light_api_shared.so # 预编译PaddleLite库
        - armhf
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

  For RK1806 EVB
  $ ./run_with_adb.sh armhf

  For TB-RK1808S0 AI计算棒
  $ ./run_with_ssh.sh arm64

    (RK1808 EVB)
    warmup: 5 repeat: 10, average: 266.276001 ms, max: 266.576996 ms, min: 266.158997 ms
    results: 3
    Top0  tabby, tabby cat - 0.522023
    Top1  Egyptian cat - 0.395266
    Top2  tiger cat - 0.073605
    Preprocess time: 2.684000 ms
    Prediction time: 266.276001 ms
    Postprocess time: 0.456000 ms

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

  For RK1806 EVB
  $ ./run_with_adb.sh armhf

  For TB-RK1808S0 AI计算棒
  $ ./run_with_ssh.sh arm64

    (RK1808 EVB)
    warmup: 5 repeat: 10, average: 6.663300 ms, max: 6.705000 ms, min: 6.590000 ms
    results: 3
    Top0  Egyptian cat - 0.514779
    Top1  tabby, tabby cat - 0.421183
    Top2  tiger cat - 0.052648
    Preprocess time: 2.391000 ms
    Prediction time: 6.663300 ms
    Postprocess time: 0.470000 ms

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

- 通过Paddle Fluid训练，或X2Paddle转换得到MobileNetv1 foat32模型[mobilenet_v1_fp32_224_fluid](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)；
- 参考[模型量化-有校准数据训练后量化](../user_guides/post_quant_with_data)使用PaddleSlim对float32模型进行量化（注意：由于RK NPU只支持tensor-wise的全量化模型，在启动量化脚本时请注意相关参数的设置），最终得到全量化MobileNetV1模型[mobilenet_v1_int8_224_fluid](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)；
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

  ```shell
  For RK1808 EVB and TB-RK1808S0 AI计算棒
  tiny_publish
  $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_rockchip_npu=ON --rockchip_npu_sdk_root=./rknpu_ddk
  full_publish
  $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_rockchip_npu=ON --rockchip_npu_sdk_root=./rknpu_ddk full_publish

  For RK1806 EVB
  tiny_publish
  $ ./lite/tools/build_linux.sh --arch=armv7 --with_extra=ON --with_log=ON --with_rockchip_npu=ON --rockchip_npu_sdk_root=./rknpu_ddk
  full_publish
  $ ./lite/tools/build_linux.sh --arch=armv7 --with_extra=ON --with_log=ON --with_rockchip_npu=ON --rockchip_npu_sdk_root=./rknpu_ddk full_publish
  ```

- 将编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.rknpu/cxx/include替换PaddleLite-linux-demo/libs/PaddleLite/arm64/include目录；
- 将tiny_publish模式下编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.rknpu/cxx/lib/libpaddle_light_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/arm64/lib/libpaddle_light_api_shared.so文件；
- 将full_publish模式下编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.rknpu/cxx/lib/libpaddle_full_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/arm64/lib/libpaddle_full_api_shared.so文件；
- 将编译生成的build.lite.armlinux.armv7.gcc/inference_lite_lib.armlinux.armv7.rknpu/cxx/include替换PaddleLite-linux-demo/libs/PaddleLite/armhf/include目录；
- 将tiny_publish模式下编译生成的build.lite.armlinux.armv7.gcc/inference_lite_lib.armlinux.armv7.rknpu/cxx/lib/libpaddle_light_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/armhf/lib/libpaddle_light_api_shared.so文件；
- 将full_publish模式下编译生成的build.lite.armlinux.armv7.gcc/inference_lite_lib.armlinux.armv7.rknpu/cxx/lib/libpaddle_full_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/armhf/lib/libpaddle_full_api_shared.so文件。

## 其它说明

- RK研发同学正在持续增加用于适配Paddle算子bridge/converter，以便适配更多Paddle模型。
