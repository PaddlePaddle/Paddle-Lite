# PaddleLite使用RK NPU预测部署

Paddle Lite已支持RK NPU的预测部署。
其接入原理是与之前华为NPU类似，即加载并分析Paddle模型，将Paddle算子转成RK组网API进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- RK1808, RK1806，暂时不支持RK3399Pro。

### 已支持的设备

- RK1808/1806 EVB。

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

### 测试设备(RK1808 EVB)

![rk1808_evb_front](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rk1808_evb_front.jpg)

![rk1808_evb_back](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rk1808_evb_back.jpg)

### 准备设备环境

- 需要依赖特定版本的firmware，请参照[rknpu_ddk](https://github.com/airockchip/rknpu_ddk)的说明对设备进行firmware的更新；
- 由于RK1808 EVB在刷firmware后，只是一个纯净的Linux系统，无法像Ubuntu那样使用apt-get命令方便的安装软件，因此，示例程序和PaddleLite库的编译均采用交叉编译方式；
- 将MicroUSB线插入到设备的MicroUSB OTG口，就可以使用Android的adb命令进行设备的交互，再也不用配置网络使用ssh或者通过串口的方式访问设备了，这个设计非常赞！

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[源码编译](../user_guides/source_compile)中的Docker开发环境进行配置。

### 运行图像分类示例程序

- 从[https://paddlelite-demo.bj.bcebos.com/devices/rockchip/PaddleLite-armlinux-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/PaddleLite-armlinux-demo.tar.gz)下载示例程序，解压后清单如下：

```shell
- PaddleLite-armlinux-demo
  - image_classification_demo
    - assets
      - images 
        - tabby_cat.jpg # 测试图片
        - tabby_cat.raw # 已处理成raw数据的测试图片
      - labels
        - synset_words.txt # 1000分类label文件
      - models
        - mobilenet_v1_int8_224_for_cpu.nb # 已通过opt转好的、适合arm cpu的mobilenetv1量化模型
        - mobilenet_v1_int8_224_for_rknpu.nb # 已通过opt转好的、适合rknpu的mobilenetv1量化模型
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
          - libGAL.so # RK DDK库
          - libOpenVX.so
          - libVSC.so
          - librknpu_ddk.so
          - libgomp.so.1 # gnuomp库
          - libpaddle_light_api_shared.so # 预编译PaddleLite库
      - armhf
        - include # PaddleLite头文件
        - lib
          - libGAL.so
          - libOpenVX.so
          - libVSC.so
          - librknpu_ddk.so
          - libgomp.so.1
          - libpaddle_light_api_shared.so
```

- 进入PaddleLite-armlinux-demo/image_classification_demo/shell，直接执行./run.sh arm64即可，注意：run.sh不能在docker环境执行，否则无法找到设备；
```shell
$ cd PaddleLite-armlinux-demo/image_classification_demo/shell
$ ./run.sh arm64 # For RK1808 EVB
$ ./run.sh armhf # For RK1806 EVB 
...
warmup: 5 repeat: 10, average: 6.499500 ms, max: 6.554000 ms, min: 6.468000 ms
results: 3
Top0  Egyptian cat - 0.532328
Top1  tabby, tabby cat - 0.345136
Top2  tiger cat - 0.111146
Preprocess time: 2.414000 ms
Prediction time: 6.499500 ms
Postprocess time: 0.414000 ms
```
- 如果需要更改测试图片，可通过convert_to_raw_image.py工具生成；
- 如果需要重新编译示例程序，直接运行./build.sh即可，注意：build.sh的执行必须在docker环境中，否则可能编译出错。


### 更新模型

- 通过Paddle Fluid训练，或X2Paddle转换得到MobileNetv1 foat32模型[mobilenet_v1_fp32_224_fluid](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)；
- 参考[模型量化-有校准数据训练后量化](../user_guides/post_quant_with_data)使用PaddleSlim对float32模型进行量化（注意：由于RK NPU只支持tensor-wise的全量化模型，在启动量化脚本时请注意相关参数的设置），最终得到全量化MobileNetV1模型[mobilenet_v1_int8_224_fluid](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)；
- 参考[模型转化方法](../user_guides/model_optimize_tool)，利用opt工具转换生成RKNPU模型，仅需要将valid_targets设置为rknpu,arm即可。
```shell
$ ./opt --model_dir=mobilenet_v1_int8_224_fluid \
    --optimize_out_type=naive_buffer \
    --optimize_out=mobilenet_v1_int8_224_for_rknpu \
    --valid_targets=rknpu,arm
```
- 注意：opt生成的模型只是标记了RKNPU支持的Paddle算子，并没有真正生成RK NPU模型，只有在执行时才会将标记的Paddle算子转成RK NPU组网API，最终生成并执行模型。

### 更新支持RK NPU的Paddle Lite库

- 下载PaddleLite源码和RK DDK；
```shell
$ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
$ cd Paddle-Lite
$ git checkout <release-version-tag>
$ git clone https://github.com/airockchip/rknpu_ddk.git
```
- 编译full_publish and tiny_publish for RK1808 and RK1806 EVB
```shell
For RK1808 EVB
$ ./lite/tools/build.sh --arm_os=armlinux --arm_abi=armv8 --arm_lang=gcc --build_extra=ON --with_log=ON --build_rknpu=ON --rknpu_ddk_root=./rknpu_ddk full_publish
$ ./lite/tools/build.sh --arm_os=armlinux --arm_abi=armv8 --arm_lang=gcc --build_extra=ON --with_log=ON --build_rknpu=ON --rknpu_ddk_root=./rknpu_ddk tiny_publish

For RK1806 EVB
$ ./lite/tools/build.sh --arm_os=armlinux --arm_abi=armv7 --arm_lang=gcc --build_extra=ON --with_log=ON --build_rknpu=ON --rknpu_ddk_root=./rknpu_ddk full_publish
$ ./lite/tools/build.sh --arm_os=armlinux --arm_abi=armv7 --arm_lang=gcc --build_extra=ON --with_log=ON --build_rknpu=ON --rknpu_ddk_root=./rknpu_ddk tiny_publish
```
- 将编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.rknpu/cxx/include替换PaddleLite-armlinux-demo/libs/PaddleLite/arm64/include目录；
- 将编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.rknpu/cxx/lib/libpaddle_light_api_shared.so替换PaddleLite-armlinux-demo/libs/PaddleLite/arm64/lib/libpaddle_light_api_shared.so文件；
- 将编译生成的build.lite.armlinux.armv7.gcc/inference_lite_lib.armlinux.armv7.rknpu/cxx/include替换PaddleLite-armlinux-demo/libs/PaddleLite/armhf/include目录；
- 将编译生成的build.lite.armlinux.armv7.gcc/inference_lite_lib.armlinux.armv7.rknpu/cxx/lib/libpaddle_light_api_shared.so替换PaddleLite-armlinux-demo/libs/PaddleLite/armhf/lib/libpaddle_light_api_shared.so文件。

## 其它说明

- RK研发同学正在持续增加用于适配Paddle算子bridge/converter，以便适配更多Paddle模型。
