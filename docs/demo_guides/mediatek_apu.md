# PaddleLite使用MTK APU预测部署

Paddle Lite已支持MTK APU的预测部署。
其接入原理是与之前华为NPU类似，即加载并分析Paddle模型，将Paddle算子转成MTK的Neuron adapter API（类似Android NN API）进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- [MT8168](https://www.mediatek.cn/products/tablets/mt8168)/[MT8175](https://www.mediatek.cn/products/tablets/mt8175)及其他智能芯片。

### 已支持的设备

- MT8168-P2V1 Tablet。

### 已支持的Paddle模型

- [全量化MobileNetV1](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/mobilenet_v1_int8_224_fluid.tar.gz)

### 已支持（或部分支持）的Paddle算子

- relu
- conv2d
- depthwise_conv2d
- elementwise_add
- elementwise_mul
- fc
- pool2d
- softmax

## 参考示例演示

### 测试设备(MT8168-P2V1 Tablet)

![mt8168_p2v1_tablet_front](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/mt8168_p2v1_tablet_front.jpg)

![mt8168_p2v1_tablet_back](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/mt8168_p2v1_tablet_back.jpg)

### 准备设备环境

- 由于需要依赖特定版本的firmware，感兴趣的同学通过MTK官网[https://www.mediatek.cn/about/contact-us](https://www.mediatek.cn/about/contact-us)提供的联系方式（类别请选择"销售"），获取测试设备和firmware；

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[源码编译](../user_guides/source_compile)中的Docker开发环境进行配置。

### 运行图像分类示例程序

- 从[https://paddlelite-demo.bj.bcebos.com/devices/mediatek/PaddleLite-android-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/PaddleLite-android-demo.tar.gz)下载示例程序，解压后清单如下：

```shell
- PaddleLite-android-demo
  - image_classification_demo
    - assets
      - images 
        - tabby_cat.jpg # 测试图片
      - labels
        - synset_words.txt # 1000分类label文件
      - models
        - mobilenet_v1_int8_224_for_cpu.nb # 已通过opt转好的、适合arm cpu的mobilenetv1量化模型
        - mobilenet_v1_int8_224_for_apu.nb # 已通过opt转好的、适合mtk apu的mobilenetv1量化模型
    - shell # android shell端的示例程序
      - CMakeLists.txt # 示例程序CMake脚本
      - build
        - image_classification_demo # 已编译好的android shell端的示例程序
      - image_classification_demo.cc # 示例程序源码
      - build.sh # 示例程序编译脚本
      - run.sh # 示例程序运行脚本
    - apk # 常规android应用程序
      - app
        - src
          - main
            - java # java层代码
            - cpp # 自定义的jni实现
        - app.iml
        - build.gradle
      - gradle
      ...
  - libs
    - PaddleLite
      - arm64-v8a
        - include # PaddleLite头文件
        - lib
          - libc++_shared.so
          - libpaddle_light_api_shared.so # 预编译PaddleLite库
    - OpenCV # OpenCV 4.2 for android
```

- Android shell端的示例程序
  - 进入PaddleLite-android-demo/image_classification_demo/shell，直接执行./run.sh即可，注意：run.sh不能在docker环境执行，否则可能无法找到设备；
  - 如果需要更改测试图片，可将图片拷贝到PaddleLite-android-demo/image_classification_demo/assets/images目录下，然后将run.sh的IMAGE_NAME设置成指定文件名即可；
  - 如果需要重新编译示例程序，直接运行./build.sh即可，注意：build.sh的执行必须在docker环境中，否则可能编译出错；
  - 需要说明的是，由于MTK APU暂时只支持NHWC的数据布局格式，而PaddleLite默认使用NCHW的数据布局格式，导致额外增加了预测中输入张量的NCHW到NHWC的转换，大约耗费8~9ms。
```shell
$ cd PaddleLite-android-demo/image_classification_demo/shell
$ ./run.sh
...
warmup: 5 repeat: 10, average: 30.998502 ms, max: 31.049002 ms, min: 30.937002 ms
results: 3
Top0  Egyptian cat - -0.122845
Top1  tabby, tabby cat - -0.122845
Top2  tiger cat - -0.544028
Preprocess time: 3.620000 ms
Prediction time: 30.998502 ms
Postprocess time: 0.069000 ms

[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1b00000, pa = 0xfb3f9000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1af8000, pa = 0xfb3fa000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1af7000, pa = 0xf8ffe000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1af6000, pa = 0xf7bfe000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1af5000, pa = 0xf7bfd000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1b0c000, pa = 0xfb3fe000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1b0b000, pa = 0xfb3ff000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1b0a000, pa = 0xf31ff000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1b09000, pa = 0xfb3f6000, len = 255
[vpuBuffer] vpuMemAllocator::freeMem: type = 1, va = 0x7ed1b08000, pa = 0xf7bff000, len = 255
```

- 常规Android应用程序
  - 安装Android Studio 3.4
  - 打开Android Studio，在"Welcome to Android Studio"窗口点击"Open an existing Android Studio project"，在弹出的路径选择窗口中进入"PaddleLite-android-demo/image_classification_demo/apk"目录，然后点击右下角的"Open"按钮即可导入工程；
  - 通过USB连接Android手机、平板或开发板；
  - 临时关闭selinux模式，允许app调用系统库；
```shell
$ adb root
# setenforce 0
```
  - 待工程加载完成后，点击菜单栏的Build->Rebuild Project按钮，如果提示CMake版本不匹配，请点击错误提示中的'Install CMake xxx.xxx.xx'按钮，重新安装CMake，然后再次点击菜单栏的Build->Rebuild Project按钮；
  - 待工程编译完成后，点击菜单栏的Run->Run 'App'按钮，在弹出的"Select Deployment Target"窗口选择已经连接的Android设备，然后点击"OK"按钮；
  - 等待大约1分钟后（第一次时间比较长，需要耐心等待），app已经安装到设备上。默认使用ARM CPU模型进行预测，由于MT8168的CPU由四核Arm-Cortex A53组成，性能较一般手机的A7x系列要弱很多，如下图所示，只有6fps；

![mt8168_p2v1_tablet_cpu](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/mt8168_p2v1_tablet_cpu.jpg)

  - 点击app界面右下角的设置按钮，在弹出的设置页面点击"Choose pre-installed models"，选择"mobilenet_v1_int8_for_apu"，点击返回按钮后，app将切换到APU模型，如下图所示，帧率提高到14fps。

![mt8168_p2v1_tablet_apu](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/mt8168_p2v1_tablet_apu.jpg)


### 更新模型

- 通过Paddle Fluid训练，或X2Paddle转换得到MobileNetv1 foat32模型[mobilenet_v1_fp32_224_fluid](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)；
- 参考[模型量化-有校准数据训练后量化](../user_guides/post_quant_with_data)使用PaddleSlim对float32模型进行量化（注意：由于MTK APU只支持量化OP，在启动量化脚本时请注意相关参数的设置），最终得到全量化MobileNetV1模型[mobilenet_v1_int8_224_fluid](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/mobilenet_v1_int8_224_fluid.tar.gz)；
- 参考[模型转化方法](../user_guides/model_optimize_tool)，利用opt工具转换生成MTK APU模型，仅需要将valid_targets设置为apu,arm即可。
```shell
$ ./opt --model_dir=mobilenet_v1_int8_224_fluid \
    --optimize_out_type=naive_buffer \
    --optimize_out=mobilenet_v1_int8_224_for_apu \
    --valid_targets=apu,arm
```
- 注意：opt生成的模型只是标记了MTK APU支持的Paddle算子，并没有真正生成MTK APU模型，只有在执行时才会将标记的Paddle算子转成MTK Neuron adapter API调用实现组网，最终生成并执行模型。

### 更新支持MTK APU的Paddle Lite库

- 下载PaddleLite源码和APU DDK；
```shell
$ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
$ cd Paddle-Lite
$ git checkout <release-version-tag>
$ wget https://paddlelite-demo.bj.bcebos.com/devices/mediatek/apu_ddk.tar.gz
$ tar -xvf apu_ddk.tar.gz
```
- 编译tiny_publish for MT8168-P2V1 Tablet
```shell
$ ./lite/tools/build.sh --arm_os=android --arm_abi=armv8 --arm_lang=gcc --android_stl=c++_shared --build_extra=ON --with_log=ON --build_apu=ON --apu_ddk_root=./apu_ddk tiny_publish
```
- 将编译生成的build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.apu/cxx/include替换PaddleLite-android-demo/libs/PaddleLite/arm64-v8a/include目录；
- 将编译生成的build.lite.android.armv8.gcc/inference_lite_lib.android.armv8.apu/cxx/lib/libpaddle_light_api_shared.so替换PaddleLite-android-demo/libs/PaddleLite/arm64-v8a/lib/libpaddle_light_api_shared.so文件。


## 其它说明

- 由于涉及到License的问题，无法提供用于测试的firmware，我们深感抱歉。如果确实对此非常感兴趣，可以参照之前提到的联系方式，直接联系MTK的销售；
- MTK研发同学正在持续增加用于适配Paddle算子bridge/converter，以便适配更多Paddle模型。
