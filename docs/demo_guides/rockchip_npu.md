# PaddleLite使用瑞芯微NPU预测部署

Paddle Lite已支持Rockchip NPU的预测部署。
其接入原理是与之前华为Kirin NPU类似，即加载并分析Paddle模型，首先将Paddle算子转成NNAdapter标准算子，其次再转换为Rockchip NPU组网API进行网络构建，在线生成并执行模型。

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

### 已支持（或部分支持）NNAdapter的Paddle算子
可以通过访问[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/bridges/paddle_use_bridges.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/bridges/paddle_use_bridges.h)获得最新的算子支持列表。

不经过NNAdapter标准算子转换，直接将Paddle算子转换成HiAI IR的方案可点击[链接](https://paddle-lite.readthedocs.io/zh/release-v2.9/demo_guides/rockchip_npu.html)。

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

- 下载PaddleLite通用示例程序[PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - images
            - tabby_cat.jpg # 测试图片
            - tabby_cat.raw # 经过convert_to_raw_image.py处理后的RGB Raw图像
          - labels
            - synset_words.txt # 1000分类label文件
          - models
            - resnet50_fp32_224 # Paddle non-combined格式的resnet50 float32模型
              - __model__ # Paddle fluid模型组网文件，可拖入https://lutzroeder.github.io/netron/进行可视化显示网络结构
              - bn2a_branch1_mean # Paddle fluid模型参数文件
              - bn2a_branch1_scale
              - subgraph_partition_config_file.txt # 自定义子图分割配置文件
              ...
        - shell
          - CMakeLists.txt # 示例程序CMake脚本
          - build.linux.amd64 # 已编译好的，适用于amd64
            - image_classification_demo # 已编译好的，适用于amd64的示例程序
          - build.linux.arm64 # 已编译好的，适用于arm64
            - image_classification_demo # 已编译好的，适用于arm64的示例程序
            ...
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序ssh运行脚本
          - run_with_adb.sh # 示例程序adb运行脚本
      - libs
        - PaddleLite
          - android
            - arm64-v8a
            - armeabi-v7a
          - linux
            - amd64
              - include # PaddleLite头文件
              - lib # PaddleLite库文件
                - huawei_ascend_npu # 华为昇腾NPU NNAdapter API运行时库和Driver HAL库
                	- libnnadapter.so # NNAdapter API运行时库
                	- libnnadapter_driver_huawei_ascend_npu.so # 华为昇腾NPU NNAdapter driver HAL库
                - libiomp5.so # Intel OpenMP库
                - libmklml_intel.so # Intel MKL库
                - libmklml_gnu.so # GNU MKL库
                - libpaddle_full_api_shared.so # 预编译PaddleLite full api库
                - libpaddle_light_api_shared.so # 预编译PaddleLite light api库
            - arm64
              - include # PaddleLite头文件
              - lib
            - armhf
            	...
        - OpenCV # OpenCV预编译库
      - ssd_detection_demo # 基于ssd的目标检测示例程序
  ```

- Android shell端示例程序用法（以RK1808 EVB为例）：  

  编译示例程序（image_classification_demo）
  ```shell
  ./build.sh android arm64
  ```
  运行示例程序（image_classification_demo）
  ```shell
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu a133d8abb26137b2

- 按照以下命令分别运行转换后的ARM CPU模型和Rockchip NPU模型，比较它们的性能和结果；

  ```shell
  注意：
  1）run_with_adb.sh不能在docker环境执行，否则可能无法找到设备，也不能在设备上运行；
  2）run_with_ssh.sh不能在设备上运行，且执行前需要配置目标设备的IP地址、SSH账号和密码。
  3）build.sh与run_with_adb.sh、run_with_ssh.sh根据入参生成针对不同操作系统、体系结构的二进制程序并运行，需查阅注释信息配置正确的参数值。

  运行适用于ARM CPU的mobilenet_v1_int8_224_per_layer全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell

  For RK1808 EVB
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu a133d8abb26137b2
    (RK1808 EVB)
    warmup: 1 repeat: 5, average: 266.965796 ms, max: 267.056000 ms, min: 266.848999 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 2.423000 ms
    Prediction time: 266.965796 ms
    Postprocess time: 0.538000 ms

  For RK1806/RV1126/RV1109 EVB
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux armhf cpu 192.168.100.13 22 root rockchip
    (RV1109 EVB)
    warmup: 1 repeat: 5, average: 331.796204 ms, max: 341.756012 ms, min: 328.386993 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.380000 ms
    Prediction time: 331.796204 ms
    Postprocess time: 0.554000 ms

  For TB-RK1808S0 AI计算棒
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64
    (TB-RK1808S0 AI计算棒)
    warmup: 1 repeat: 5, average: 357.467200 ms, max: 358.815002 ms, min: 356.808014 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.199000 ms
    Prediction time: 357.467200 ms
    Postprocess time: 0.596000 ms

  运行适用于Rockchip NPU的mobilenet_v1_int8_224_per_layer全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell

  For RK1808 EVB
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 rockchip_npu a133d8abb26137b2
    (RK1808 EVB)
    warmup: 1 repeat: 5, average: 6.982800 ms, max: 7.045000 ms, min: 6.951000 ms
    results: 3
    Top0  Egyptian cat - 0.514779
    Top1  tabby, tabby cat - 0.421183
    Top2  tiger cat - 0.052648
    Preprocess time: 2.417000 ms
    Prediction time: 6.982800 ms
    Postprocess time: 0.509000 ms

  For RK1806/RV1126/RV1109 EVB
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux armhf rockchip_npu 192.168.100.13 22 root rockchip
    (RV1109 EVB)
    warmup: 1 repeat: 5, average: 7.494000 ms, max: 7.724000 ms, min: 7.321000 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 3.532000 ms
    Prediction time: 7.494000 ms
    Postprocess time: 0.577000 ms

  For TB-RK1808S0 AI计算棒
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 rockchip_npu
    (TB-RK1808S0 AI计算棒)
    warmup: 1 repeat: 5, average: 9.330400 ms, max: 9.753000 ms, min: 8.421000 ms
    results: 3
    Top0  Egyptian cat - 0.514779
    Top1  tabby, tabby cat - 0.421183
    Top2  tiger cat - 0.052648
    Preprocess time: 3.170000 ms
    Prediction time: 9.330400 ms
    Postprocess time: 0.634000 ms
  ```

- 如果需要更改测试图片，可将图片拷贝到PaddleLite-generic-demo/image_classification_demo/assets/images目录下，然后调用convert_to_raw_image.py生成相应的RGB Raw图像，最后修改run_with_adb.sh、run_with_ssh.sh的IMAGE_NAME变量即可；

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
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_rockchip_npu=ON --nnadapter_rockchip_npu_sdk_root=$(pwd)/rknpu_ddk

      ```
    - full_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_rockchip_npu=ON --nnadapter_rockchip_npu_sdk_root=$(pwd)/rknpu_ddk full_publish

      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录：
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter相关so：
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter* PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/rockchip_npu/
      # 替换 libpaddle_full_api_shared.so或libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

  - For RK1806/RV1126/RV1109 EVB
    - tiny_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_rockchip_npu=ON --nnadapter_rockchip_npu_sdk_root=$(pwd)/rknpu_ddk
      ```

    - full_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_rockchip_npu=ON --nnadapter_rockchip_npu_sdk_root=$(pwd)/rknpu_ddk full_publish
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录：
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      # 替换 NNAdapter相关so：
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter* PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/rockchip_npu/
      # 替换 libpaddle_full_api_shared.so或libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      ```
  
- 替换头文件后需要重新编译示例程序

## 其它说明

- RK研发同学正在持续增加用于适配Paddle算子bridge/converter，以便适配更多Paddle模型。
