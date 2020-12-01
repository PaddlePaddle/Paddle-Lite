# PaddleLite使用百度XPU预测部署

Paddle Lite已支持百度XPU在x86和arm服务器（例如飞腾 FT-2000+/64）上进行预测部署。
目前支持Kernel和子图两种接入方式，其中子图接入方式与之前华为Kirin NPU类似，即加载并分析Paddle模型，将Paddle算子转成XTCL组网API进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- 昆仑818-100（推理芯片）
- 昆仑818-300（训练芯片）

### 已支持的设备

- K100/K200昆仑AI加速卡

### 已支持的Paddle模型

- [开源模型支持列表](../introduction/support_model_list)
- 百度内部业务模型（由于涉密，不方便透露具体细节）

### 已支持（或部分支持）的Paddle算子（Kernel接入方式）

- [算子支持列表](../introduction/support_operation_list)


## 参考示例演示

### 测试设备(K100昆仑AI加速卡)

![baidu_xpu](https://paddlelite-demo.bj.bcebos.com/devices/baidu/baidu_xpu.jpg)

### 准备设备环境

- K100/200昆仑AI加速卡[规格说明书](https://paddlelite-demo.bj.bcebos.com/devices/baidu/K100_K200_spec.pdf)，如需更详细的规格说明书或购买产品，请联系欧阳剑ouyangjian@baidu.com；
- K100为全长半高PCI-E卡，K200为全长全高PCI-E卡，要求使用PCI-E x16插槽，且需要单独的8针供电线进行供电；
- 安装K100/K200驱动，目前支持Ubuntu和CentOS系统，由于驱动依赖Linux kernel版本，请正确安装对应版本的驱动安装包。

### 准备本地编译环境

- 为了保证编译环境一致，建议参考[编译环境准备](../source_compile/compile_env.html#linux)中的Linux开发环境进行配置；
- 由于编译示例程序需要依赖OpenCV和CMake 3.10.3，请执行如下命令进行安装；

  ```shell
  $ sudo apt-get update
  $ sudo apt-get install gcc g++ make wget unzip libopencv-dev pkg-config
  $ wget https://www.cmake.org/files/v3.10/cmake-3.10.3.tar.gz
  $ tar -zxvf cmake-3.10.3.tar.gz
  $ cd cmake-3.10.3
  $ ./configure
  $ make
  $ sudo make install
  ```

### 运行图像分类示例程序

- 下载示例程序[PaddleLite-linux-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/baidu/PaddleLite-linux-demo.tar.gz)，解压后清单如下：

  ```shell
  - PaddleLite-linux-demo
    - image_classification_demo
      - assets
        - images
          - tabby_cat.jpg # 测试图片
          - tabby_cat.raw # 经过convert_to_raw_image.py处理后的RGB Raw图像
        - labels
          - synset_words.txt # 1000分类label文件
        - models
          - resnet50_fp32_224_fluid # Paddle fluid non-combined格式的resnet50 float32模型
            - __model__ # Paddle fluid模型组网文件，可拖入https://lutzroeder.github.io/netron/进行可视化显示网络结构
            - bn2a_branch1_mean # Paddle fluid模型参数文件
            - bn2a_branch1_scale
            ...
      - shell
        - CMakeLists.txt # 示例程序CMake脚本
        - build
          - image_classification_demo # 已编译好的，适用于amd64的示例程序
        - image_classification_demo.cc # 示例程序源码
        - build.sh # 示例程序编译脚本
        - run.sh # 示例程序运行脚本
    - libs
      - PaddleLite
        - amd64
          - include # PaddleLite头文件
          - lib
            - libiomp5.so # Intel OpenMP库
            - libmklml_intel.so # Intel MKL库
            - libxpuapi.so # XPU API库，提供设备管理和算子实现。
            - llibxpurt.so # XPU runtime库
            - libpaddle_full_api_shared.so # 预编译PaddleLite full api库
        - arm64
          - include # PaddleLite头文件
          - lib
            - libxpuapi.so # XPU API库，提供设备管理和算子实现。
            - llibxpurt.so # XPU runtime库
            - libpaddle_full_api_shared.so # 预编译PaddleLite full api库
  ```

- 进入PaddleLite-linux-demo/image_classification_demo/shell，直接执行./run.sh amd64即可；

  ```shell
  $ cd PaddleLite-linux-demo/image_classification_demo/shell
  $ ./run.sh amd64 # 默认已生成amd64版本的build/image_classification_demo，因此，无需重新编译示例程序就可以执行。
  $ ./run.sh arm64 # 需要在arm64(FT-2000+/64)服务器上执行./build.sh arm64后才能执行该命令。
  ...
  AUTOTUNE:(12758016, 16, 1, 2048, 7, 7, 512, 1, 1, 1, 1, 0, 0, 0) = 1by1_bsp(1, 32, 128, 128)
  Find Best Result in 150 choices, avg-conv-op-time = 40 us
  [INFO][XPUAPI][/home/qa_work/xpu_workspace/xpu_build_dailyjob/api_root/baidu/xpu/api/src/wrapper/conv.cpp:274] Start Tuning: (12758016, 16, 1, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 0)
  AUTOTUNE:(12758016, 16, 1, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 0) = wpinned_bsp(1, 171, 16, 128)
  Find Best Result in 144 choices, avg-conv-op-time = 79 us
  I0502 22:34:18.176113 15876 io_copy_compute.cc:75] xpu to host, copy size 4000
  I0502 22:34:18.176406 15876 io_copy_compute.cc:36] host to xpu, copy size 602112
  I0502 22:34:18.176697 15876 io_copy_compute.cc:75] xpu to host, copy size 4000
  iter 0 cost: 2.116000 ms
  I0502 22:34:18.178530 15876 io_copy_compute.cc:36] host to xpu, copy size 602112
  I0502 22:34:18.178792 15876 io_copy_compute.cc:75] xpu to host, copy size 4000
  iter 1 cost: 2.101000 ms
  I0502 22:34:18.180634 15876 io_copy_compute.cc:36] host to xpu, copy size 602112
  I0502 22:34:18.180881 15876 io_copy_compute.cc:75] xpu to host, copy size 4000
  iter 2 cost: 2.089000 ms
  I0502 22:34:18.182726 15876 io_copy_compute.cc:36] host to xpu, copy size 602112
  I0502 22:34:18.182976 15876 io_copy_compute.cc:75] xpu to host, copy size 4000
  iter 3 cost: 2.085000 ms
  I0502 22:34:18.184814 15876 io_copy_compute.cc:36] host to xpu, copy size 602112
  I0502 22:34:18.185068 15876 io_copy_compute.cc:75] xpu to host, copy size 4000
  iter 4 cost: 2.101000 ms
  warmup: 1 repeat: 5, average: 2.098400 ms, max: 2.116000 ms, min: 2.085000 ms
  results: 3
  Top0  tabby, tabby cat - 0.689418
  Top1  tiger cat - 0.190557
  Top2  Egyptian cat - 0.112354
  Preprocess time: 1.553000 ms
  Prediction time: 2.098400 ms
  Postprocess time: 0.081000 ms

  ```

- 如果需要更改测试图片，请将图片拷贝到PaddleLite-linux-demo/image_classification_demo/assets/images目录下，修改并执行convert_to_raw_image.py生成相应的RGB Raw图像，最后修改run.sh的IMAGE_NAME即可；
- 如果需要重新编译示例程序，直接运行./build.sh amd64或./build.sh arm64即可。

  ```shell
  $ cd PaddleLite-linux-demo/image_classification_demo/shell
  $ ./build.sh amd64 # For amd64
  $ ./build.sh arm64 # For arm64(FT-2000+/64) 
  ```

### 更新模型

- 通过Paddle Fluid训练，或X2Paddle转换得到ResNet50 float32模型[resnet50_fp32_224_fluid](https://paddlelite-demo.bj.bcebos.com/models/resnet50_fp32_224_fluid.tar.gz)；
- 由于XPU一般部署在Server端，因此将使用PaddleLite的full api加载原始的Paddle Fluid模型进行预测，即采用CXXConfig配置相关参数。

### 更新支持百度XPU的Paddle Lite库

- 下载PaddleLite源码；

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 下载xpu_toolchain for amd64 or arm64(FT-2000+/64)；

  ```shell
  $ wget <URL_to_download_xpu_toolchain>
  $ tar -xvf output.tar.gz
  $ mv output xpu_toolchain
  ```

- 编译full_publish for amd64 or arm64(FT-2000+/64)；

  ```shell
  For amd64，如果报找不到cxx11::符号的编译错误，请将gcc切换到4.8版本。
  $ ./lite/tools/build.sh --build_xpu=ON --xpu_sdk_root=./xpu_toolchain --build_extra=ON x86

  For arm64(FT-2000+/64)
  $ ./lite/tools/build.sh --arm_os=armlinux --arm_abi=armv8 --arm_lang=gcc --build_extra=ON --build_xpu=ON --xpu_sdk_root=./xpu_toolchain --with_log=ON full_publish
  ```

- 将编译生成的build.lite.x86/inference_lite_lib/cxx/include替换PaddleLite-linux-demo/libs/PaddleLite/amd64/include目录；
- 将编译生成的build.lite.x86/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/amd64/lib/libpaddle_full_api_shared.so文件；
- 将编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.xpu/cxx/include替换PaddleLite-linux-demo/libs/PaddleLite/arm64/include目录；
- 将编译生成的build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8.xpu/cxx/lib/libpaddle_full_api_shared.so替换PaddleLite-linux-demo/libs/PaddleLite/arm64/lib/libpaddle_full_api_shared.so文件。

## 其它说明

- 如需更进一步的了解相关产品的信息，请联系欧阳剑ouyangjian@baidu.com；
- 百度昆仑的研发同学正在持续适配更多的Paddle算子，以便支持更多的Paddle模型。
