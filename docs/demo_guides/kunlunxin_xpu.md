# 昆仑芯 XPU 部署示例

Paddle Lite 已支持昆仑芯 XPU 在 X86 和 ARM 服务器（例如飞腾 FT-2000+/64）上进行预测部署。
目前支持 Kernel 和子图两种接入方式。其中子图接入方式原理是，在线分析 Paddle 模型，将 Paddle 算子先转为统一的 NNAdapter 标准算子，再通过 XTCL 组网 API 进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- 昆仑 818-100（推理芯片）
- 昆仑 818-300（训练芯片）

### 已支持的设备

- K100/K200 昆仑 AI 加速卡

### 已支持的 Paddle 模型

- [开源模型支持列表](../quick_start/support_model_list)
- 百度内部业务模型（由于涉密，不方便透露具体细节）

### 已支持（或部分支持）的 Paddle 算子（ Kernel 接入方式）

- [算子支持列表](../quick_start/support_operation_list)


## 参考示例演示

### 测试设备( K100 昆仑 AI 加速卡)

![kunlunxin_xpu](https://paddlelite-demo.bj.bcebos.com/devices/baidu/baidu_xpu.jpg)

### 准备设备环境

- K100/200 昆仑 AI 加速卡[规格说明书](https://paddlelite-demo.bj.bcebos.com/devices/baidu/K100_K200_spec.pdf)，如需更详细的规格说明书或购买产品，请联系欧阳剑ouyangjian@baidu.com；
- K100 为全长半高 PCI-E 卡，K200 为全长全高 PCI-E 卡，要求使用 PCI-E x16 插槽，且需要单独的 8 针供电线进行供电；
- 安装 K100/K200 驱动，目前支持 Ubuntu 和 CentOS 系统，由于驱动依赖 Linux kernel 版本，请正确安装对应版本的驱动安装包。

### 准备编译环境

- 为了保证编译环境一致，建议根据机器的实际情况参考[ linux(x86) 编译](../source_compile/linux_x86_compile_linux_x86)或[ linux(ARM) 编译](../source_compile/arm_linux_compile_arm_linux)中的``准备编译环境``进行环境配置

### 运行图像分类示例程序

- 下载示例程序[ PaddleLite-linux-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/baidu/PaddleLite-linux-demo.tar.gz)，解压后清单如下：

  ```shell
  - PaddleLite-linux-demo
    - image_classification_demo
      - assets
        - images
          - tabby_cat.jpg # 测试图片
          - tabby_cat.raw # 经过 convert_to_raw_image.py 处理后的 RGB Raw 图像
        - labels
          - synset_words.txt # 1000 分类 label 文件
        - models
          - resnet50_fp32_224_fluid # Paddle fluid non-combined 格式的 resnet50 float32 模型
            - __model__ # Paddle fluid 模型组网文件，可拖入 https://lutzroeder.github.io/netron/ 进行可视化显示网络结构
            - bn2a_branch1_mean # Paddle fluid 模型参数文件
            - bn2a_branch1_scale
            ...
      - shell
        - CMakeLists.txt # 示例程序 CMake 脚本
        - build
          - image_classification_demo # 已编译好的，适用于 amd64 的示例程序
        - image_classification_demo.cc # 示例程序源码
        - build.sh # 示例程序编译脚本
        - run.sh # 示例程序运行脚本
    - libs
      - PaddleLite
        - amd64
          - include # Paddle Lite 头文件
          - lib
            - libiomp5.so # Intel OpenMP 库
            - libmklml_intel.so # Intel MKL 库
            - libxpuapi.so # XPU API 库，提供设备管理和算子实现。
            - llibxpurt.so # XPU runtime 库
            - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
        - arm64
          - include # Paddle Lite 头文件
          - lib
            - libxpuapi.so # XPU API 库，提供设备管理和算子实现。
            - llibxpurt.so # XPU runtime 库
            - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
  ```

- 进入 `PaddleLite-linux-demo/image_classification_demo/shell`，直接执行 `./run.sh amd64` 即可；

  ```shell
  $ cd PaddleLite-linux-demo/image_classification_demo/shell
  $ ./run.sh amd64 # 默认已生成 amd64 版本的 build/image_classification_demo，因此，无需重新编译示例程序就可以执行。
  $ ./run.sh arm64 # 需要在 arm64(FT-2000+/64) 服务器上执行 ./build.sh arm64 后才能执行该命令。
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

- 如果需要更改测试图片，请将图片拷贝到 `PaddleLite-linux-demo/image_classification_demo/assets/images` 目录下，修改并执行 `convert_to_raw_image.py` 生成相应的 RGB Raw 图像，最后修改 `run.sh` 的 IMAGE_NAME 即可；
- 如果需要重新编译示例程序，直接运行 `./build.sh amd64` 或 `./build.sh arm64` 即可。

  ```shell
  $ cd PaddleLite-linux-demo/image_classification_demo/shell
  $ ./build.sh amd64 # For amd64
  $ ./build.sh arm64 # For arm64(FT-2000+/64) 
  ```

### 更新模型

- 通过 Paddle 训练，或 X2Paddle 转换得到 ResNet50 float32 模型[ resnet50_fp32_224_fluid ](https://paddlelite-demo.bj.bcebos.com/models/resnet50_fp32_224_fluid.tar.gz)；
- 由于 XPU 一般部署在 Server 端，因此将使用 Paddle Lite 的 `full api` 加载原始的 Paddle 模型进行预测，即采用 `CxxConfig` 配置相关参数。

### 更新支持昆仑芯 XPU 的 Paddle Lite 库

- 下载 Paddle Lite 源码；

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译 publish so for amd64 or arm64(FT-2000+/64)；

  ```shell
  $ # For amd64，如果报找不到 cxx11:: 符号的编译错误，请将 gcc 切换到 4.8 版本。
  $ ./lite/tools/build_linux.sh --arch=x86 --with_kunlunxin_xpu=ON

  $ # For arm64(FT-2000+/64), arm 环境下需要设置环境变量 CC 和 CXX，分别指定 C 编译器和 C++ 编译器的路径
  $ export CC=<path_to_your_c_compiler>
  $ export CXX=<path_to_your_c++_compiler>
  $ ./lite/tools/build_linux.sh --arch=armv8 --with_kunlunxin_xpu=ON
  ```

- 替换库文件和头文件

  ```shell
  cd PaddleLite-linux-demo/image_classification_demo/shell
  ./update_libs.sh <lite_inference_dir> <demo_libs_dir>
  # For amd64，lite_inference_dir 一般为编译生成的 build.lite.linux.x86.gcc.kunlunxin_xpu/inference_lite_lib，demo_libs_dir 为 PaddleLite-linux-demo/libs/PaddleLite/amd64
  # For arm64，lite_inference_dir一般为编译生成的 build.lite.linux.armv8.gcc.kunlunxin_xpu/inference_lite_lib.armlinux.armv8.xpu，demo_libs_dir 为 PaddleLite-linux-demo/libs/PaddleLite/amd64
  ```

  备注：替换头文件后需要重新编译示例程序

## 子图方式接入

### 运行图像分类示例程序

### 更新模型

### 更新支持昆仑芯 XPU XTCL 方式的 Paddle Lite 库

- 下载 Paddle Lite 源码；

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译 publish so for amd64 or arm64(FT-2000+/64)；

  ```shell
  $ # For amd64
  $ ./lite/tools/build_linux.sh \
  $     --arch=x86 \
  $     --with_nnadapter=ON \
  $     --nnadapter_with_kunlunxin_xtcl=ON \
  $     --nnadapter_kunlunxin_xtcl_sdk_root=<path-to-kunlunxin-xtcl-sdk-root>

  $ # For arm64(FT-2000+/64), arm 环境下需要设置环境变量 CC 和 CXX，分别指定 C 编译器和 C++ 编译器的路径
  $ export CC=<path_to_your_c_compiler>
  $ export CXX=<path_to_your_c++_compiler>
  $ ./lite/tools/build_linux.sh \
  $     --arch=armv8 \
  $     --with_nnadapter=ON \
  $     --nnadapter_with_kunlunxin_xtcl=ON \
  $     --nnadapter_kunlunxin_xtcl_sdk_root=<path-to-kunlunxin-xtcl-sdk-root>
  ```

- 替换库文件和头文件

  备注：替换头文件后需要重新编译示例程序

## windows 版本的编译适配

  1) Paddle Lite 使用 XPU kernel 的方案

  ```shell
  $ cd Paddle-Lite
  $ lite\\tools\\build_windows.bat with_extra without_python use_vs2017 with_dynamic_crt  with_kunlunxin_xpu kunlunxin_xpu_sdk_root D:\\xpu_toolchain_windows\\output
  ```

  编译脚本 `build_windows.bat` 使用可参考[源码编译( Windows )](../source_compile/windows_compile_windows)进行环境配置和查找相应编译参数

## 其它说明

- 如需更进一步的了解相关产品的信息，请联系欧阳剑 ouyangjian@baidu.com；
- 昆仑芯的研发同学正在持续适配更多的 Paddle 算子，以便支持更多的 Paddle 模型。
