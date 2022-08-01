.. role:: raw-html-m2r(raw)
   :format: html

ARM Linux 环境下编译适用于 ARM Linux 的库
===============================================================

简介
------------------------------------------------------

本文档旨在介绍如何在 ARM Linux 操作系统环境下编译 Paddle Lite 源码，生成目标硬件为 ARM Linux 的预测库。

..

   **说明：**


   * 
     通常情况下，你不需要自行从源码构建编译包，优先推荐\ `下载 Paddle Lite 官方发布的预编译包 <https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html>`_\ ，可满足一部分场景的需求。如果官方发布的编译包未覆盖你的场景，或者需要修改 Paddle Lite 源代码，则可参考本文构建。

   * 
     本文介绍的编译方法只适用于 Paddle Lite v2.6及以上版本。v2.3及之前版本请参考\  `release/v2.3 源码编译方法 <https://paddle-lite.readthedocs.io/zh/release-v2.10_a/source_compile/v2.3_compile.html>`_\ 。

在该场景下 Paddle Lite 已验证的软硬件配置如下表所示:

.. list-table::
   :header-rows: 1

   * - Host环境
     - 目标硬件环境
   * - ARM-Linux
     - CPU arm64/armhf :raw-html-m2r:`<br>` Huawei Ascend NPU :raw-html-m2r:`<br>` Kunlunxin XPU :raw-html-m2r:`<br>` OpenCL :raw-html-m2r:`<br>` 注：查询以上芯片支持的具体型号，可参考\ `支持硬件列表 <https://paddle-lite.readthedocs.io/zh/develop/quick_start/support_hardware.html>`_\ 章节。

准备编译环境
------------------------------------------------------

适用于基于 ARMv8 和 ARMv7 架构 CPU 的各种开发板，例如 RK3399，树莓派等，目前支持交叉编译和本地编译两种方式，对于交叉编译方式，在完成目标程序编译后，可通过 scp 方式将程序拷贝到开发板运行。
因为本教程使用 Host 环境为 ARM 架构，因此下面仅介绍本地编译 ARM Linux 方式。

本地编译ARM Linux
^^^^^^^^^^^^^^^^^^^^^^^^
* gcc、g++、git、make、wget、python、pip、python-dev、patchelf
* cmake（建议使用 3.10 或以上版本）

环境安装命令
^^^^^^^^^^^^

 以 Ubuntu 为例介绍安装命令。其它 Linux 发行版安装步骤类似，在此不再赘述。
 注意需要 root 用户权限执行如下命令。

.. code-block:: shell

  # 1. Install basic software
  apt update
  apt-get install -y --no-install-recommends \
    gcc g++ make wget python unzip patchelf python-dev

  # 2. install cmake 3.10 or above
  wget https://www.cmake.org/files/v3.10/cmake-3.10.3.tar.gz
  tar -zxvf cmake-3.10.3.tar.gz
  cd cmake-3.10.3
  ./configure
  make
  sudo make install

了解基础编译参数
----------------

Paddle Lite 仓库中\ ``./lite/tools/build_linux.sh``\ 脚本文件用于构建 linux 版本的编译包，通过修改\ ``build_linux.sh``\ 脚本文件中的参数，可满足不同场景编译包的构建需求，常用的基础编译参数如下表所示：
有特殊硬件需求的编译参数见后文。

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - arch
     - 目标硬件的架构版本
     - armv8 / armv7hf / armv7 
     - armv8
   * - toolchain
     - C++ 语言的编译器工具链
     - gcc
     - gcc
   * - with_python
     - 是否包含 python 编译包，目标应用程序是 python 语言时需配置为 ON
     - OFF / ON
     - OFF
   * - with_cv
     - 是否将 cv 函数加入编译包中
     - OFF / ON
     - OFF
   * - with_log
     - 是否在执行过程打印日志
     - OFF / ON
     - ON
   * - with_exception
     - 是否开启 C++ 异常
     - OFF / ON
     - OFF
   * - with_extra
     - 是否编译完整算子（见\ `支持算子 <https://paddle-lite.readthedocs.io/zh/develop/quick_start/support_operation_list.html>`_\ 一节）
     - OFF / ON
     - OFF
   * - with_profile
     - 是否打开执行耗时分析
     - OFF / ON
     - OFF
   * - with_precision_profile
     - 是否打开逐层精度结果分析
     - OFF / ON
     - OFF
   * - with_opencl
     - 是否编译支持 OpenCL 的预测库
     - OFF / ON
     - OFF

.. code-block:: shell

   # 打印 help 信息，查看更多编译选项
   ./lite/tools/build_linux.sh help

..


编译步骤
--------

.. code-block:: shell

   # 下载 Paddle Lite 源码并切换到发布分支，如 develop
   git clone https://github.com/PaddlePaddle/Paddle-Lite.git
   cd Paddle-Lite && git checkout develop

   # (可选) 删除 third-party 目录，编译脚本会自动从国内 CDN 下载第三方库文件
   # rm -rf third-party

   # 执行编译脚本
   ./lite/tools/build_linux.sh

..
   
   **说明：**
   编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在完成 Paddle Lite 源码下载后，删除本地仓库根目录下的 third-party 目录，编译脚本会自动下载存储于国内 CDN 的第三方依赖文件压缩包，节省从 GitHub repo 同步第三方库的时间。

验证编译结果
------------

如果按\ ``./lite/tools/build_linux.sh``\ 中的默认参数执行，成功后会在 ``Paddle-Lite/build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/`` 生成 Paddle Lite 编译包，文件目录如下。

.. code-block:: shell

   inference_lite_lib.armlinux.armv8/
   ├── cxx                                               C++ 预测库和头文件
   │   ├── include                                       C++ 头文件
   │   │   ├── paddle_api.h
   │   │   ├── paddle_image_preprocess.h
   │   │   ├── paddle_lite_factory_helper.h
   │   │   ├── paddle_place.h
   │   │   ├── paddle_use_kernels.h
   │   │   ├── paddle_use_ops.h
   │   │   └── paddle_use_passes.h
   │   └── lib                                           C++ 预测库
   │       └── libpaddle_light_api_shared.so             C++ light_api 动态库
   │
   └── demo                                              C++
   │   └── cxx                                           C++ 预测库 demo


多设备支持
------------

.. include:: include/multi_device_support/opencl.rst

.. include:: include/multi_device_support/kunlunxin_xpu.rst

.. include:: include/multi_device_support/nnadapter_support_introduction.rst

.. include:: include/multi_device_support/nnadapter_support_huawei_ascend_npu.rst
