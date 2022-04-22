.. role:: raw-html-m2r(raw)
   :format: html

Linux x86 环境下编译适用于 Linux x86 的库
===============================================================

简介
------------------------------------------------------

本文档旨在介绍如何在 x86 Linux 操作系统环境下编译 Paddle Lite 源码，生成目标硬件为 x86 Linux 的预测库。

..

   **说明：**


   * 
     通常情况下，你不需要自行从源码构建编译包，优先推荐\ `下载 Paddle Lite 官方发布的预编译包 <https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html>`_\ ，可满足一部分场景的需求。如果官方发布的编译包未覆盖你的场景，或者需要修改Paddle Lite源代码，则可参考本文构建。

   * 
     本文介绍的编译方法只适用于 Paddle Lite v2.6 及以上版本。v2.3 及之前版本请参考\ `release/v2.3 源码编译方法 <https://paddle-lite.readthedocs.io/zh/release-v2.10_a/source_compile/v2.3_compile.html>`_\ 。

在该场景下 Paddle Lite 已验证的软硬件配置如下表所示:

.. list-table::
   :header-rows: 1

   * - Host 环境
     - 目标硬件环境
   * - x86 Linux
     - CPU x86_32/64 :raw-html-m2r:`<br>` Huawei Ascend NPU :raw-html-m2r:`<br>` Kunlunxin XPU :raw-html-m2r:`<br>` OpenCL :raw-html-m2r:`<br>` 注：查询以上芯片支持的具体型号，可参考\ `支持硬件列表 <https://paddle-lite.readthedocs.io/zh/develop/quick_start/support_hardware.html>`_\ 章节。

准备编译环境
------------------------------------------------------

环境要求
^^^^^^^^
* gcc、g++ == 8.2.0
* CMake >= 3.10
* git、make、wget、python

环境安装命令
^^^^^^^^^^^^

 以 Ubuntu 为例介绍安装命令。其它 Linux 发行版安装步骤类似，在此不再赘述。
 注意需要 root 用户权限执行如下命令。

.. code-block:: shell

   # 1. 安装 gcc g++ git make wget python unzip curl等基础软件
   apt update
   apt-get install -y --no-install-recommends \
     gcc g++ git make wget python unzip curl

   # 2. 安装 CMake，以下命令以 3.10.3 版本为例，其他版本步骤类似。
   wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
       tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
       mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 &&  
       ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
       ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake


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
     - armv8 / armv7hf / armv7 / x86 
     - armv8
   * - toolchain
     - C++语言的编译器工具链
     - gcc / clang
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
   * - with_static_mkl 
     - 是否编译静态链接的 MKL 库，否则为动态链接(目标 os 架构为 x86 时可以选择是否开启)
     - OFF / ON
     - OFF
   * - with_avx
     - 是否使用 AVX/SSE 指令对 X86 Kernel 进行加速(目标 os 架构为 x86 时可以选择是否开启)
     - OFF / ON
     - ON
   * - with_opencl
     - 是否编译支持 OpenCL 的预测库
     - OFF / ON
     - OFF

   
编译步骤
--------

.. code-block:: shell

   # 下载 Paddle Lite 源码并切换到发布分支，如 develop
   git clone https://github.com/PaddlePaddle/Paddle-Lite.git
   cd Paddle-Lite && git checkout develop

   # (可选) 删除 third-party 目录，编译脚本会自动从国内 CDN 下载第三方库文件
   # rm -rf third-party

   ./lite/tools/build_linux.sh --arch=x86

..
   
   **说明：**
   编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在完成 Paddle Lite 源码下载后，删除本地仓库根目录下的 third-party 目录，编译脚本会自动下载存储于国内 CDN 的第三方依赖文件压缩包，节省从 GitHub repo 同步第三方库的时间。

验证编译结果
------------

如果按\ ``./lite/tools/build_linux.sh --arch=x86``\ 中的默认参数执行，成功后会在 ``Paddle-Lite/build.lite.linux.x86.gcc/inference_lite_lib/`` 生成 Paddle Lite 编译包，文件目录如下。

.. code-block:: shell

   inference_lite_lib/
   ├── bin
   │   └── test_model_bin                                可执行工具文件
   ├── cxx                                               C++ 预测库和头文件
   │   ├── include                                       C++ 头文件
   │   │   ├── paddle_api.h
   │   │   ├── paddle_lite_factory_helper.h
   │   │   ├── paddle_place.h
   │   │   ├── paddle_use_kernels.h
   │   │   ├── paddle_use_ops.h
   │   │   └── paddle_use_passes.h
   │   └── lib                                           C++ 预测库
   │       ├── libpaddle_api_light_bundled.a             C++ light_api 静态库
   │       ├── libpaddle_api_full_bundled.a              C++ full_api 静态库
   │       ├── libpaddle_light_api_shared.so             C++ light_api 动态库
   │       └── libpaddle_full_api_shared.so              C++ full_api 动态库
   │
   └── demo                                              C++
   │   └── cxx                                           C++ 预测库 demo
   └── third_party
   │   └── mklml                                         依赖的第三方加速库 Intel(R) MKL
   │       ├── include                                   
   │       └── lib                                       
   │           ├── libiomp5.so
   │           ├── libmklml_gnu.so
   │           └── libmklml_intel.so


多设备支持
------------

.. include:: include/multi_device_support/opencl.rst

.. include:: include/multi_device_support/kunlunxin_xpu.rst

.. include:: include/multi_device_support/nnadapter_support_introduction.rst

.. include:: include/multi_device_support/nnadapter_support_huawei_ascend_npu.rst
