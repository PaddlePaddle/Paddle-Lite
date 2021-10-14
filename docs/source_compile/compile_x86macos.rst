.. role:: raw-html-m2r(raw)
   :format: html


使用 MacOS 环境编译 Paddle Lite / 目标硬件 OS 为 MacOS
======================================================

简介
----

如果你的本机环境是 MacOS 操作系统（暂不支持 M1 机型），需要部署模型到 MacOS 系统的目标硬件上，则可以参考本文的介绍，从源码构建 Paddle Lite 编译包，用于后续应用程序的开发。

在该场景下 Paddle Lite 已验证的软硬件配置如下表所示：

.. list-table::
   :header-rows: 1

   * - ---
     - 本机环境
     - 目标硬件环境
   * - **操作系统**
     - MacOS\ :raw-html-m2r:`<br>`
     - MacOS\ :raw-html-m2r:`<br>`
   * - **芯片层**
     - x86 Intel CPU
     - x86 Intel CPU :raw-html-m2r:`<br>`


[^1]：OpenCL 是面向异构硬件平台的编译库，Paddle Lite 支持在 x86 MacOS 系统上运行基于 OpenCL 的程序。

   **说明：**


   * 
     通常情况下，你不需要自行从源码构建编译包，优先推荐\ `下载 Paddle Lite 官方发布的预编译包 <https://paddle-lite.readthedocs.io/zh/develop/quick_start/release_lib.html>`_\ ，可满足一部分场景的需求。如果官方发布的编译包未覆盖你的场景，或者需要修改Paddle Lite源代码，则可参考本文构建。

   * 
     本文介绍的编译方法适用于 Paddle Lite v2.10 及以上版本。v2.3 及之前版本请参考\ `release/v2.3源码编译方法 <https://paddle-lite.readthedocs.io/zh/develop/source_compile/v2.3_compile.html>`_\ 。



准备编译环境
------------

环境要求
^^^^^^^^


* gcc、g++（推荐版本为 8.2.0）
* git、make、wget
* CMake（请使用 3.15 或以上版本）


环境安装命令
^^^^^^^^^^^^

 注意需要提前安装 Homebrew。

.. code-block:: shell

   # 1. Install basic software
   brew install curl gcc git make unzip wget 

   # 2. 安装 CMake，以下命令以3.15版本为例，其他版本步骤类似。
   mkdir /usr/local/Cellar/cmake/ && cd /usr/local/Cellar/cmake/ \
       cd /usr/local/Cellar/cmake/ \
       wget https://cmake.org/files/v3.15/cmake-3.15.2-Darwin-x86_64.tar.gz \
       tar zxf ./cmake-3.15.2-Darwin-x86_64.tar.gz \
       mv cmake-3.15.2-Darwin-x86_64/CMake.app/Contents/ ./3.15.2 \
       ln -s /usr/local/Cellar/cmake/3.15.2/bin/cmake /usr/local/bin/cmake

了解基础编译参数
----------------

Paddle Lite 仓库中\ ``/lite/tools/build_macos.sh``\ 脚本文件用于构建 MacOS 版本的编译包，通过修改\ ``build_macos.sh``\ 脚本文件中的参数，可满足不同场景编译包的构建需求，常用的基础编译参数如下表所示：
有特殊硬件需求的编译参数见后文。

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
     - 可选范围
     - 默认值
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
     - 是否编译完整算子（见\ `支持算子 <https://paddle-lite.readthedocs.io/zh/develop/introduction/support_operation_list.html>`_\ 一节）
     - OFF / ON
     - OFF
   * - with_profile
     - 是否打开执行耗时分析
     - OFF / ON
     - OFF
   * - build_opencl
     - 是否编译 openCL
     - OFF / ON
     - OFF


..

   **说明：**
   以上参数可在下载 Paddle Lite 源码后直接在\ ``build_macos.sh``\ 文件中修改，也可通过命令行指定，具体参见下面编译步骤。


编译步骤
--------

下载并构建 Paddle Lite 编译包。

.. code-block:: shell

   # 1. 下载 Paddle Lite 源码并切换到发布分支，如 develop
   git clone https://github.com/PaddlePaddle/Paddle-Lite.git
   cd Paddle-Lite && git checkout develop

   # (可选) 删除 third-party 目录，编译脚本会自动从国内CDN下载第三方库文件
   # rm -rf third-party

   # 2. 编译 Paddle Lite x86 MacOS 预测库
   ./lite/tools/build_macos.sh x86

..

   **说明：**
   编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在完成 Paddle Lite 源码下载后，删除本地仓库根目录下的 third-party 目录，编译脚本会自动下载存储于国内 CDN 的第三方依赖文件压缩包，节省从 GitHub repo 同步第三方库的时间。


验证编译结果
------------

如果执行\ ``/lite/tools/build_macos.sh x86``\ 中的默认参数执行，成功后会在 ``Paddle-Lite/build.lite.x86/inference_lite_lib`` 生成 Paddle Lite 编译包，文件目录如下。

.. code-block:: shell

   inference_lite_lib/
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
   │       ├── libpaddle_api_light_bundled.a             C++ 静态库(轻量库)
   │       └── libpaddle_light_api_shared.dylib          C++ 动态库(轻量库)
   │       ├── libpaddle_api_full_bundled.a.a            C++ 静态库(全量库)
   │       └── libpaddle_full_api_shared.dylib           C++ 动态库(全量库)
   │
   ├── third_party                                       第三方库
   │   ├── gflags
   │   ├── glog
   │   ├── mklml
   │   ├── protobuf
   │   └──  xxhash
   │
   └── demo                                              C++ 示例代码
       └── cxx                                           C++ 预测库demo


多设备支持
------------

.. include:: include/multi_device_support/opencl.rst

