.. role:: raw-html-m2r(raw)
   :format: html

Linux x86 环境下编译适用于 Android 的库
======================================================

简介
----

如果你的本机环境是 x86 架构 + Linux 操作系统，需要部署模型到 Android 系统的目标硬件上，则可以参考本文的介绍，通过 Android NDK 交叉编译工具从源码构建 Paddle Lite 编译包，用于后续应用程序的开发。

..

   **说明：**


   * 
     通常情况下，你不需要自行从源码构建编译包，优先推荐\ `下载 Paddle Lite 官方发布的预编译包 <https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html>`_\ ，可满足一部分场景的需求。如果官方发布的编译包未覆盖你的场景，或者需要修改 Paddle Lite 源代码，则可参考本文构建。

   * 
     本文介绍的编译方法只适用于 Paddle Lite v2.6 及以上版本。v2.3 及之前版本请参考\ `release/v2.3 源码编译方法 <https://paddle-lite.readthedocs.io/zh/release-v2.10_a/source_compile/v2.3_compile.html>`_\ 。


在该场景下 Paddle Lite 已验证的软硬件配置如下表所示：

.. list-table::
   :header-rows: 1

   * - ---
     - 本机环境
     - 目标硬件环境
   * - **操作系统**
     - Linux\ :raw-html-m2r:`<br>`
     - Android 4.1 及以上（芯片版本为 ARM v7 时）\ :raw-html-m2r:`<br>` Android 5.0 及以上（芯片版本为 ARM v8 时）
   * - **芯片层**
     - x86 架构
     - arm64-v8a / armeabi-v7a CPU :raw-html-m2r:`<br>` Huawei Kirin NPU :raw-html-m2r:`<br>`\ MediaTek APU :raw-html-m2r:`<br>` Amlogic NPU :raw-html-m2r:`<br>` OpenCL :raw-html-m2r:`<br>` 注：查询以上芯片支持的具体型号以及对应的手机型号，可参考\ `支持硬件列表 <https://paddle-lite.readthedocs.io/zh/latest/quick_start/support_hardware.html>`_\ 章节。


[1]：OpenCL 是面向异构硬件平台的编译库，Paddle Lite 支持在 Android 系统上运行基于 OpenCL 的程序。

准备编译环境
------------

推荐环境
^^^^^^^^

C++ 环境
""""""""

* gcc、g++ == 8.2.0
* CMake >= 3.10
* Android NDK >= r17c（注意从 ndk-r18 开始，NDK 交叉编译工具仅支持 Clang, 不支持 GCC）
* git、make、wget、python、adb

java 环境
""""""""

* OpenJDK == 1.8.0
* Gradle == 4.1.2
* Android SDK >= 21

环境安装命令
^^^^^^^^^^^^

 以 Ubuntu 为例介绍安装命令。其它 Linux 发行版安装步骤类似，在此不再赘述。
 注意需要 root 用户权限执行如下命令。

.. code-block:: shell

   # 1. 安装 gcc g++ git make wget python unzip adb curl 等基础软件
   apt update
   apt-get install -y --no-install-recommends \
     gcc g++ git make wget python unzip adb curl

   # 2. 安装 jdk
   apt-get install -y default-jdk

   # 3. 安装 CMake，以下命令以 3.10.3 版本为例，其他版本步骤类似。
   wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
       tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
       mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 &&  
       ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
       ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake

   # 4. 下载 linux-x86_64 版本的 Android NDK，以下命令以 r17c 版本为例，其他版本步骤类似。
   cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
   cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip

   # 5. 添加环境变量 NDK_ROOT 指向 Android NDK 的安装路径
   echo "export NDK_ROOT=/opt/android-ndk-r17c" >> ~/.bashrc
   source ~/.bashrc

了解基础编译参数
----------------

Paddle Lite 仓库中\ ``/lite/tools/build_android.sh``\ 脚本文件用于构建 Android 版本的编译包，通过修改\ ``build_android.sh``\ 脚本文件中的参数，可满足不同场景编译包的构建需求，常用的基础编译参数如下表所示，有特殊硬件需求的编译参数见后文。

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - arch
     - 目标硬件的 ARM 架构版本
     - armv8 / armv7
     - armv8
   * - toolchain
     - C++语言的编译器工具链
     - gcc / clang
     - gcc
   * - android_stl
     - 链接到的 Android C++ STL 类型
     - c++_static / c++_shared
     - c++_static
   * - with_java
     - 是否包含 Java 编译包，目标应用程序是 Java 语言时需配置为 ON
     - OFF / ON
     - ON
   * - with_static_lib
     - 是否发布 C++ 静态库
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
   * - with_arm82_fp16
     - 是否开启半精度算子
     - OFF / ON
     - OFF
   * - android_api_level
     - Android API 等级[^2]
     - 16～27
     - armv7:16 / armv8:21


[^2] Paddle Lite 支持的最低安卓版本是 4.1（芯片版本为 ARM v7 时）或 5.0（芯片版本为 ARM v8 时），可通过\ ``--android_api_level``\ 选项设定一个具体的数值，该数值应不低于下表中最低支持的 Android API Level。

.. list-table::
   :header-rows: 1

   * - ARM ABI
     - armv7
     - armv8
   * - 支持的最低 Android API 等级
     - 16
     - 21
   * - 支持的最低 Android 版本
     - 4.1
     - 5.0


..

   **说明：**
   以上参数可在下载 Paddle Lite 源码后直接在\ ``build_android.sh``\ 文件中修改，也可通过命令行指定，具体参见下面编译步骤。


编译步骤
--------

运行编译脚本之前，请先检查系统环境变量 ``NDK_ROOT`` 指向正确的 Android NDK 安装路径。
之后可以下载并构建 Paddle Lite 编译包。

.. code-block:: shell

   # 1. 检查环境变量 `NDK_ROOT` 指向正确的 Android NDK 安装路径
   echo $NDK_ROOT

   # 2. 下载 Paddle Lite 源码并切换到发布分支，如 release/v2.10
   git clone https://github.com/PaddlePaddle/Paddle-Lite.git
   cd Paddle-Lite && git checkout release/v2.10

   # (可选) 删除 third-party 目录，编译脚本会自动从国内 CDN 下载第三方库文件
   # rm -rf third-party

   # 3. 编译 Paddle Lite Android 预测库
   ./lite/tools/build_android.sh

..

   **说明：**
   编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在完成 Paddle Lite 源码下载后，删除本地仓库根目录下的 third-party 目录，编译脚本会自动下载存储于国内 CDN 的第三方依赖文件压缩包，节省从 GitHub repo 同步第三方库的时间。


验证编译结果
------------

如果按\ ``./lite/tools/build_android.sh``\ 中的默认参数执行，成功后会在 ``Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8`` 生成 Paddle Lite 编译包，文件目录如下。

.. code-block:: shell

   inference_lite_lib.android.armv8/
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
   │       ├── libpaddle_api_light_bundled.a             C++ 静态库
   │       └── libpaddle_light_api_shared.so             C++ 动态库
   │
   ├── java                                              Java 预测库
   │   ├── jar
   │   │   └── PaddlePredictor.jar                       Java JAR 包
   │   ├── so
   │   │   └── libpaddle_lite_jni.so                     Java JNI 动态链接库
   │   └── src
   │
   └── demo                                              C++ 和 Java 示例代码
       ├── cxx                                           C++ 预测库示例
       └── java                                          Java 预测库示例

多设备支持
------------

.. include:: include/multi_device_support/opencl.rst

.. include:: include/multi_device_support/nnadapter_support_introduction.rst

.. include:: include/multi_device_support/nnadapter_support_huawei_kirin_npu.rst

.. include:: include/multi_device_support/nnadapter_support_mediatek_apu.rst

.. include:: include/multi_device_support/nnadapter_support_amlogic_npu.rst

.. include:: include/multi_device_support/nnadapter_support_verisilicon_timvx.rst

.. include:: include/multi_device_support/nnadapter_support_android_nnapi.rst