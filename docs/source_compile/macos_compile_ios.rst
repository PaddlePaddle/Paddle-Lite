.. role:: raw-html-m2r(raw)
   :format: html

macOS 环境下编译适用于 iOS 的库
======================================================

简介
----

如果你的本机环境是 macOS 操作系统，需要部署模型到 iOS 系统的目标硬件上，则可以参考本文的介绍，通过 Xcode 工具从源码构建 Paddle Lite 的编译包，用于后续应用程序的开发。

..

   **说明：**


   *
     通常情况下，你不需要自行从源码构建编译包，优先推荐\ `下载 Paddle Lite 官方发布的预编译包 <https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html>`_\ ，可满足一部分场景的需求。如果官方发布的编译包未覆盖你的场景，或者需要修改 Paddle Lite 源代码，则可参考本文构建。

   *
     自 release/v2.10 版本起，Paddle Lite 支持了 Metal 后端。


在该场景下 Paddle Lite 已验证的软硬件配置如下表所示：

.. list-table::
   :header-rows: 1

   * - ---
     - 本机环境
     - 目标硬件环境
   * - **操作系统**
     - macOS\ :raw-html-m2r:`<br>`
     - iOS 9.0 及以上\ :raw-html-m2r:`<br>`
   * - **芯片层**
     - x86/arm 架构
     - arm64-v8a/armeabi-v7a


准备编译环境
------------

推荐环境
^^^^^^^^


* Xcode IDE >= 10.1
* CMake >= 3.15
* git、make、wget、python

环境安装命令
^^^^^^^^^^^^


.. code-block:: shell

   # 1. 安装 curl gcc git make unzip wget python cmake 等基础软件。
   brew install curl gcc git make unzip wget python cmake

   # 2. 安装 Xcode，可通过 App Store 下载并安装，安装后需要启动一次并执行下面语句。
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

了解基础编译参数
----------------

Paddle Lite 仓库中\ ``./lite/tools/build_ios.sh``\ 脚本文件用于构建 iOS 版本的编译包，通过修改\ ``build_ios.sh``\ 脚本文件中的参数，可满足不同场景编译包的构建需求，常用的基础编译参数如下表所示：
有特殊硬件需求的编译参数见后文。

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
   * - with_metal
     - 是否编译支持 Metal 的预测库
     - OFF / ON
     - OFF
   * - ios_deployment_target
     - 运行系统的最低版本
     - 9.0 及以上 / 9.0
     - 9.0


..

   **说明：**
   执行\ ``./lite/tools/build_ios.sh help``\ 可输出各选项的使用说明信息。


编译步骤
--------


.. code-block:: shell

   # 1. 下载 Paddle Lite 源码并切换到发布分支，如 release/v2.10
   git clone https://github.com/PaddlePaddle/Paddle-Lite.git
   cd Paddle-Lite && git checkout release/v2.10

   # (可选) 删除 third-party 目录，编译脚本会自动从国内 CDN 下载第三方库文件
   # rm -rf third-party

   # 2. 编译 Paddle Lite iOS 预测库
   ./lite/tools/build_ios.sh

..

   **说明：**
   编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在完成 Paddle Lite 源码下载后，删除本地仓库根目录下的 third-party 目录，编译脚本会自动下载存储于国内 CDN 的第三方依赖文件压缩包，节省从 GitHub repo 同步第三方库的时间。


验证编译结果
------------

iOS CPU 预测库 (armv8)
^^^^^^^^^^^^^^^^^^^^^^


如果执行\ ``./lite/tools/build_ios.sh``\ ，成功后会在 ``Paddle-Lite/build.ios.ios64.armv8/inference_lite_lib.ios64.armv8`` 生成 Paddle Lite 编译包，文件目录如下：

.. code-block:: shell

   inference_lite_lib.ios64.armv8
   ├── include                                                C++ 头文件
   │   ├── paddle_api.h
   │   ├── paddle_image_preprocess.h
   │   ├── paddle_lite_factory_helper.h
   │   ├── paddle_place.h
   │   ├── paddle_use_kernels.h
   │   ├── paddle_use_ops.h
   │   └── paddle_use_passes.h
   └── lib                                                    C++ 预测库（静态库）
       └── libpaddle_api_light_bundled.a


iOS GPU 预测库 (armv8)
^^^^^^^^^^^^^^^^^^^^^^


如果执行\ ``./lite/tools/build_ios.sh --with_metal=ON``\ ，成功后会在 ``Paddle-Lite/build.ios.metal.ios64.armv8/inference_lite_lib.ios64.armv8.metal`` 生成 Paddle Lite 编译包，文件目录如下：

.. code-block:: shell

   inference_lite_lib.ios64.armv8.metal
   ├── include                                                C++ 头文件
   │   ├── paddle_api.h
   │   ├── paddle_image_preprocess.h
   │   ├── paddle_lite_factory_helper.h
   │   ├── paddle_place.h
   │   ├── paddle_use_kernels.h
   │   ├── paddle_use_ops.h
   │   └── paddle_use_passes.h
   ├── metal                                                  metallib 文件
   │   └── lite.metallib
   └── lib                                                    C++ 预测库（静态库）
       └── libpaddle_api_light_bundled.a

