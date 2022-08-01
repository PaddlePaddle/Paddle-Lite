.. role:: raw-html-m2r(raw)
   :format: html

Windows 环境下编译适用于 Windows 的库
==================================================================

简介
----

本文介绍在 Windows 操作系统环境下，如何将 Paddle Lite 源代码编译生成 Windows 平台的预测库。

..

  **说明：**

  * 通常情况下，你不需要自行从源码构建编译包，优先推荐\ `下载Paddle Lite官方发布的预编译包 <https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html>`_\ ，可满足一部分场景的需求。如果官方发布的编译包未覆盖你的场景，或者需要修改Paddle Lite源代码，则可参考本文构建。
  * 本文适用于 release/v2.9 及以上版本，面向对源代码有修改需求的开发者。目前只支持 x86 架构 CPU 的 Windows 操作系统平台，CPU需同时支持 AVX 及 FMA 指令集。

在该场景下 Paddle Lite 已验证的软硬件配置如下表所示：

.. list-table::
   :header-rows: 1

   * - ---
     - 本机环境
     - 目标硬件环境
   * - **操作系统**
     - Windows 10\ :raw-html-m2r:`<br>`
     - Windows 10\ :raw-html-m2r:`<br>`
   * - **芯片层**
     - x86 CPU 架构
     - x86 CPU 架构

准备编译环境
------------

推荐环境
^^^^^^^^^^^^


* Windows 10 专业版
* CMake >= 3.15
* Python == 2.7/3.5.1+
* pip/pip3 >= 9.0.1
* Microsoft Visual Studio >= 2015版本

环境配置步骤
^^^^^^^^^^^^^^^^


#. CMake 需要 3.15 版本, 可在官网\ `下载 <https://cmake.org/download/>`_ Windows 版本，并添加到环境变量中。
#. Python 需要 2.7 及以上版本, 可在官网\ `下载 <https://www.python.org/downloads/windows/>`_\ 。
#. Git 可以在官网\ `下载 <https://gitforwindows.org/>`_\ ，并添加到环境变量中。
#. Visual Studio 请在官网\ `下载 <https://visualstudio.microsoft.com/zh-hans/downloads/>`_\ 所需版本。

编译
--------

编译步骤
^^^^^^^^^^^^

1、 下载代码

.. include:: include/download_code.rst

2、 编译 Paddle Lite Windows 预测库

.. code-block:: dos

   lite\tools\build_windows.bat

.. 

  **说明：** 
  编译过程中，如出现源码编译耗时过长，通常是第三方库下载过慢或失败导致。请在完成 Paddle Lite 源码下载后，删除本地仓库根目录下的 third-party 目录，编译脚本会自动下载存储于国内 CDN 的第三方依赖文件压缩包，节省从 GitHub repo 同步第三方库的时间。

编译参数说明
^^^^^^^^^^^^^^^^

``build_windows.bat`` 编译参数详细说明如下。

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
   * - without_log
     - 编译不带日志的预测库（默认带日志）
   * - without_python
     - 不编译 Python 预测库
   * - with_extra
     - 编译完整算子的预测库（当编译 Python 预测库时，默认编译包含完整算子预测库)，详情可参考\ `预测库说明 <./library.html>`_\ 。
   * - with_profile
     - 支持逐层耗时分析
   * - with_precision_profile
     - 支持逐层精度分析
   * - build_x86
     - 编译 Windows 32位预测库（默认为 Windows 64位）
   * - with_static_mkl
     - 静态链接 Intel(R) MKL 加速库
   * - with_dynamic_crt
     - 动态链接 MSVC Rumtime 即 MD_DynamicRelease
   * - with_opencl
     - 开启 OpenCL，编译出的预测库支持在 GPU 上运行（默认编译的预测库仅在 CPU 上运行)
   * - use_ninja
     - 使用 `Ninja <https://ninja-build.org/>`_ 构建系统（默认使用 vs2015 的 MSBuild 构建方案，添加上此编译选项使用 Ninja 编译构建)
   * - use_vs2017
     - 使用 vs2017 构建系统（默认使用 vs2015 的构建方案，添加上此编译选项使用 vs2017 编译构建)
   * - use_vs2019
     - 使用 vs2019 构建系统（默认使用 vs2015 的构建方案，添加上此编译选项使用 vs2019 编译构建)
   * - without_avx
     - 使用 AVX/SSE 指令对 x86 Kernel 进行加速
   * - with_kunlunxin_xpu
     - 使用昆仑芯 XPU kernel 进行加速
   * - kunlunxin_xpu_sdk_root
     - 启用 with_kunlunxin_xpu 时，需要添加昆仑芯 XPU 的 Windows 产出包的相关路径


编译脚本使用示例
^^^^^^^^^^^^^^^^^^^^

编译 Windows 平台不带日志 32 位的预测库

.. code-block:: dos

   lite\tools\build_windows.bat without_log build_x86

验证编译结果
------------

编译结果位于 ``build.lite.x86\inference_lite_lib``

详细内容如下：

1、 ``cxx``\ 文件夹：包含 c++ 的库文件与相应的头文件


* ``include``  ： 头文件
* ``lib`` ： 库文件

  * 静态库文件：

    * ``libpaddle_api_full_bundled.lib``  ：full_api 静态库
    * ``libpaddle_api_light_bundled.lib`` ：light_api 静态库

2、 ``third_party`` 文件夹：依赖的第三方预测库 mklml


* mklml ： Paddle Lite 预测库依赖的 mklml 数学库

3、 ``demo\cxx``\ 文件夹：C++ 示例 demo


* ``mobilenetv1_full`` ：使用 full_api 执行 mobilenet_v1 预测的 C++ demo
* ``mobilenetv1_light`` ：使用 light_api 执行 mobilenet_v1 预测的 C++ demo

4、 ``demo\python``\ ： Python 示例 demo


* ``mobilenetv1_full_api.py``\ ：使用 full_api 执行 mobilenet_v1 预测的 Python demo
* ``mobilenetv1_light_api.py``\ ：使用 full_api 执行 mobilenet_v1 预测的 Python demo

5、 ``python``\ 文件夹：包含 Python 的库文件和对应的 .whl 包


* ``install``\ 文件夹：编译成功的 .whl 包位于\ ``install\dist\*.whl``
* ``lib``\ 文件夹：.whl 包依赖的库文件
