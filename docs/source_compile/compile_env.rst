.. role:: raw-html-m2r(raw)
   :format: html


Paddle Lite 交叉编译概述
======================================================
Paddle Lite 提供了Android/iOS/X86/MacOS平台的官方Release预测库下载，如果您使用的是这四个平台，我们优先推荐您直接下载 `Paddle Lite 预编译库 <https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html>`_。

您也可以根据目标平台选择对应的源码编译方法，Paddle Lite 提供了源码编译脚本，位于*lite/tools/*文件夹下，只需要“编译准备环境”和“执行编译脚本”两个步骤即可一键编译得到目标平台的 Paddle Lite 预测库。

Paddle Lite 已支持多种交叉编译。我们优先建议您使用 `Docker 开发环境 <../>`_ 进行编译，以避免复杂繁琐的环境搭建过程。您也可以遵循 Paddle Lite 官方提供的环境搭建指南，自行在宿主机器上搭建编译环境。

.. list-table::
   :header-rows: 1

   * - No
     - 本机环境（体系结构 / 操作系统）
     - 目标硬件环境（体系结构 / 操作系统）
     - 编译环境搭建及编译指南
   * - 1.
     - x86 Linux
     - x86 Linux
     - `点击进入 <../>`_
   * - 2.
     - x86 Linux
     - ARM Linux
     - `点击进入 <../>`_
   * - 3.
     - x86 Linux
     - ARM Android
     - `点击进入 <../>`_
   * - 4.
     - ARM Linux
     - ARM Linux
     - `点击进入 <../>`_
   * - 5.
     - MacOS
     - MacOS
     - `点击进入 <../>`_
   * - 6.
     - MacOS
     - iOS
     - `点击进入 <../>`_
   * - 7.
     - Windows
     - Windows
     - `点击进入 <../>`_
