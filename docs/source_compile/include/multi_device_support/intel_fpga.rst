Intel FPGA
~~~~~~~~~~~~

* 介绍

Paddle Lite 已支持英特尔 FPGA 平台的预测部署，Paddle Lite 通过调用底层驱动实现对 FPGA 硬件的调度。

如需进行 Intel FPGA相关的编译工作: 请参考 `Paddle Lite 使用英特尔 FPGA 预测部署 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/fpga.html>`_

* 基本参数

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - with_intel_fpga
     - 是否包含 Intel fpga 编译
     - OFF / ON
     - OFF
   * - intel_fpga_sdk_root
     - 设置 Intel fpga sdk 目录
     - `intel_sdk <https://paddlelite-demo.bj.bcebos.com/devices/intel/intel_fpga_sdk_1.0.0.tar.gz>`_
     - 空值