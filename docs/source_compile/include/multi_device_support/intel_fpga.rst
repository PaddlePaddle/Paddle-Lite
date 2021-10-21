英特尔 FPGA
^^^^^^^^^^^^

* 介绍

Paddle Lite 已通过算子方式支持英特尔 FPGA 平台的预测部署。

* 基本参数

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - with_intel_fpga
     - 是否包含英特尔 FPGA 编译
     - OFF / ON
     - OFF
   * - intel_fpga_sdk_root
     - 设置英特尔 FPGA sdk 目录
     - `intel_sdk <https://paddlelite-demo.bj.bcebos.com/devices/intel/intel_fpga_sdk_1.0.0.tar.gz>`_
     - 空值

详细请参考 `英特尔 FPGA 部署示例 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/fpga.html>`_
