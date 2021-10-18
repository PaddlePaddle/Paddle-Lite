NNAdapter 支持瑞芯微 NPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * -  nnadapter_with_rockchip_npu
     - 是否编译瑞芯微 NPU 的 NNAdapter HAL 库
     - OFF / ON
     - OFF
   * - nnadapter_rockchip_npu_sdk_root
     - 设置瑞芯微 NPU SDK 目录
     - `rk_npu_ddk <https://github.com/airockchip/rknpu_ddk.git>`_
     - 空值

如需进行瑞芯微 NPU 相关的编译工作: 请参考 `Paddle Lite 使用瑞芯微 NPU 预测部署 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/rockchip_npu.html>`_
