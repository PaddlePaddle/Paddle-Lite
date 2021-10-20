NNAdapter 支持联发科 APU
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - nnadapter_with_mediatek_apu
     - 是否编译联发科 APU 的 NNAdapter HAL 库
     - OFF / ON
     - OFF
   * - nnadapter_mediatek_apu_sdk_root
     - 设置联发科 Neuron Adapter SDK 目录
     - `apu_ddk <https://paddlelite-demo.bj.bcebos.com/devices/mediatek/apu_ddk.tar.gz>`_
     - 空值

详细请参考 `联发科 APU 部署示例 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/mediatek_apu.html>`_
