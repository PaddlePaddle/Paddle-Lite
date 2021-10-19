NNAdapter 支持华为麒麟 NPU
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - nnadapter_with_huawei_kirin_npu
     - 是否编译华为麒麟 NPU 的 NNAdapter HAL 库
     - OFF / ON
     - OFF
   * - nnadapter_huawei_kirin_npu_sdk_root
     - 设置华为 HiAI DDK 目录
     - `hiai_ddk_lib_510 <https://paddlelite-demo.bj.bcebos.com/devices/huawei/kirin/hiai_ddk_lib_510.tar.gz>`_
     - 空值

详细请参考 `华为麒麟 NPU 部署示例 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/huawei_kirin_npu.html>`_