NNAdapter 支持的硬件
---------------------------

* 介绍

NNAdapter 是 Paddle Lite 提供的神经网络算子适配器，您可以通过它调用多种硬件。

* 基本参数

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - with_nnadapter
     - 是否编译 NNAdapter
     - OFF / ON
     - OFF

* 华为麒麟 NPU

华为开发者支持：https://developer.huawei.com/consumer/cn/doc/

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

* 联发科 APU

联发科开发者支持：https://labs.mediatek.com/zh-cn.html

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
