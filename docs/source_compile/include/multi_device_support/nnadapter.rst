NNAdapter 支持的硬件
~~~~~~~~~~~~~~~~~~~~

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

如需进行华为麒麟 NPU 相关的编译工作: 请参考 `Paddle Lite 使用华为麒麟 NPU 预测部署 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/huawei_kirin_npu.html>`_

* 华为昇腾 NPU

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - nnadapter_with_huawei_ascend_npu
     - 是否编译华为昇腾 NPU 的 NNAdapter HAL 库
     - OFF / ON
     - OFF
   * - nnadapter_huawei_ascend_npu_sdk_root
     - 设置华为昇腾 NPU DDK 目录
     - 用户自定义
     - /usr/local/Ascend/ascend-toolkit/latest

如需进行华为昇腾 NPU 相关的编译工作: 请参考 `Paddle Lite 使用华为昇腾 NPU 预测部署 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/huawei_ascend_npu.html>`_

* 联发科 APU

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

如需进行联发科 APU 相关的编译工作: 请参考 `Paddle Lite 使用联发科 APU 预测部署 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/mediatek_apu.html>`_

* 瑞芯微 NPU

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

* Amlogic NPU

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - nnadapter_with_amlogic_npu
     - 是否编译 Amlogic NPU 的 NNAdapter HAL 库
     - OFF / ON
     - OFF
   * - nnadapter_amlogic_npu_sdk_root
     - 设置 Amlogic NPU SDK 目录
     - 用户自定义
     - 空值

如需进行 Amlogic NPU 相关的编译工作: 请参考 `Paddle Lite 使用 Amlogic NPU 预测部署 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/amlogic_npu.html>`_

* Imagination NNA

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - nnadapter_with_imagination_nna
     - 是否编译 Imagination NNA 的 NNAdapter HAL 库
     - OFF / ON
     - OFF
   * - nnadapter_imagination_nna_sdk_root
     - 设置 Imagination NNA SDK 目录
     - 用户自定义
     - 空值

如需进行 Imagination NNA 相关的编译工作: 请参考 `Paddle Lite 使用颖脉 NNA 预测部署 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/imagination_nna.html>`_

