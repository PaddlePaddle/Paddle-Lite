昆仑芯 XPU
^^^^^^^^^^^^

* 介绍

Paddle Lite 已通过算子方式支持昆仑芯 XPU 在 x86 和 ARM 服务器（例如飞腾 FT-2000+/64）上进行预测部署。

* 基本参数

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - with_kunlunxin_xpu
     - 是否包含 kunlunxin xpu 编译
     - OFF / ON
     - OFF
   * - kunlunxin_xpu_sdk_url
     - kunlunxin xpu sdk 下载链接
     - 用户自定义
     - https://baidu-kunlun-product.cdn.bcebos.com/KL-SDK/klsdk-dev_paddle
   * - kunlunxin_xpu_sdk_env
     - kunlunxin xpu sdk 环境
     - bdcentos_x86_64 / centos7_x86_64 / ubuntu_x86_64 / kylin_aarch64
     - bdcentos_x86_64(x86) / kylin_aarch64(arm)
   * - kunlunxin_xpu_sdk_root
     - 设置 kunlunxin xpu sdk 目录
     - 用户自定义
     - 空值

详细请参考 `昆仑芯 XPU 部署示例 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/kunlunxin_xpu.html>`_
