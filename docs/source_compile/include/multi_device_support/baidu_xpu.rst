Baidu XPU
~~~~~~~~~~~~

* 介绍

Paddle Lite 已支持百度 XPU 在 x86 和 arm 服务器（例如飞腾 FT-2000+/64）上进行预测部署。


如需进行百度 XPU 相关的编译工作: 请参考 `Paddle Lite 使用百度 XPU 预测部署 <https://paddle-lite.readthedocs.io/zh/latest/demo_guides/baidu_xpu.html>`_

* 基本参数

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - with_baidu_xpu=ON
     - 是否包含 baidu xpu 编译
     - OFF / ON
     - OFF
   * - baidu_xpu_sdk_root
     - 设置 baidu xpu sdk 目录
     - 空值
     - 空值
   * with_baidu_xpu_xtcl
     - 编译 xpu 库时是否使能 xtcl
     - OFF / ON
     - OFF
   * baidu_xpu_sdk_url
     - baidu xpu sdk 下载链接
     - `baidu_xpu_sdk <https://baidu-kunlun-product.cdn.bcebos.com/KL-SDK/klsdk-dev_paddle>`_
     - 空值
   * baidu_xpu_sdk_env
     - baidu xpu sdk 环境
     - bdcentos_x86_64 / centos7_x86_64 / ubuntu_x86_64 / kylin_aarch64
     - bdcentos_x86_64(x86) / kylin_aarch64(arm)