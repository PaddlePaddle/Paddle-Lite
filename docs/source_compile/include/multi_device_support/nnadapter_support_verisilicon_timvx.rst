NNAdapter 支持芯原 TIM-VX
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - nnadapter_with_verisilicon_timvx
     - 是否编译芯原 TIM-VX 的 NNAdapter HAL 库
     - OFF / ON
     - OFF
   * - nnadapter_verisilicon_timvx_src_git_tag
     - 设置芯原 TIM-VX 的代码分支
     - TIM-VX repo 分支名
     - main
   * - nnadapter_verisilicon_timvx_viv_sdk_url
     - 设置芯原 TIM-VX SDK 的下载链接
     - 用户自定义
     - Android系统：http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_android_9_armeabi_v7a_6_4_4_3_generic.tgz
       Linux系统：http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_4_3_generic.tgz
   * - nnadapter_verisilicon_timvx_viv_sdk_root
     - 设置芯原 TIM-VX 的本地路径
     - 用户自定义
     - 空值
详细请参考 `芯原 TIM-VX 部署示例 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/verisilicon_timvx.html>`_
