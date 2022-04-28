NNAdapter 支持英伟达 TensorRT
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::

   * - 参数
     - 说明
     - 可选范围
     - 默认值
   * - nnadapter_with_nvidia_tensorrt
     - 是否编译英伟达 TensorRT 的 NNAdapter HAL 库
     - OFF / ON
     - OFF
   * - nnadapter_nvidia_cuda_root
     - 设置 CUDA 路径
     - 用户自定义
     - 空值
   * - nnadapter_nvidia_tensorrt_root
     - 设置 Tensor  路径
     - 用户自定义
     - 空值

详细请参考 `英伟达 TensorRT 部署示例 <https://paddle-lite.readthedocs.io/zh/develop/demo_guides/nvidia_tensorrt.html>`_