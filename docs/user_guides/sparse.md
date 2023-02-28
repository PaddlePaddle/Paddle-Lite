# 模型的非结构化稀疏

常见的稀疏方式可分为结构化稀疏和非结构化稀疏。前者在某个特定维度（特征通道、卷积核等等）上对卷积、矩阵乘法做剪枝操作，然后生成一个更小的模型结构，这样可以复用已有的卷积、矩阵乘计算，无需特殊实现推理算子；后者以每一个参数为单元稀疏化，然而并不会改变参数矩阵的形状，只是变成了含有大量零值的稀疏矩阵，所以更依赖于推理库、硬件对于稀疏后矩阵运算的加速能力。更多介绍请参照[这篇技术文章](https://mp.weixin.qq.com/s/l__C5IOu3z7uQdcWKnViOw)。

本文从推理的视角，介绍如何基于 Paddle Lite 的系列工具，在稀疏模型上获得更优性能。

## 非结构化稀疏训练

### 1 简介

稀疏化训练是使用全量训练数据，对训练好的稠密模型进行稀疏。在训练过程中，该方法只优化部分重要参数，对不重要的参数置零，达到保证稀疏模型精度的效果。

使用条件：

- 有预训练模型
- 有全量的训练数据

使用步骤：

-  产出稀疏模型：使用 PaddleSlim 调用稀疏训练接口，产出稀疏模型
-  稀疏模型预测：使用 Paddle Lite 加载稀疏模型进行预测推理

优点：

-  减小计算量、降低计算内存、减小 FP32 模型大小
-  模型精度受稀疏影响小

缺点：

-  需要全量数据，训练时间较长

建议首先使用 [虚拟稀疏](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/utils/prune_model.py#L12) 的接口对稠密推理模型进行快速稀疏（只保证稀疏度，不保证精度）；然后使用稀疏模型进行预测。如果该稀疏模型的性能达不到要求或超出要求，再调大或者调小稀疏度；最后使用适合的稀疏度开始稀疏训练。

### 2 产出稀疏模型

目前，PaddleSlim 的稀疏训练主要针对 1x1卷积，对应算子是 conv2d。Paddle Lite 支持运行 PaddleSlim 稀疏训练产出的模型，可以加快模型在移动端的执行速度。

温馨提示：如果您是初次接触 PaddlePaddle 框架，建议首先学习[使用文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/index_cn.html)。

使用 PaddleSlim 模型压缩工具训练稀疏模型，请参考文档：
* 稀疏训练接口 [动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/pruners/unstructured_pruner.rst)|[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/prune/unstructured_prune_api.rst)
* 稀疏训练Demo [动态图](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/dygraph/unstructured_pruning)| [静态图](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/unstructured_prune)


### 3 使用 Paddle Lite 运行稀疏模型推理

首先，使用 Paddle Lite 提供的模型转换工具（model_optimize_tool）将稀疏模型转换成移动端预测的模型，然后加载转换后的模型进行预测部署。

#### 3.1 模型转换

参考[模型转换](../user_guides/model_optimize_tool.md)准备模型转换工具，建议从 Release 页面下载。

参考[模型转换](../user_guides/model_optimize_tool.md)使用模型转换工具，参数按照实际情况设置。比如在安卓手机ARM端进行预测，模型转换的命令为：

```bash
./OPT --model_dir=./mobilenet_v1_quant \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_quant_opt \
      --valid_targets=arm \
      --sparse_model=true --sparse_threshold=0.5
```

注意，我们通过上述的 sparse_model 和 sparse_threshold 两个参数控制是否对模型进行稀疏优化：

 - 当 sparse_model=false时，稀疏优化关闭，所有的参数都不会被稀疏
 - 当 sparse_model=true时，稀疏优化打开
	 - 当前参数矩阵稀疏度大于 sparse_threshold 时，会被稀疏
	 - 当前参数矩阵稀疏度小于 sparse_threshold 时，不会被稀疏

#### 3.2 稀疏模型预测

和 FP32 模型一样，转换后的稀疏模型可以在 Android APP 中加载预测，建议参考[C++ Demo](./cpp_demo.md)。


### FAQ

**问题**：为什么模型优化(*.nb文件)后，稀疏 FP32 模型的体积比稠密 FP32 小了，但是稀疏 INT8 模型的体积反而比稠密 INT8 模型体积大了？

**解答**：这是可能出现的现象，因为稀疏格式中，我们虽然节省了部分 INT8 参数的存储空间，但是引入了 INT32 类型的 index，所以理论上75%稀疏度以下时，INT8 模型体积是会有些增大的。

**问题**：当前非结构化稀疏的适用范围是什么
  
**解答**：在推理上， PaddleLite-2.11 支持 1x1卷积的非结构化和半结构化稀疏（2x1 的block为一个单元进行稀疏）；全连接层的稀疏正在开发中。同时，暂时只支持 ARM CPU （例如高通系列，瑞芯微系列）上的稀疏推理。
