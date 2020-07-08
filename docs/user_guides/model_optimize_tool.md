
# 模型优化工具 opt

Paddle-Lite 提供了多种策略来自动优化原始的训练模型，其中包括量化、子图融合、混合调度、Kernel优选等等方法。为了使优化过程更加方便易用，我们提供了**opt** 工具来自动完成优化步骤，输出一个轻量的、最优的可执行模型。

具体使用方法介绍如下：

**注意**：`v2.2.0` 之前的模型转化工具名称为`model_optimize_tool`，从 `v2.3` 开始模型转化工具名称修改为 `opt`，从`v2.6.0`开始支持python调用`opt`转化模型（Windows/Ubuntu/Mac）

## 准备opt
当前获得`opt`工具的方法有三种：

- 方法一: 安装opt的python版本

安装`paddlelite` python库，安装成功后调用opt转化模型（支持`windows\Mac\Ubuntu`）

```bash
pip install paddlelite
```

- 方法二: 下载opt可执行文件
从[release界面](https://github.com/PaddlePaddle/Paddle-Lite/releases)，选择当前预测库对应版本的`opt`转化工具

本文提供`release/v2.6.1`和`release/v2.2.0`版本的优化工具下载

|版本 | Linux | MacOS|
|---|---|---|
| `release/v2.6.1` | [opt](https://paddlelite-data.bj.bcebos.com/Release/2.6.1/opt/opt) | [opt_mac](https://paddlelite-data.bj.bcebos.com/Release/2.6.1/opt/opt_mac) |
|`release/v2.2.0`  | [model_optimize_tool](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/model_optimize_tool) | [model_optimize_tool_mac](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/model_optimize_tool_mac) |

- 方法三: 源码编译opt
源码编译 opt 可执行文件

```
cd Paddle-Lite && ./lite/tools/build.sh build_optimize_tool
```

编译结果位于`build.opt/lite/api/`下的可执行文件`opt`

## 使用opt

当前使用`opt`工具转化模型的方法有以下三种：

- 方法一： [安装 python版本opt后，使用终端命令](./opt/opt_python) （支持Mac/Ubuntu)
- 方法二： [安装python版本opt后，使用python脚本](../api_reference/python_api/opt)（支持window/Mac/Ubuntu）
- 方法三：[直接下载并执行opt可执行工具](./opt/opt_bin)（支持Mac/Ubuntu)
- Q&A：如何安装python版本opt ?

可以通过以下命令安装paddlelite的python库(支持`windows/Mac/Ubuntu`)：
```shell
pip install paddlelite
```



## 合并x2paddle和opt的一键脚本

**背景**：如果想用Paddle-Lite运行第三方来源（tensorflow、caffe、onnx）模型，一般需要经过两次转化。即使用x2paddle工具将第三方模型转化为PaddlePaddle格式，再使用opt将PaddlePaddle模型转化为Padde-Lite可支持格式。
为了简化这一过程，我们提供了：

 [合并x2paddle和opt的一键脚本](./opt/x2paddle&opt)
