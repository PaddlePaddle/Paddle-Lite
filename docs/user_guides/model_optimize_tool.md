
# 模型优化工具 opt

Paddle-Lite 提供了多种策略来自动优化原始的训练模型，其中包括量化、子图融合、混合调度、Kernel优选等等方法。为了使优化过程更加方便易用，我们提供了**opt** 工具来自动完成优化步骤，输出一个轻量的、最优的可执行模型。

具体使用方法介绍如下：

## opt 下载和使用方法
### 方法一： 通过Python 安装和调用 opt 工具
- 支持环境：`windows\Mac\Ubuntu`
- 安装方法: 通过`pip`工具安装`Paddle-Lite`到`Python`
```bash
# 当前最新版本是 2.8rc0
pip install paddlelite==2.8rc0
```
- 使用`opt`转化和分析模型
    - 方法一： [使用终端命令](./opt/opt_python) （支持Mac/Ubuntu)
    - 方法二： [使用python脚本](../api_reference/python_api/opt)（支持window/Mac/Ubuntu）


### 方法二： 下载和调用 opt 可执行文件
- 支持环境：`Mac\Ubuntu`
- 安装方法: 直接下载可执行文件
从[release界面](https://github.com/PaddlePaddle/Paddle-Lite/releases)或[预测库下载界面](../quick_start/release_lib)下载与预测库版本一致的`opt`工具
- 使用`opt`转化和分析模型
    - 方法：[直接下载并执行opt可执行工具](./opt/opt_bin)（支持Mac/Ubuntu)


## 合并x2paddle和opt的一键脚本

**背景**：如果想用Paddle-Lite运行第三方来源（tensorflow、caffe、onnx）模型，一般需要经过两次转化。即使用x2paddle工具将第三方模型转化为PaddlePaddle格式，再使用opt将PaddlePaddle模型转化为Padde-Lite可支持格式。
为了简化这一过程，我们提供了：

 [合并x2paddle和opt的一键脚本](./opt/x2paddle&opt)
