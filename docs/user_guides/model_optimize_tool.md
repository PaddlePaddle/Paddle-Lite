
# 模型优化工具 opt

Paddle Lite 提供了多种策略来自动优化原始的训练模型，其中包括量化、子图融合、混合调度、Kernel 优选等等方法。为了使优化过程更加方便易用，我们提供了**opt** 工具来自动完成优化步骤，输出一个轻量的、最优的可执行模型。

具体使用方法介绍如下：

## opt 安装和使用方法
- 安装方法
  - 环境要求：`windows\Mac\Ubuntu`
  - 环境依赖： 
    - python == `2.7\3.5\3.6\3.7`
    - pip
```bash
# 当前最新版本是 2.9
pip install paddlelite==2.9
```
- `opt`转化和分析模型： 可通过**终端命令**或**Python脚本**调用
    - [终端命令方法](./opt/opt_python) （支持`Mac/Ubuntu`)
    - [ python 脚本方法](../api_reference/python_api/opt)（支持`Window/Mac/Ubuntu`）


#### 源码编译 opt 工具
您也可以选择从源代码编译 opt 工具，使用编译指令
```shell
./lite/tools/build.sh build_optimize_tool
```

如果在 arm64 架构的 MacOS 下编译 opt 工具失败，试着删除 third-party 目录并重新`git checkout third-party`，然后将上一条指令改为
```shell
arch -x86_64 ./lite/tools/build.sh build_optimize_tool
```
该命令会编译 x86 格式的 opt 工具，但是不会影响工具的正常使用，编译成功后，在./build.opt/lite/api目录下，生成了可执行文件 opt

 [使用可执行文件 opt 工具](./opt/opt_bin)
## 合并 x2paddle 和 opt 功能的一键脚本

**背景**：如果想用 Paddle Lite 运行第三方来源（tensorflow、caffe、onnx）模型，一般需要经过两次转化。即使用 x2paddle 工具将第三方模型转化为 PaddlePaddle 格式，再使用 opt 将 PaddlePaddle 模型转化为 Padde Lite 可支持格式。
为了简化这一过程，我们提供了：

 [合并 x2paddle 和 opt 的一键脚本](./opt/x2paddle&opt)
