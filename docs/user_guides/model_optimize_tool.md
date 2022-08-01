
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
# 版本号需高于或等于1.3.3
pip install x2paddle
```
- `opt`转化和分析模型： 可通过**终端命令**或**Python脚本**调用
    - [终端命令方法](./opt/opt_python) （支持`Mac/Ubuntu`)
    - [python 脚本方法](../api_reference/python_api/opt)（支持`Window/Mac/Ubuntu`）


#### 源码编译 opt 工具
您也可以选择从源代码编译 opt 工具，使用编译指令
```shell
./lite/tools/build.sh build_optimize_tool
```

如果在 arm64 架构的 MacOS 下编译 opt 工具失败

- 方法1: 试着删除 third-party 目录并重新`git checkout third-party`，然后将上一条指令改为:

```shell
arch -x86_64 ./lite/tools/build.sh build_optimize_tool
```
  该命令会编译 x86 格式的 opt 工具，但是不会影响工具的正常使用，编译成功后，在./build.opt/lite/api目录下，生成了可执行文件 opt
- 方法2: 使用 `build_macos.sh` 脚本进行编译

```shell
./lite/tools/build_macos.sh build_optimize_tool
```

[使用可执行文件 opt 工具](./opt/opt_bin)

## 使用 X2paddle 导出 Padde Lite 支持格式

**背景**：如果想用 Paddle Lite 运行第三方来源（TensorFlow、Caffe、ONNX、PyTorch）模型，一般需要经过两次转化。即使用 X2paddle 工具将第三方模型转化为 PaddlePaddle 格式，再使用 opt 将 PaddlePaddle 模型转化为Padde Lite 可支持格式。

**使用方法**：为了简化这一过程，X2Paddle 集成了 opt 工具，提供一键转换 API，以 ONNX 为例：

***API方式***
 ```python
from x2paddle.convert import onnx2paddle

onnx2paddle(model_path, save_dir,
            convert_to_lite=True,
            lite_valid_places="arm",
            lite_model_type="naive_buffer")
# model_path(str) 为 ONNX 模型路径
# save_dir(str) 为转换后模型保存路径
# convert_to_lite(bool) 表示是否使用 opt 工具，默认为 False
# lite_valid_places(str) 指定转换类型，默认为 arm
# lite_model_type(str) 指定模型转化类型，目前支持两种类型：protobuf 和 naive_buffer，默认为 naive_buffer
```

Notes:
- ```lite_valid_places```参数目前可支持 arm、 opencl、 x86、 metal、 xpu、 bm、 mlu、 intel_fpga、 huawei_ascend_npu、imagination_nna、 rockchip_npu、 mediatek_apu、 huawei_kirin_npu、 amlogic_npu，可以同时指定多个硬件平台(以逗号分隔，优先级高的在前)，opt 将会自动选择最佳方式。如果需要支持华为麒麟 NPU，应当设置为 "huawei_kirin_npu,arm"。

***命令行方式***
```shell
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model --to_lite=True --lite_valid_places=arm --lite_model_type=naive_buffer
```

TensorFlow、Caffe 以及 PyTorch 模型转换参考 [X2Paddle API](https://github.com/PaddlePaddle/X2Paddle/tree/develop/docs/inference_model_convertor/convert2lite_api.md)
