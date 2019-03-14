# paddle-mobile 设计文档


#### 以下是 paddle-mobile 代码的执行流程图:

![执行流程图](http://mms-graph.bj.bcebos.com/paddle-mobile/git_images/flow_chart.png)


#### 主要分为: Loader 模块、 Program 模块、 Executor 模块、 op 模块、 kernel 模块、scope variable Tensor 模块

#### 下面展开说一下各个模块的作用以及设计思路

### 一. Loader
先来看一下模型, 模型分为两种结构:
 一种为参数文件是散开的, 如下图, 红框为模型结构的 protobuf 文件, 其余为参数文件

![模型描述](http://mms-graph.bj.bcebos.com/paddle-mobile/git_images/model_desc.png)


另一种为参数文件结合在一起的, 如下图, 红框内为模型结构描述的 protobuf 文件, 另一个文件为结合在一起的参数文件

![模型描述combined](http://mms-graph.bj.bcebos.com/paddle-mobile/git_images/model_desc_combined.png)


loader 模块的作用是将模型结构信息 load 进内存, 将红框内的 protobuf 文件 load 进内存, 并对模型结构进行优化(如将几个细粒度的 op 融合成 粗粒度的 op, 如将 conv、 add、 batchnorm、 relu 融合为 conv\_add\_batchnorm\_relu).
方便进行算法优化.

__那么为什么融合在一起能够做算法优化 ?__

如果未融合的 conv add batchnorm relu 运算是这样的

```
[n]
[conv_res] = conv([n])

for &res in conv_res {
	res = add_biase(res)
}

for &res in conv_res {
	res = batchnorm(res)
}

for &res in conv_res {
	res = relu(res)
}

```
融合后的 conv\_add\_batchnorm\_relu 运算是这样的:

```
[n]
[conv_res] = conv([n])

for &res in conv_res {
	res = relu(batchnorm(add_biase(res)))
}

```
由于 conv 可以转换为两个大矩阵相乘, 更进一步可以分为若干个一行一列的小矩阵相乘, 那最终的运算是这样的:

```
[n]
for &res in [res] {
	res = relu(batchnorm(add_biase(A * B)))
}

其中 A 和 B 为 1 * k 和 k * 1 矩阵

```



### 二. Program

program 为 loader 模块的结果, 包含了优化前的模型结构对象, 以及优化后的模型结构对象, 此模块基本对应着 paddle 模型的结构, 关于paddle 模型的一些概念的定义, 详细设计可以参考 [program.md](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md), 以下是一个简单的概况: 

* programDesc 中包含着若干个(googlenet mobilenet yolo squeezenet resnet 常见的模型只有一个)可以嵌套的 block, blocks中的第一个block中的某个 op 可能会执行 blocks 中后边 block 中的一系列 op 运算(只有多个block才会有此概念)
* block 包含着 ops 和 vars
* ops 为一系列 op 的描述, 描述着每个 op 的类型, 输入输出, 所需参数
* vars 里包含的为所有 op 运算所需的参数描述

### 三. Executor

executor 主要是用于 op 运算的上层调度操作, 主要有两个操作,  executor 实例化 和 暴露给上层的 predict 方法

* executor 实例化过程中, 主要进行了这几个操作 
	1. 根据 loader 产出的 program 初始化 operator 对象 
	2. 分配所有需要用到的内存, 包括每个op 的输入输出, 权重参数, 目前模型的权重参数文件的内存格式为 NCHW, op 的输入输出中间矩阵参数也是 NCHW 格式
	3. 调用每个 op 的 init 方法, init 方法是每个 op 实现者进行参数预处理的地方, 有助于减少 predict 的耗时

* predict, 主要用于拿到外部的输入, 顺序调用 op 的 run 方法进行运算, 并返回最终的结果.


### 四. op
关于 op 模块代码的详细设计可以参考 [operator部分代码设计](https://github.com/PaddlePaddle/paddle-mobile/issues/300), operator主要包含一个kernel用于运算、一个 param 用于存储属性, operator 主要有三个操作, Init、RunImp、InferShape

* Init: Init 函数主要用于参数预处理, 如对 batchNorm 参数进行预处理, 可以将 batchNorm 运算转化为 a * x + b 形式的运算, 这个函数也会调用, kernel 的 Init 函数对 kernel 进行初始化
* RunImp: RunImp 函数会调用自己的kernel 的 compute 方法进行运算
* InferShape: InferShape 函数会根据输入和参数得出输出的形状, 这个函数会在 executor 实例化时, 内存初始化前调用

每个 operator 都需要进行注册才可以被使用, 以 conv 为例, 需在 conv_op.cpp 底部这样写: 

```c++
// 三个平台都注册了 conv op
namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
USE_OP_CPU(conv2d);
REGISTER_OPERATOR_CPU(conv2d, ops::ConvOp);
#endif

#ifdef PADDLE_MOBILE_FPGA
USE_OP_FPGA(conv2d);
REGISTER_OPERATOR_FPGA(conv2d, ops::ConvOp);
#endif

```

__一个关于包大小的优化__:

每个 operator 都由一个宏控制编译, 如 conv_op.h(除了 conv_op.h ,  conv_op.cpp、conv_kernle.h、conv_kernle.cpp 也都需要加此宏控制)

```c++

#ifdef CONV_OP    //这个宏控制着 conv_op 是否被编译, 除了 conv_op.h ,  conv_op.cpp、conv_kernle.h conv_kernle.cpp 也都需要加此宏控制

#pragma once

#include <string>
#include "framework/operator.h"
#include "operators/kernel/conv_kernel.h"

namespace paddle_mobile {
namespace operators {
using std::string;
template <typename DeviceType, typename T>
class ConvOp
	//impl  
};

}  // namespace operators
}  // namespace paddle_mobile

#endif

```
这样做的目的是为了根据不同类型的网络编译特定的op, 在 cmake 中已经配置好不同网络编译的宏, 如果你要进行编译支持 yolo 的模型, 仅需执行:

```sh
cd toools
sh build.sh android yolo

```
这样只会编译 yolo 所包含的四种 op, 极大的减小了包体积和编译时间

### 五. kernel
kernel 为 op 的底层运算实现, 主要有两个函数, Init 和 Compute, 分别用来初始化、预处理 和 运算操作, 值得提出的是, kernel 会根据泛型特化到不同的平台, 如图所示:

![设备特化](http://mms-graph.bj.bcebos.com/paddle-mobile/git_images/devices.png)

不同平台的 kernel 实现, 为同一个 kernel 类不同泛型的特化实现, 目前有三个平台, arm、mali、fpga, 图中的 central-arm-func\ 目录为 op kernel 的 arm 实现, 它承担了 arm\ 目录下 kernel 的底层实现, 同时 arm 处理器作为中央处理器, central-arm-func\ 也可以作为其他协处理器的底层实现, 如: fpga 的某一个 op kernel 还没有 fpga 协处理器的实现, 就可以直接调用使用这里的 arm 实现.

__如果你有兴趣新增一个协处理器实现, 就可以在次添加一个 kernel 目录, 提供协处理器实现, 如果某个 kernel 你没有实现完, 你也可以直接使用 arm 实现__

### 六. scope variable Tensor
* scope 用来存储管理所需用到的所有 variable(用来存储不同类型的对象, 主要是矩阵Tensor, 也就是说 scpoe 管理着 op 运算过程中所有参数矩阵, 输入输出矩阵), 可以将 scope 理解为一个 map, 这里在 map 上封了一层 scope 的概念是为了方便内存管理
* variable 可以用来存储不同类型的对象, paddle-mobile 里主要用它来存储矩阵 Tensor
* tensor 代表着矩阵, 通过泛型可以用来存储不同类型的矩阵, 但需要注意的是, 存入和取出时的类型必须保持一致, 如果类型不一致,  使用 inline const T \*data() const 获取指针会不能通过类型检查, 通过  inline T \*mutable_data() 获取指针会重新分配内存, 以下是关于 Tensor 的一些小概念:
	1. DDim: 用来存储矩阵的维度信息.
	2. Slice(): 这个函数用来获取 N 维 (NCHW中的 N) 上切片
	3. 当实例化未分配内存时, 调用 inline T *mutable_data() 会分配内存






