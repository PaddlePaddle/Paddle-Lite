# 架构详解

这篇文档会从开发者角度详细介绍开发 Paddle-Lite 需要的相关信息。

## 设计及思考

近年来，各种深度学习预估硬件称出不穷，从手机APP到车载设备，再到音箱，均需要部署深度学习预测，且有如下共性需求：

1. 高性能
2. 硬件支持和扩展容易
3. 轻量级部署

Paddle-Lite 的架构方面便是定向参考如上需求设计实现的，具体地

- 高性能方面
  - 通过 MIR(Machine IR) 实现精细复杂的计算图的分析和优化
  - 执行期 Kernel 的简单设计，几乎没有额外调度开销
  - 适当的硬件层抽象，框架支持各个硬件后端中做特定的调度实现
- 轻量级部署方面
  - 拆分分析和执行两个阶段，执行阶段轻量级实现，可以单独部署
  - 轻量级 Op 和 Kernel 设计
- 硬件支持和扩展方面
  - 通过 MIR 支撑带硬件和执行信息的宏观分析优化
  - TypeSystem 抽象带硬件的不同计算模式的表示，实现整个计算图的强类型推导，以及执行状态机的静态分析

Paddle-Lite 的架构尝试从强类型推导的角度建模支持多硬件，多种计算模式（不同量化精度、不同的 data layout等）的混合计算，从而实现宏观上的各异硬件和计算模式的混合。

框架部分已经经过 FPGA，GPU，NPU 等异构硬件的打磨，各项能力也在完善中。

## 重要模块介绍

### OpLite

[OpLite](https://github.com/PaddlePaddle/Paddle-Lite/blob/v2.0.0-beta1-prerel/lite/core/op_lite.h#L52) 是 Paddle-Lite 中的 Operator，用户扩展单个硬件时，最多的就是扩展 Op 和 Kernel。

重要方法如下：

```c++
class OpLite : public Registry {
 public:
  // Check the shape.
  virtual bool CheckShape() const { return true; }
  // Inference the outputs' shape.
  virtual bool InferShape() const { return true; }
  // Link the external execution environ to internal context.
  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope);
};
```

其中，分析期执行

- `AttachImpl`

执行期执行

- `CheckShape`
- `InferShape`

扩展须知：

1. `CheckShape` 只在第一个 batch 执行，所以耗时不敏感

2. `InferShape` 需要在每个 batch 执行，应该严格耗时

   1. 可以通过添加 member variable 的方式，对其中一部分信息增加 cache，比如

   ```c++
   class XXOp : public OpLite {
       void InferShape() {
           int batch_size = param().input.shape[0];
           if (!shape_cache_.empty()) {
               shape_cache_[0] = batch_size;
               param().output->Resize(shape_cache_);
           }
       }
       
    private:
       shape_t shape_cache_;
   }
   ```

   

### OpParam

[OpParam](https://github.com/PaddlePaddle/Paddle-Lite/blob/v2.0.0-beta1-prerel/lite/operators/op_params.h) 用于存储执行期 Kernel 需要的各项参数。 所有字段可以直接存储（比如指针或者 `int`），以避免执行中获取参数的延迟。

因为没有需求，OpParam 暂时没有设置基类。

实际例子：

```c++
// For Softmax op
struct SoftmaxParam {
  lite::Tensor* x{};
  lite::Tensor* output{};
  int axis{-1};
};
```

OpLite 的 `AttachImpl` 方法就用于构建 `OpParam` ，复制传递给 `Kernel` 用于执行。

OpParam  是执行期的重要模块，需要严格保证性能，相应的扩展要求：

1. 字段的获取必须是低延迟的，可以直接用指针，或者直接复制值
2. 避免执行无关信息混入，包括 debug 信息
3. 命名需要与 Paddle OpDesc 中的信息严格一致，以降低功能对齐和理解的难度

### Kernel

```c++
template <TargetType Target,
          PrecisionType Precision,
          DataLayoutType DataLayout = DataLayoutType::kNCHW>
class KernelLite : public KernelBase {
 public:
  // Run the kernel.
  virtual void Run() { CHECK(false) << "Not Implemented"; }

  TargetType target() const override { return Target; }
  PrecisionType precision() const override { return Precision; }
  DataLayoutType layout() const override { return DataLayout; }
  Place place() const override { return Place{Target, Precision, DataLayout}; }
  std::string name() const override;
};
```

由于是执行期的重要概念，因此 Kernel 设计地非常简单高效。 

其中，执行期的 `Run` 是其唯一重要的接口，其中包含具体的计算逻辑。

模板中的参数主要用于方便多硬件编译，以及自解释：

- Target: 执行硬件
- Precision: 主要的计算精度
- DataLayout：主要计算的 data layout

这部分信息用于帮助挑选 kernel，具体的值并不严格。



Kernel 的注册需要用到 TypeSystem，不光对 Kernel 本身的特性进行描述，对其输入和输出均进行详尽的定义。

例如 FullyConnected 的注册

```c++
REGISTER_LITE_KERNEL(
    fc, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::FcCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat), LAYOUT(kNCHW))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
```

Kernel自身定义是 `kARM` 的，也就是ARM上的kernel，主要的计算精度是 `kFloat`，主要的 Data layout 是 `kNCHW`。

接着会对其所有的输入和输出做详细定义，比如看 `Input` 输入的定义是 `LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat), LAYOUT(kNCHW))`，也就是声明其 Target 是 `kARM`， PRECISION 是 `kFloat`，Data Layout 是 `kNCHW`。

这里的设计思想是类似C++中的函数重载，同一个 Kernel（的名字），在重载了其输入输出的类型之后可以是不同的kernel。

#### 扩展须知

1. 模板参数选用计算中主要的来表示
   1. 比如，scale kernel，同时能接受 `float` 和 `int` 的输入，但其不算量化 kernel，那应该设置为 `Precision=float`，代表常规的计算精度中使用
2. Kernel 输入输出的定义需要足够精确，是什么类型就是什么类型；框架会根据其输入输出的定义来动态构建状态机，否则会出现分析期和执行期的状态机不一致，造成未定义行为

### MIR

MIR 类似于 LLVM 里的 IR，只是加上了硬件和执行期的信息参与分析优化。

Pass 是MIR中的模块化策略，其输入和输出都是 SSA Graph.

框架会自动基于模型的Program 构建 SSA Graph，之后按 [Optimizer](https://github.com/PaddlePaddle/Paddle-Lite/blob/v2.0.0-beta1-prerel/lite/core/optimizer.h) 中定义的pass的顺序调用一系列 Pass。

#### Op Fusion

MIR 中的 [PatternMacher](https://github.com/PaddlePaddle/Paddle-Lite/blob/v2.0.0-beta1-prerel/lite/core/mir/pattern_matcher.h) 实现了简单有效的基于图的模板识别的算法，相关的 op fusion 的图操作可以基于此实现。

实际的例子可以参考 [fc_fuse_pass.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/v2.0.0-beta1-prerel/lite/core/mir/fusion/fc_fuse_pass.h)。

### TypeSystem

TypeSystem 是 Paddle-Lite 中构建复杂计算图的基础模块，核心思想是协助 SSA Graph 构建一个状态机，表示其中不同的状态。

这里的 Type 主要包含下面四组信息，更多的信息可以按需扩展：

- TargetType
- Precision
- DataLayout
- device id，用于表示卡号



状态机的表示：

```python
Tensor0(kARM, kFloat, kNCHW) --pass--> Tensor1(kOpenCL, kFloat, kNCHW)
```

MIR 会识别出，Tensor0 和 Tensor1 的硬件位置不同，因此触发相依的 Pass 插入对应的 cast op 来进行 type cast，比如

```
Tensor0(kARM, kFloat, kNCHW) --pass-> IoCopyOp(kARM, kOpenCL) --pass-> Tensor1(kOpenCL, kFloat, kNCHW)
```

### KernelContext

KernelContext 是硬件支持的核心封装，主要用于为 Kernel 提供执行期的硬件上下文。

KernelContext 的设计类似于 OpParam，两者均没有基类；对于 KernelContext，其假定是，不同的硬件间的接口和逻辑可能完全不同，比如 kARM 和 kCUDA，因此不设定基类，也不需要提供统一的接口来封装不同硬件行为。

不同硬件的 KernelContext 直接与该硬件对应的 Kernel 对接。

KernelContext 的行为可以被 MIR 在分析期确定和调度。

注意事项：

1. 由于是执行期概念，KernelContext 也需要注意性能和轻量化
2. 移动端部署时只会部署执行期，因此 MIR 和 KernelContext 会拆开，因此 KernelContext 相应的设置需要能够序列化到 ProgramDesc 中，以便执行期载入和执行

## 扩展硬件后端

### 扩展现有的硬件后端

主要是扩充 Op 和 Kernel 的工作，如果需要 fuse，则参考 MIR 章节，增加相应的fuse pass便可，具体地，可以参考

- [fc_op](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/operators/fc_op.h) 实现类似的 Op
- [fc_compute](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/kernels/arm/fc_compute.h) 实现类似的 Kernel
- [fc_fuse_pass](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/core/mir/fusion/fc_fuse_pass.h) 实现fuse逻辑，并注册到 [optimizer](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/core/optimizer.h)

### 扩展全新硬件后端

需要额外扩充如下模块，让框架能够支撑硬件执行：

- TypeSystem，需要扩充其中相关的 type
  - 相关 [enum](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/api/paddle_place.h#L44)
- MIR，需要扩展其中的 type cast 相关的 pass
  - [TargetType cast pass](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/core/mir/type_target_cast_pass.cc) 用于拷贝不同硬件上的tensor
  - [Data layout cast pass](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/core/mir/type_target_cast_pass.h) 用于转化不同的 data layout
  - [Precision cast pass](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/core/mir/type_precision_cast_pass.h) 用于转化不同 tensor 的量化精度
- KernelContext，具体地可以参考
  - [ARM context](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.0.0-beta1/lite/core/context.h#L91)
  - 需要注意的是，硬件 context 的接口只服务于该硬件的 kernel
  - context 有分析期和执行期两个阶段，如果分析期没有特殊的优化，则无需考虑；否则，需要注意将分析期的信息整理并序列化到离线模型中，用于执行期直接加载。
