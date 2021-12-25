# 新增 Pass

本文从三个方面介绍了 Lite 中的 Pass 结构：**Pass 是什么**、**Pass 的实现与接口**、**Pass 的一般注册流程**。最后以`Fc_fuse_pass`为例介绍了`fusion_pass`的作用与注册方法。

## 前述：Pass 是什么？

**CreatePaddlePredictor() 加载模型时，在执行预测前会先优化模型。模型优化过程是通过 Pass 实现的。**

具体调用关系如下：

- `CreatePaddlePredictor()`：暴露给用户的API。
  - `CxxPaddleApiImpl::Init()`。
    - 上层函数中通过`raw_predictor_->Build(config, places, passes)`调用`Predictor::Build()`。
      - 上层函数中继续调用`Predictor::Build()`的另一个重载函数。
        - `LoadModelPb()`加载模型文件到`program_desc_`中。
        - 调用`Predictor::Build()`的另一重载函数。
          - 调`RunDefaultOptimizer()`对`program_desc_`中的原始图形结构进行优化，对图结构的优化是通过调用 `Pass->Apply(const std::unique_ptr<SSAGraph>& graph)`实现的。

上面调用涉及到的几个函数都在`lite/api/cxx_api.cc`和`lite/api/cxx_api_impl.cc`中，其中`Predictor::Build()`存在3个同名函数，注意区分。

**每一类 Pass 定义了一种优化过程**，包括：原模型中的 kernel 选取、OP 融合、冗余 OP 去除、子图创建、内存优化、类型推导、类型转换等。


## Pass 的实现与接口 ：Pass 基类、PassManager 和 Pass 注册

### 1、Pass 基类：`paddle::lite::mir::Pass`

Pass 基类中的部分代码如下。

```c++
class Pass {
 public:
  // Pass的类型，Pass按照作用的不同可以分为三种
  enum class Kind {
    // 1. 修改模型中的图拓扑结构的Pass
    kProgramWise = 0,
    // 2. 不修改图结构，修改状态的Pass
    kStmtWise,     
    // 3. 不修改 IR，用于搜集信息和可视化信息的Pass.
    kDebug,
  };
  
  explicit Pass(Kind kind) : kind_(kind) {}

  // 主要实现函数：Apply 函数定义了 Pass 运行时执行的操作
  virtual void Apply(const std::unique_ptr<SSAGraph>& graph) = 0;

  bool is_program_pass() const { return kind_ == Kind::kProgramWise; }
  bool is_stmt_pass() const { return kind_ == Kind::kStmtWise; }

  virtual ~Pass() = default;

 private:
  const Kind kind_;  // pass 的种类
  std::string name_; // pass 的名称
  std::set<TargetType> bound_targets_; // 指定了Pass运行的硬件平台，模型优化过程会根据当前硬件平台是否匹配筛选Pass。
  std::unordered_map<std::string, std::set<lite_api::Place>> bound_kernels_; // 绑定的kernel
};


// Different kinds.
class ProgramPass : public Pass {
 public:
  ProgramPass() : Pass(Kind::kProgramWise) {}
};
class StmtPass : public Pass {
 public:
  StmtPass() : Pass(Kind::kStmtWise) {}
};

class DebugPass : public Pass {
 public:
  DebugPass() : Pass(Kind::kDebug) {}
};
```
**完整代码位置**：`lite/core/optimizer/mir/pass.h`

**主要类成员**：

- `const Kind kind_` : Pass 类型。pass 有三种基本基本类型 ：修改图结构的`ProgramPass`、修改状态量的`StmtPass`和 Debug 过程采集信息与控制可视化的`DebugPass`。
- `std::string name_` ：Pass 的名称。
- `std::set<TargetType> bound_targets_` : Pass 运行的硬件平台，优化过程会根据硬件平台选择匹配的 Pass 。
- `std::unordered_map<std::string, std::set<lite_api::Place>> bound_kernels_` : Pass 绑定的 kernel 。

**主要接口**：

- `Pass::Apply(const std::unique_ptr& graph)` : Pass 优化过程的具体操作，是新注册 Pass 需要实现的接口。输入为`SSAGraph`型指针，是对模型结构的拓扑表示。

### 2、Pass 管理 `paddle::lite::mir::PassManager` 

```c++
class PassManager {
 public:
  // 内部静态变量PassManager，用来存储使用的Pass和图优化操作
  static PassManager& Global() {
    static PassManager x;
    return x;
  }

  // 执行所有的 Pass 
  void Run(const std::unique_ptr<SSAGraph>& graph) {
    for (auto& pass : passes_) {
      LOG(INFO) << "Running MIR pass " << pass->name();
      pass->Apply(graph);
    }
  }

  bool AddNewPass(const std::string& name, Pass* pass) {
    passes_.emplace_back(pass);
    pass_map_.emplace(name, passes_.back().get());
    passes_.back()->set_name(name);
    return true;
  }

 private:
  std::list<std::unique_ptr<mir::Pass>> passes_; //存储所有的 Pass
  std::map<std::string, mir::Pass*> pass_map_; //使用map变量存储 PassName::Pass
};
```
**完整代码位置**：`lite/core/optimizer/mir/pass_manager.h`。

**主要类成员**：

- `std::list<std::unique_ptr<mir::Pass>> passes_;`  : List 类型，存储了所有已注册 Pass 。
- `std::map<std::string, mir::Pass*> pass_map_; `  :   Map 类型，存储了所有" Pass 名称 - Pass 类"键对，用于根据名称查找 Pass 。

**主要接口**：

- `static PassManager& Global()` 返回 PassManager 全局静态变量,该变量存储了所有已注册的 Pass。
- ` bool AddNewPass(const std::string& name, Pass* pass)` 添加新的 Pass 到 PassManager 中。


### 3、 Pass 注册 `paddle::lite::mir::PassRegistry`

**代码位置**：`lite/core/optimizer/mir/pass_registry.h`。

**主要接口**：

- `REGISTER_MIR_PASS(name__, class__)` ：宏定义函数，用于注册 Pass 。它实现的是`PassManager::Global().AddNewPass(name__, class__)`，将新注册 Pass 添加到全局变量`PassManager`中。



## Pass 的一般注册流程与使用方法

### 1. Pass 注册流程

在`lite/core/optimizer/mir`或其子目录下继承`Pass`基类，实现`Pass::Apply`接口，并使用宏`REGISTER_MIR_PASS(name__, class__)`将 Pass 注册到`PassManager`即完成了新 Pass 注册。

以新建 `example_pass`为例，具体流程如下：

（1）在`lite/core/optimizer/mir`路径下新建`example_pass.cc` 和 `example_pass.h` 文件。

（2）在`example_pass.h` 文件中继承 Pass 基类（ ProgramPass 、StmtPass 或 DebugPass ）定义自己的 Pass 类。

```c++
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {
class ExamplePass : public ProgramPass {
  void Apply(const std::unique_ptr<SSAGraph> &graph) override {}
   ...
};
}  // namespace mir
}  // namespace lite
}  // namespace paddle
```

（3）在`example_pass.cc` 文件中实现`ExamplePass::Apply()`接口，并注册`ExamplePass`。

```c++
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/example_pass.h"

namespace paddle {
namespace lite {
namespace mir {
void ExamplePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
    ...
}
}  // namespace mir
}  // namespace lite
}  // namespace paddle
REGISTER_MIR_PASS(example_pass, paddle::lite::mir::ExamplePass)
    .BindTargets({TARGET(kARM), TARGET(kX86)}); // Pass执行的目标硬件平台
    // .BindKernel("conv2d");     //Pass绑定的 kernel
```

（4）`lite/core/optimizer`下的所有`*.cc`文件会被自动地在`lite/core/CMakeLists.txt`中被编译。

```cmake
# optimizer source code
FILE(GLOB_RECURSE OPTIMIZER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/*.cc)
LIST(REMOVE_ITEM OPTIMIZER_SRC ${UNIT_TEST_SRC})
```

上面 cmake **代码位置**：`lite/core/CMakeLists.txt`。


### 2. Pass 使用流程

将 Pass 注册到 PassManager 后不会自动生效。需要在`optimizer->run()` 函数中添加该 Pass 才会在模型优化过程中调用。

（1）在`paddle_use_passes.h`文件中调用该 Pass。

```c++
#pragma once
#include "paddle_lite_factory_helper.h"  // NOLINT
    ...
USE_MIR_PASS(example_pass);  //调用 example_pass
```

（2）要想在优化模型时调用该 Pass ，需要在`optimizer->run()`函数中手动添加调用。

修改`lite/core/optimizer/optimizer.cc`文件，添加`example_pass`到`RunDefaultOptimizer`函数中的局部变量  passes_local 中。

（3）只有 `CreatePaddlePredictor` 才会在模型加载时根据 Pass 优化模型。

`lite/demo/cxx/armlinux_mobilenetv1_full_demo/mobilenet_full_api.cc`文件中有类似下面这段代码。

```c++
 ...
#include "paddle_use_passes.h"   // 引用Pass优化模型
void RunModel() {
  // 1. 创建 CxxConfig
  CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places(Place{TARGET(kARM), PRECISION(kFloat)});

  // 2. 创建CxxPredictor,该过程包括加载模型和用Pass优化模型
  std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<CxxConfig>(config);
}
```


## Fusion Pass 的定义与注册

`Fusion Pass`是一种常见图结构优化 Pass ，可将多个连续 OP 融合成单个等效 OP ，减少数据交换并简化图结构。Pass 运行时调用`Fuser`自动查找并替换指定图结构，所以注册`FuserPass`时还需要实现对应的 Fuser 类。

下面以`fc_fuse_pass`为例，详细说明`FusionPass`的效果和注册方法。

### `fc_fuse_pass`的作用
将相邻的`mul`算子和 `element_wise add`算子 融合成一个 `FC`  算子。
```c++
mul(X) =  X * W 
elementwise_add( mul(x) ) = X * W + Bias
//----------> after fusion
FC(X) = X * W +Bias
```

Pass 运行效果如下：
![图片](https://user-images.githubusercontent.com/45189361/69639193-12383100-1097-11ea-9063-21f030414080.png)
mul 和 elementwise_add 的原有参数映射到 FC 的参数上：
![图片](https://user-images.githubusercontent.com/45189361/69638836-74446680-1096-11ea-9cdc-a961fa995dfe.png)

### `fc_fuse_pass`的注册方法
#### 1、创建 FcFuser

（1）在`lite/core/optimizer/mir/fusion`路径下新建`fc_fuser.cc` 和 `fc_fuser.h` 文件。

（2）在`fc_fuser.h` 文件中继承`FuseBase`定义自己的 Fuser 类。

```c++
#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class FcFuser : public FuseBase {
 public:
  explicit FcFuser(bool with_relu) : with_relu_(with_relu) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
  bool with_relu_;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
```
**主要接口**：
`FuseBase::BuildPattern` ：  描述需要替换位置的图结构 (pattern)，Fuser 运行时会自动查找并替换该 pattern 。
`FuseBase::GenOpDesc` ：       创建融合后的等效 Fused_op 。
`FuseBase::InsertNewNode` ：用 Fused_op 替换原始图结构（pattern）。

对于 `FcFuser`：BuildPattern 描述的Pattern是`mul+elementwise add`，GenOpDesc 创建的 FC_op，InsertNewNode 函数的效果是用新建的`FC_op`替换模型中的`mul+elementwise add` pattern。


（3） 在`fc_fuser.cc`文件中实现 `BuildPattern()` 、`GenOpDesc()`、`InsertNewNode() `接口。

下面以 FcFuser 为例介绍三种接口的实现：

```c++
// 1. BuildPattern函数，描述需要替换的图结构
// FcFuser::BuildPattern() 描述了 mul + element_wise add 的图结构
void FcFuser::BuildPattern() {
  // x和W是 mul OP的输入
  // b是elementwise_add OP的一个输入
  auto* x = VarNode("x")->assert_is_op_input("mul", "X");
  auto* W = VarNode("W")->assert_is_op_input("mul", "Y");
  auto* b = VarNode("b")->assert_is_persistable_var();
  // mul OP 和 mul OP的输出
  // 注意，Op对应的是OpNode，变量对应的是VarNode
  auto* mul = OpNode("mul", "mul");
  auto* mul_out = VarNode("mul_out");
  // elementwise_add OP 和 elementwise_add OP的输出（最终输出）
  auto* add = OpNode("add", "elementwise_add");
  auto* Out = VarNode("Out");

  //（2） 描述拓扑连接 （Fuse之前mul 和elementwise_add的连接）
  // 对于mul这个OP，输入是W和x两个变量，然后连接到mul这个Op，紧接着这个连接到mul_out这个变量
  // 对于elementwise_add OP ，一个输入是mul_out，另一个是b这个变量
  std::vector<PMNode*> mul_inputs{W, x};
  std::vector<PMNode*> add_inputs{mul_out, b};
  mul_inputs >> *mul >> *mul_out;

  // Some op specialities.
  // (3） 声明新的拓扑结构中将会被移除的节点，包括被fuse的OP和OP之间的中间变量
  mul_out->AsIntermediate();
  mul->AsIntermediate();
  add->AsIntermediate();

  if (with_relu_) {
    auto* add_out = VarNode("add_out");
    auto* relu = OpNode("relu", "relu");
    std::vector<PMNode*> relu_inputs{add_out};
    add_inputs >> *add >> *add_out;
    relu_inputs >> *relu >> *Out;
    add_out->AsIntermediate();
    relu->AsIntermediate();
  } else {
    add_inputs >> *add >> *Out;
  }
}

// 2. GenOpDesc函数新建等效 Fused_op
// FcFuser::GenOpDesc() 新建了Fc_op
cpp::OpDesc FcFuser::GenOpDesc(const key2nodes_t& matched) {
  // (1) 得到第一个OP节点的 OpDesc ，并清空输入输出信息
  auto op_desc = *matched.at("mul")->stmt()->op_info();

  // Get the input scale from mul
  std::vector<float> x_scale_vct;
  std::vector<float> y_scale_vct;
  auto input_x_name = op_desc.Input("X").front();
  auto input_y_name = op_desc.Input("Y").front();
  bool is_quantized_op = op_desc.HasInputScale(input_x_name) &&
                         op_desc.HasInputScale(input_y_name);
  if (is_quantized_op) {
    x_scale_vct = op_desc.GetInputScale(input_x_name);
    y_scale_vct = op_desc.GetInputScale(op_desc.Input("Y").front());
  }

  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  // (2) 修改OpDesc , 将OpType设置为 "fc" (FC OP 的OP_type)，
  op_desc.SetType("fc");
  // (3) 设置OpDesc中的Input、Output、Attrbute。分别连接到BuildPattern（）函数中创建的VarNode
  op_desc.SetInput("Input", {matched.at("x")->arg()->name});
  op_desc.SetInput("W", {matched.at("W")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("b")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
  op_desc.SetAttr(
      "in_num_col_dims",
      matched.at("mul")->stmt()->op_info()->GetAttr<int>("x_num_col_dims"));
  if (with_relu_) {
    op_desc.SetAttr("activation_type", std::string{"relu"});
  }

  // Set the input scale into fc
  if (is_quantized_op) {
    op_desc.SetInputScale(matched.at("x")->arg()->name, x_scale_vct);
    op_desc.SetInputScale(matched.at("W")->arg()->name, y_scale_vct);
  }

  return op_desc;
}

// 3. InsertNewNode函数用Fused OP 替换模型图中的原始 Pattern
// FcFuser::InsertNewNode() 用Fc_OP替换原始模型图中的  " mul + element_wise add "
void FcFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  // (1) 创建FC OP的参数（OpDesc）
  auto op_desc = GenOpDesc(matched);
  // 创建一个 FC OP
  auto fc_op = LiteOpRegistry::Global().Create("fc");
  
  // 找到原拓扑结构中的scope (作用域)和 valid_places （可支持设备类型）
  auto mul = matched.at("mul")->stmt()->op();
  auto* scope = mul->scope();
  auto& valid_places = mul->valid_places();
  
  // (2) 将 FC OP的 scope和 valid_places设置与fuse前相同，并在图中创建该节点（node）
  fc_op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);
  
  // (3) 将FC节点连接到输入输出（var_node）
  IR_NODE_LINK_TO(matched.at("W"), new_op_node);
  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("b"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}
```

#### 2、注册 fc_fuse_pass

（1）在`lite/core/optimizer/mir/fusion`路径下新建`fc_fuse_pass.cc` 和 `fc_fuse_pass.h` 文件。

（2）在`fc_fuse_pass.h` 文件中，继承`ProgramPass`定义`FcFusePass`。

```c++
#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

class FcFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
```

（3）在`fc_fuse_pass.cc` 文件中实现`FcFusePass::Apply()`接口，并注册`FcFusePass`

```c++
#include "lite/core/optimizer/mir/fusion/fc_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/fc_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void FcFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
#if defined(LITE_WITH_X86) || defined(LITE_WITH_CUDA)
#ifdef LITE_WITH_MLU
  fusion::FcFuser fuser(false);
  fuser(graph.get());
#else
  fusion::FcFuser fuser(true);
  fuser(graph.get());
#endif
#endif
  fusion::FcFuser fuser2(false);
  fuser2(graph.get());
#ifdef LITE_WITH_FPGA
  fusion::FcFuser fpga_fuser(true);
  fpga_fuser(graph.get());
#endif
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_fc_fuse_pass, paddle::lite::mir::FcFusePass)
    .BindTargets({TARGET(kAny)}) // FcFusePass 可以在任何硬件平台执行
    .ExcludeTargets({TARGET(kXPU)})
#if (!defined(LITE_WITH_MLU) && !defined(LITE_WITH_NNADAPTER) && \
      !defined(LITE_WITH_METAL))
    .ExcludeTargets({TARGET(kX86)})
#endif
    .ExcludeTargets({TARGET(kBM)})
    .BindKernel("fc"); // FcFusePass 绑定 fc_kernel

```

（4）不需手动添加`fc_fuse_pass.cc`和`fc_fuser.cc`到 CMakeLists 中。

2.10 版本中不再需要手动添加`fc_fuse_pass.cc`和`fc_fuser.cc`到 CMakeLists 中，`lite/core/optimizer`下的所有`*.cc`文件会被自动地在`lite/core/CMakeLists.txt`中被编译。

#### 3、使用 fc_fuse_pass

（1） `lite/api/paddle_use_passes.h`使用`USE_LITE_PASS`宏来引入新加入的 Pass 。

```c++
USE_MIR_PASS(lite_fc_fuse_pass);
```

（2）  修改`lite/core/optimizer/optimizer.cc`文件，添加`lite_fc_fuse_pass`到`RunDefaultOptimizer`函数中的局部变量 passes_local 中。

（3） 以上修改完成后，`CreatePaddlePredictor(CxxConfig)`时，模型优化过程会调用`lite_fc_fuse_pass `，扫描`mul + element_wise add`结构并替换为等效的 Fc_OP 。
