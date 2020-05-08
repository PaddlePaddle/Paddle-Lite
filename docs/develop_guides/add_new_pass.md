# 新增Pass

本文从三个方面介绍了`Lite`中的`Pass`结构：**Pass是什么**、**Pass的实现与接口**、**Pass的一般注册流程**。最后以`Fc_fuse_pass`为例介绍了`fusion_pass`的作用与注册方法。

## 前述：Pass是什么？

**CxxPredictor加载模型后，在执行预测前会先优化模型。模型优化过程是通过Pass实现的。**
具体调用关系如下：
![图片](https://user-images.githubusercontent.com/45189361/69638690-20d21880-1096-11ea-8169-1d2c7e1a1609.png)

 - `CreatePredictor(CxxConfig)`函数调用了Predictor->Build(CxxConfig)
   - CxxPredictor的构建过程（Build）分为两步：
     - Predictor->LoadModel()          加载模型文件到program中
     - Predicotr->optimizer_.Run()    对Program中的原始图形结构进行优化
          - 对图结构的优化是通过调用 `Pass->Apply(const std::unique_ptr<SSAGraph>& graph)`方法实现的。


**每一类Pass定义了一种优化过程**，包括：原模型中的kernel选取、OP融合、冗余OP去除、子图创建、内存优化、类型推导、类型转换等。




## Pass的实现与接口 ：Pass基类、PassManager和Pass注册

### 1、Pass基类：`paddle::lite::mir::Pass`
```c++
class Pass {
 public:
  // Pass的类型，Pass按照作用的不同可以分为三种
  enum class Kind {   //种类的作用不太清楚
    // 1. 修改模型中的图拓扑结构的Pass
    kProgramWise = 0,
    // 2. 不修改图结构，修改状态的Pass
    kStmtWise,     
    // 3. 不修改 IR，用于搜集信息和可视化信息的Pass.
    kDebug,
  };
  
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
**代码位置**：`lite/core/mir/pass.h`
**主要类成员**：
  `const Kind kind_` : Pass类型。pass 有三种基本基本类型 ：修改图结构的`ProgramPass`、修改状态量的`StmtPass`和Debug过程采集信息与控制可视化的`DebugPass`。  
  `std::string name_` ：pass 的名称
  `std::set<TargetType> bound_targets_` : Pass运行的硬件平台，optimizer.Run()优化过程会根据硬件平台选择匹配的Pass。------根据硬件平台自动选择需要的pass
  `std::unordered_map<std::string, std::set<lite_api::Place>> bound_kernels_` : Pass 绑定的kernel   (what's this used for)
**主要接口**： 
  `Pass::Apply(const std::unique_ptr& graph)` : Pass优化过程的具体操作，是新注册Pass需要实现的接口。输入为`SSAGraph`型指针，是对模型结构的拓扑表示。

### 2、Pass管理 `paddle::lite::mir::PassManager` 

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

 private:
  std::list<std::unique_ptr> passes_;  //存储所有的 Pass
  std::map<std::string, mir::Pass*> pass_map_;    //使用map变量存储 PassName::Pass
  
 }

```
**代码位置**：`lite/core/mir/pass_manager.h`
**主要类成员**：
`std::list:unique_ptr> passes_;`  : List类型，存储了所有已注册Pass。
`std::map<std::string, mir::Pass*> pass_map_; `  :   Map类型，存储了所有"Pass名称-Pass类"键对，用于根据名称查找Pass。

**主要接口**：
 `static PassManager& Global()` 返回PassManager全局静态变量,该变量存储了所有已注册的Pass
` bool AddNewPass(const std::string& name, Pass* pass)` 添加新的Pass到PassManager中


### 3、 Pass 注册 `paddle::lite::mir::PassRegistry`
**代码位置**：`lite/core/mir/pass_registry.h`
**主要接口**：
`REGISTER_MIR_PASS(name__, class__)` ：宏定义函数，用于注册Pass。注册Pass过程实现的是 `PassManager::Global().AddNewPass(name__, class__)`，将新注册Pass添加到全局变量`PassManager`中。



## Pass的一般注册流程与使用方法

### 1. Pass 注册流程
在`lite/core/mir`或其子目录下继承`Pass基类`，实现`Pass::Apply`接口，并使用宏`REGISTER_MIR_PASS(name__, class__)`将Pass注册到`PassManager`即完成了新Pass注册。

**以新建 **`new_demo_pass`**为例**，具体流程如下：
（1）在`lite/core/mir`路径下新建`example_pass.cc` 和 `new_demo_pass.h` 文件
（2）在`example_pass.h` 文件中继承Pass基类（ProgramPass、StmtPass或DebugPass）定义自己的Pass类。
```c++
#include "lite/core/mir/pass.h"

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
（3）在`example_pass.cc` 文件中实现`ExamplePass::Apply()`接口，并注册`ExamplePass`
```c++
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/example_pass.h"

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
    .BindTargets({TARGET(kARM)}); // Pass执行的目标硬件平台
    // .BindKernel("conv2d");     //Pass绑定的 kernel
```

（4）修改`lite/core/mir/CMakeLists.txt`文件，将`example_pass.cc` 编译到`mir_passes`库中

```cmake
lite_cc_library(mir_passes
  SRCS
      demo_pass.cc  // 新建的Pass文件
      ...
      memory_optimize_pass.cc
  DEPS mir_pass types context ${mir_fusers} ${subgraph_passes})
```
### 2. Pass使用流程

将Pass注册到PassManager后不会自动生效。需要在`optimizer->run()` 函数中添加该Pass才会在模型优化过程中调用。
（1）在`paddle_use_passes.h`文件中调用该Pass

```cmake
#include "paddle_lite_factory_helper.h"  // NOLINT
    ...
USE_MIR_PASS(new_demo_pass);  //调用 new_demo_pass
```
（2）要想在优化模型时调用该Pass，需要在`optimizer->run()`函数中手动添加调用。

修改`lite/core/optimizer.h`文件，添加`new_demo_pass`到`Optimizer::Run()`函数；
```c++
 class Optimizer {
 public:
  void Run(...) {
   ...
    if (passes.empty()) {
      RunPasses(std::vector<std::string>{
          {"new_demo_pass"     //将新注册的Pass添加在这里
             ...
           }
    ...
 }      
```
（3）只有CxxPredictor才会在模型加载后根据Pass优化模型。
```c++
 ...
#include "paddle_use_passes.h"   // 引用Pass优化模型
void RunModel() {
  // 1. 创建 CxxConfig
  CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places(Place{TARGET(kARM), PRECISION(kFloat)});

  // 2. 创建CxxPredictor,该过程包括加载模型和用Pass优化模型
  std::shared_ptr> predictor =
      Creat<CxxConfig>(config);
}
```




## Fusion Pass的定义与注册

`Fusion Pass`是一种常见图结构优化Pass，可将多个连续OP融合成单个等效OP，减少数据交换并简化图结构。Pass运行时调用`Fuser`自动查找并替换指定图结构，所以注册`FuserPass`时还需要实现对应的Fuser类。

下面以`fc_fuse_pass`为例，详细说明`FusionPass`的效果和注册方法。

### `fc_fuse_pass`的作用
将相邻的`mul`算子和 `element_wise add `算子 融合成一个 `FC`  算子
```c++
mul(X) =  X * W 
elementwise_add( mul(x) ) = X * W + Bias
//----------> after fusion
FC(X) = X * W +Bias
```

Pass 运行效果如下：
![图片](https://user-images.githubusercontent.com/45189361/69639193-12383100-1097-11ea-9063-21f030414080.png)
mul和elementwise_add的原有参数映射到FC的参数上：
![图片](https://user-images.githubusercontent.com/45189361/69638836-74446680-1096-11ea-9cdc-a961fa995dfe.png)

### `fc_fuse_pass`的注册方法
#### 1、创建FcFuser
（1）在`lite/core/mir/fusion`路径下新建`fc_fuser.cc` 和 `fc_fuser.h` 文件
（2）在`fc_fuser.h` 文件中继承`FuseBase`定义自己的Fuser类。

```c++
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class FcFuser : public FuseBase {
 public:
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
```
**主要接口**：
`FuseBase::BuildPattern` ：  描述需要替换位置的图结构（pattern），Fuser运行时会自动查找并替换该pattern。
`FuseBase::GenOpDesc` ：       创建融合后的等效Fused_op。
`FuseBase::InsertNewNode` ：用Fused_op替换原始图结构（pattern）。

对于 `FcFuser`：BuildPattern描述的Pattern是`mul+elementwise add`，GenOpDesc创建的FC_op，InsertNewNode函数的效果是用新建的`FC_op`替换模型中的`mul+elementwise add` pattern。


（3） 在`fc_fuser.cc`文件中实现 `BuildPattern()` 、`GenOpDesc()`、`InsertNewNode() `接口

下面以FcFuser为例介绍三种接口的实现：

```c++
// 1. BuildPattern函数，描述需要替换的图结构
// FcFuser::BuildPattern() 描述了 mul + element_wise add 图结构
void FcFuser::BuildPattern() {
  // （1） 用OpNode描述和VarNode
  // mul OP
  auto* mul = OpNode("mul", "mul");
  // mul OP 的输入和输出
  auto* x = VarNode("x")->assert_is_op_input("mul", "X");
  auto* W = VarNode("W")->assert_is_op_input("mul", "Y");
  auto* mul_out = VarNode("mul_out");
  
  // elementwise_add OP
  auto* add = OpNode("add", "elementwise_add");
  //elementwise_add 的输入
  auto* b = VarNode("b")->assert_is_persistable_var();
  // elementwise_add OP的输出（最终输出）
  auto* Out = VarNode("Out");

  //（2） 描述拓扑连接 （Fuse之前mul 和elementwise_add的连接）
  std::vector<PMNode*> mul_inputs{W, x};
  std::vector<PMNode*> add_inputs{mul_out, b};
  mul_inputs >> *mul >> *mul_out;
  add_inputs >> *add >> *Out;
 

  //（3） 声明新的拓扑结构中将会被移除的节点，包括被fuse的OP和OP之间的中间变量
  mul_out->AsIntermediate();
  mul->AsIntermediate();
  add->AsIntermediate();
}


// 2. GenOpDesc函数新建等效 Fused_op
// FcFuser::GenOpDesc() 新建了Fc_op
cpp::OpDesc FcFuser::GenOpDesc(const key2nodes_t& matched) {
  // (1) 得到第一个OP节点的 OpDesc ，并清空输入输出信息
  cpp::OpDesc op_desc = *matched.at("mul")->stmt()->op_info();
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

#### 2、注册fc_fuse_pass

（1）在`lite/core/mir/fusion`路径下新建`fc_fuse_pass.cc` 和 `fc_fuse_pass.h` 文件
（2）在`fc_fuse_pass.h` 文件中，继承`ProgramPass`定义`FcFusePass`。

```c++
#include "lite/core/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {
class FcFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override; namespace mir namespace lite namespace paddle
```
（3）在`fc_fuse_pass.cc` 文件中实现`FcFusePass::Apply()`接口，并注册`FcFusePass`
```c++
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/example_pass.h"

namespace paddle {
namespace lite {
namespace mir {
void FcFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  fusion::FcFuser fuser;
  fuser(graph.get());namespace mir
}  // namespace lite
}  // namespace paddle
REGISTER_MIR_PASS(lite_fc_fuse_pass, paddle::lite::mir::FcFusePass)
    .BindTargets({TARGET(kAny)})  // FcFusePass 可以在任何硬件平台执行
    .BindKernel("fc");            // FcFusePass 绑定 fc_kernel
```

（4）修改`lite/core/mir/fusion/CMakeLists.txt`文件，将`fc_fuser.cc` 编译到`mir_fusers`库

```cmake
lite_cc_library(fuse_fc
        SRCS fc_fuser.cc
        DEPS pattern_matcher_high_api) 

set(mir_fusers
    fuse_fc
     ... 
    CACHE INTERNAL "fusers")
```

（5）修改`lite/core/mir/CMakeLists.txt`文件，将`fc_fuse_pass.cc` 编译到`mir_pass`库
```cmake
lite_cc_library(mir_passes
  SRCS
      fusion/fc_fuse_pass.cc
       ...
  DEPS mir_pass types context ${mir_fusers} ${subgraph_passes})
```

#### 3、使用 fc_fuse_pass

（1） `lite/api/paddle_use_passes.h`使用`USE_LITE_PASS`宏来引入新加入的pass

```c++
USE_MIR_PASS(lite_fc_fuse_pass);
```
（2）  在`lite/core/optimizer.h`文件的`Optimizer::Run()`函数中添加新注册的pass
```C++
class Optimizer {
 public:
  void Run(Program&& program,
           const std::vector<Place>& valid_places,
           core::KernelPickFactor kernel_pick_factor,
           const std::vector<std::string>& passes = {}) {
           ...    
    if (passes.empty()) {
      RunPasses(std::vector<std::string>{
          {"lite_fc_fuse_pass",                // the newly registered pass
            ...
           "argument_type_display_pass"}});
    } else {
      RunPasses(passes);
    }
    exec_scope_ = program.exec_scope();
  }
```
（3） 以上修改完成后，在CreatePredictor（CxxConfig）创建CxxPredictor时，模型优化过程会调用`lite_fc_fuse_pass `，扫描`mul + element_wise add`结构并替换为等效的Fc_OP。
