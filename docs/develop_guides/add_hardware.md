# 新增硬件

## 背景
- 深度学习技术在安防、交通、医疗、工业制造等行业获得了较广泛的应用，为了满足实际需求，越来越多算力更高、功耗更低的专用硬件被研发出来投向市场，涌现了诸如华为昇腾/麒麟SoC的达芬奇架构NPU、瑞芯微RK/RV系列SoC的NPU、寒武纪MLU、谷歌TPU和百度XPU等不同形态的硬件，以明显优于传统CPU、GPU的性能和功耗的特点，逐步获得市场的认可，被广泛用于服务器端、边缘端和移动端。

## 意义
- 良好的软件生态是硬件获得成功的关键，硬件驱动的兼容性、成熟的编译工具链、完善的SDK、API和文档、更多第三方深度学习框架的支持，都是影响硬件能否积攒用户、建立完善的软件生态的重要因素。特别的，在实际的项目部署过程中，推理引擎对硬件的支持尤为关键，它能屏蔽底层硬件细节，提供统一的接口，实现同一个模型在多种硬件间无缝迁移和异构计算，帮助应用提供方降低迁移成本并获得更高的性能；
- PaddleLite是一款支持服务器端、边缘端和移动端场景的推理引擎，在设计之初就考虑了如何友好的支持不同形态的硬件（例如CPU、GPU、DSP、FPGA和ASIC等），主要表现在推理框架与硬件解耦合，提出了统一的图优化Pass层、算子层和Kernel层接口，在实现灵活配置多硬件间异构执行的同时，最大限度地减少适配过程中对框架的修改，真正做到硬件细节对用户透明；
- PaddleLite支持的硬件目前已多达十余种，这其中不乏像华为NPU、瑞芯微NPU、联发科APU、颖脉（Imagination）NNA、百度XPU和寒武纪MLU等一线芯片（或IP）厂商研发的ASIC芯片，我们也希望更多的硬件（或IP）厂商与我们合作，共建PaddleLite和PaddlePaddle的硬件生态。
- 在阐述硬件接入的具体步骤前，我们将简单介绍下PaddleLite的工作原理，即从读入模型文件到硬件执行过程中都经历了哪些步骤？

## PaddleLite是如何工作的？
- 如下图所示，PaddleLite整个推理的过程，可以简单分成分析(Analysis phase)和执行(Execution phase)两个阶段，分析阶段包括Paddle模型文件的加载和解析、计算图的转化、图分析和优化、运行时程序的生成和执行等步骤。具体地，

  ![](https://user-images.githubusercontent.com/9973393/105955490-5830c080-60b1-11eb-9c6d-2b3bafb0ef7a.png)

- **模型文件的加载和解析** Paddle模型由程序（Program）、块（Block）、算子（Operator）和变量（Variable）组成（如下图所示，程序由若干块组成，块由若干算子和变量组成，变量包括中间变量和持久化变量，如卷积的权值），经序列化保存后形成Combined和Non-combined两种形式的模型文件，Non-combined形式的模型由一个网络拓扑结构文件__model__和一系列以变量名命名的参数文件组成，Combined形式的模型由一个网络拓扑结构文件__model__和一个合并后的参数文件__params__组成，其中网络拓扑结构文件是基于[Protocol Buffers](https://github.com/protocolbuffers/protobuf)格式以[Paddle proto 文件](https://github.com/PaddlePaddle/Paddle/blob/c5f0293cf318a8d68b7b6c9bfab58cbd744000f7/paddle/fluid/framework/framework.proto)规则序列化后的文件。现在以Non-combined格式的Paddle模型为例，将网络拓扑结构文件（要求文件名必须是__model__）拖入到[Netron](https://netron.app/)工具即可图形化显示整个网络拓扑结构。

  ![](https://user-images.githubusercontent.com/9973393/102584042-af518600-4140-11eb-8005-3109433ed7fd.png)

  该步骤的具体实现：[https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/model_parser](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/model_parser)

- **计算图的转化** 将每个块按照如下规则生成对应的计算图的过程：每个算子或变量都对应计算图的一个节点，节点间的有向边由算子的输入、输出决定（依赖关系确定边的方向），算子节点与变量节点相邻。为了方便调试，分析阶段的各个步骤都会将计算图的拓扑结构以[DOT](https://en.wikipedia.org/wiki/DOT_(graph_description_language))格式的文本随log打印，可以将DOT文本复制、粘贴到[webgraphviz](http://www.webgraphviz.com/)进行可视化，如下图所示，黄色矩形节点为算子，椭圆形节点为变量。

  ![](https://user-images.githubusercontent.com/9973393/102598007-5c82c900-4156-11eb-936a-260d8e5d7538.png)

  该步骤的具体实现：[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/ssa_graph.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/ssa_graph.cc)

- **图分析和优化** 将一系列pass（优化器，用于描述一个计算图优化生成另一个计算图的算法过程）按照一定的顺序依次应用到每个块对应的计算图的过程，包括量化信息处理、算子融合、Kernel选择、类型转化、上下文创建、内存复用优化和子图检测等，实现不同设备的适配、高效的计算和更少的内存占用。其中，算子融合作为一种行之有效的优化策略，普遍存在于各种推理框架中，它通过相邻算子间的融合，减少访存和计算量，有效提高模型的整体性能，例如前一步骤的计算图中，conv_bn_fuse_pass、conv_activation_fuse_pass分别以conv2d+batch_norm和conv2d+relu为pattern，先后搜索整个计算图并完成融合，如下图所示，conv2d+batch_norm+relu结构，经过前面的pass处理后只保留了1个conv2d算子。

  ![](https://user-images.githubusercontent.com/9973393/102776582-ef776980-43c9-11eb-9e23-c675cff55e73.png)

  该步骤的具体实现：[https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/core/mir](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/core/mir)
  
  - Pass的注册方法、管理机制可以参考文档[新增Pass](./add_new_pass)；[Pass列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/2e1c3ec48b46721093e9e999fd7209d6b71a61c0/lite/core/optimizer.h#L87)是指按照规定的顺序处理的Pass的集合，它使用std::vector<<std::string>>存储，每个元素代表已注册到框架的Pass的名称，如果需要在Pass列表中新增一个Pass，只需在合适的位置增加一个字符串即可，例如，为了可视化conv_bn_fuse_pass优化后的计算图，可以在它后面增加一个名为[graph_visualize_pass](https://github.com/PaddlePaddle/Paddle-Lite/blob/2e1c3ec48b46721093e9e999fd7209d6b71a61c0/lite/core/mir/graph_visualize_pass.cc)的特殊Pass，用于在log中生成以DOT文本的表示计算图结构。

    ```cpp
    diff --git a/lite/core/optimizer.h b/lite/core/optimizer.h
    index 678db707..fb0be753 100644
    --- a/lite/core/optimizer.h
    +++ b/lite/core/optimizer.h
    @@ -85,6 +85,7 @@ class Optimizer {
          "weight_quantization_preprocess_pass",  //
          "lite_conv_elementwise_fuse_pass",      // conv-elemwise-bn
          "lite_conv_bn_fuse_pass",               //
    +     "graph_visualize_pass",
          "lite_conv_elementwise_fuse_pass",      // conv-bn-elemwise
          "lite_conv_conv_fuse_pass",             //
          // TODO(Superjomn) Refine the fusion related design to select fusion
    ```

- **运行时程序的生成和执行** 按照拓扑顺序遍历优化后的计算图，生成算子和Kernel列表的过程，它基于[generate_program_pass](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/generate_program_pass.cc)实现。具体地，只遍历计算图中的算子节点，提取所携带的算子和Kernel（经过static_kernel_pick_pass选取的、适合目标硬件的、最优的Kernel）对象，以[Instruction](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/program.h)封装后按顺序依次存放到[RuntimeProgram](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/program.h)对象。运行时程序的执行也非常简单，即依次遍历RuntimeProgram对象存储的每个Instruction，调用其算子对象的CheckShape和InfereShape方法，最后执行Kernel对象的Launch方法。

  该步骤的具体实现：https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/generate_program_pass.cc 和 https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/program.cc

## 硬件接入方式
- 按照层次硬件提供给开发者的接口一般可以分为两类：
  - **通用编程接口（Low Level）**
    - 通用的编程语言：例如NVIDIA的CUDA、Knorons的OpenCL和寒武纪的BANG-C语言等；
    - 高性能库函数：例如NVIDIA的cuDNN、Intel的MKL和MKL-DNN等。

    优点是灵活，但缺点也显而易见，性能的好坏取决于负责接入框架的研发同学的能力和经验，更依赖对硬件的熟悉程度；

  - **中间表示层（Intermediate Representation，IR）接口（High Level）** 
    - 组网IR和运行时API：例如NVIDIA的TensorRT、Intel的nGraph、华为HiAI IR和百度昆仑的XTCL接口。
    
    优点是屏蔽硬件细节，模型的优化、生成和执行由运行时库完成，对负责接入框架的研发同学要求较低，性能取决于硬件厂商（或IP提供商）的研发能力，相对可控；

- 提供这两类接口的硬件可分别按照如下两种接入方式接入到框架：

### 算子Kernel接入方式
- 主要涉及PaddleLite架构图中算子、Kernel层的硬件适配工作，具体是在[lite/kernels](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/kernels)下增加待新增硬件的目录，为每个算子实现待新增硬件的Kernel，具体可参考[新增OP](./add_operation)中"添加Argmax Kernel并绑定"步骤；

  ARM Kernel的参考实现：[https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/kernels/arm](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/kernels/arm)

- 为了将硬件细节与Kernel的实现剥离，减少冗余代码，建议在lite/backends目录下增加待新增硬件的目录，利用硬件提供的编程接口实现诸如gemm等通用数学运算，向Kernel提供统一的数学运算接口；

  ARM Backend的参考实现：[https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/backends/arm](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/backends/arm)

- 其它诸如添加新增硬件的Target、Place、Context等方面的内容可参考即将详细介绍的"子图接入方式"中的相关章节。

### 子图接入方式
- **硬件要求提供以下接口：**
  - Graph组网接口，必须保证不同硬件型号、不同软件版本间的兼容性；
  - Graph生成Model的接口；
  - 设置Model的输入、输出张量和Model执行的运行时接口；
  - 输入、输出张量的内存管理接口。
  
  具体可参考华为[HiAI DDK v330](https://paddlelite-demo.bj.bcebos.com/devices/huawei/kirin/hiai_ddk_lib_330.tar.gz)、瑞芯微[rknpu_ddk](https://github.com/airockchip/rknpu_ddk.git)和[MTK Neuron Adapter](https://paddlelite-demo.bj.bcebos.com/devices/mediatek/apu_ddk.tar.gz)（类似Android NNAPI）进行接口设计。

- **什么是子图？** 将计算图依据某种规则分割为多个部分，每个部分都被称为一个子图，它包含一个或多个算子和变量，规则一般依据硬件支持能力而定。

- **框架如何实现子图检测和融合？** 在"PaddleLite是如何工作的？"章节中的"图分析和优化"步骤曾提到了[子图检测Pass](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/subgraph/subgraph_pass.cc)，它依据硬件对Paddle算子的支持情况，将每个块对应的计算图分别进行分割，生成一个或多个子图，如下图所示，具体包括以下三个步骤：

  ![](https://user-images.githubusercontent.com/9973393/102796707-9fa89a80-43e9-11eb-913b-d954238994cf.png)

  - **算子标记** 按照拓扑顺序依次遍历计算图中每个算子，依据[已注册的Paddle算子->硬件IR的转换表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/npu/bridges/paddle_use_bridges.h)，标记能够转为硬件IR的算子。例如，在上图第一幅图的计算图中，包含10个算子Op1~Op10，假设Op1、Op3和Op10不能够转为硬件IR，如第二幅图所示，这三个算子会被标记为默认颜色（黄色），代表使用CPU Kernel进行计算，而Op2、Op4、Op5、Op6、Op7、Op8和Op9则标记为红色，代表这些算子可以被转换成硬件的IR。

  - **子图检测** 对标记的算子作进一步分析，采用反向DFS算法将相邻的算子标记为同一个子图。例如上图第三幅图所示，Op2被单独分到子图1，而Op4、Op5、Op6、Op7、Op8和Op9则划到子图2。

  - **子图融合** 为了减少硬件与Host端过多的数据拷贝而带来的额外开销，如果某个子图的算子过少，则删除该子图，即它所包含的所有算子都不会放在目标硬件上执行，然后对保留下来的子图进行算子融合，具体是利用一个子图算子代替该子图包含的所有算子，但所有算子信息将以新的块（Block desc）的形式保存在程序（Program desc）中，块索引则以属性的形式保存子图算子中。

  - 由于子图检测代码较为通用，在硬件接入的过程中无需做过多的修改，只需参照着增加对应硬件的subgraph pass即可。具体可参考HuaweiKirinNPUSubgraphPass和BaiduXPUSubgraphPass的实现：[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/subgraph/subgraph_pass.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/subgraph/subgraph_pass.cc)
  
- **框架如何执行子图？** 当计算图被分割成若干普通算子和多个子图算子后（如上图的第四幅图所示，包含4个普通算子Op1、Op2、Op3和Op10，1个子图算子Op1），通过"运行时程序的生成和执行"步骤将普通算子（Kernel）和子图算子（Kernel，参考[Huawei Kirin NPU subgraph op kernel](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/npu/subgraph_compute.h)或[Baidu XPU subgraph op kernel](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/xpu/subgraph_compute.h)）保存在运行时程序中，当运行时程序执行时，如果遇到子图算子，则执行如下步骤：

  - **读取并加载子图中的原始算子** 通过保存在子图算子中的sub_block属性，在程序描述对象（Program desc）中找到相应的块描述对象（Block desc），然后依次读取算子描述对象（Op desc），根据算子类型创建算子对象。

  - **加载原始算子的Kernel，创建原始运行时程序** 通过算子描述对象中保存的[kKernelTypeAttr](https://github.com/PaddlePaddle/Paddle-Lite/blob/bb1cf7ffbba3fb4a116dad96178d4455f14f07eb/lite/core/program.h#L30)属性找到对应的Kernel，与上个步骤获得的算子对象，一起封装在Instruction对象中，所有算子的Instruction对象便组成了子图原始运行时程序。

    上述两个步骤的具体实现：[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/subgraph_engine_base.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/subgraph_engine_base.cc) 和 [https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/program.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/program.cc)

  - **原始算子转为硬件IR、组网生成Graph** 遍历子图中的所有原始算子（已按照拓扑顺序排序），依次将每个原始算子转为硬件IR，具体地，通过算子类型查询是否注册对应的桥接器（Op bridge/converter），如果[已注册](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/npu/bridges/paddle_use_bridges.h)，则执行桥接器实现算子到硬件IR的转换，并调用硬件组网API生成Graph。桥接器是子图接入方式最重要的模块，也是工作量最大的部分，为了尽可能将算子放到硬件上执行，应当为每个算子增加相应的桥接器，桥接器的实现可参考[Huawei Kirin NPU activation op bridge](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/npu/bridges/act_op.cc)和[Baidu XPU activation op bridge](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/xpu/bridges/act_op.cc)的实现。
 
  - **Graph生成Model，设置输入、输出张量**：当子图中所有原始算子都转换完成后，调用硬件提供的接口将Graph生成Model并设置输入、输出张量。

  - **执行Model，读取输出张量的数据**：将原始输入张量的数据拷贝至（或将指针传递至，以防止重复拷贝实现ZeroCopy）硬件Model的输入张量，然后调用硬件提供Model执行接口，待执行结束后将硬件输出张量的数据拷贝至原始输出张量。

    上述三个步骤的具体实现：https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/npu/subgraph_compute.cc
  
  - 前四个步骤一般在子图算子Kernel第一次运行的时候执行，只有在输入尺寸发生变更且需要重新生成Model时，才会回到步骤三重新执行。

  - 为什么需要创建原始运行时程序？在硬件IR转换失败、Graph或Model生成失败的时候，例如不同硬件型号、不同软件版本导致的不兼容，或者运行在不支持该硬件的设备上时，就需要回退到原始运行时程序进行执行，完成推理任务。

- **硬件接入时需要做哪些代码改动？**
  - 参考最近接入的Imagination NNA的Pull Request(PR)的代码修改[https://github.com/PaddlePaddle/Paddle-Lite/pull/4335](https://github.com/PaddlePaddle/Paddle-Lite/pull/4335)

## 代码提交、Review、合入机制、CI机制
  - 参考[编译环境准备](../source_compile/compile_env)中的Docker开发环境（由于代码提交时会使用git pre-commit hooks，对clang-format版本约束）
  - 注册[github](https://www.github.com/)账户，将[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)代码仓库Fork到自己的账户.
  - 将自己github账户的Paddle-Lite仓库克隆到本地。
  ```
  # git clone https://github.com/UserName/Paddle-Lite
  # cd Paddle-Lite
  ```
  - 创建本地分支：从develop分支创建一个新的本地分支，命名规则为UserName/FeatureName，例如hongming/print_ssa_graph
  ```
  $ git checkout -b UserName/FeatureName
  ```
  - 启用pre-commit钩子：[pre-commit](http://pre-commit.com/) 作为git预提交钩子，帮助我们在git commit时进行自动代码（C++，Python）格式化和其它检查（如每个文件只有一个 EOL，Git 中不要添加大文件等），可通过以下命令进行安装（注意：pre-commit测试是 Travis-CI 中单元测试的一部分，不满足钩子的PR不能被提交到Paddle-Lite）：
  ```
  $ pip install pre-commit
  $ pre-commit install
  ```
  - 修改代码：提交代码前通过git status和git diff命令查看代码改动是否符合预期，避免提交不必要或错误的修改。
  ```
  $ git status
  On branch hongming/print_ssa_graph
  Changes not staged for commit:
    (use "git add <file>..." to update what will be committed)
    (use "git checkout -- <file>..." to discard changes in working directory)
    (commit or discard the untracked or modified content in submodules)

          modified:   lite/core/optimizer.h

  $ git diff
  diff --git a/lite/core/optimizer.h b/lite/core/optimizer.h
  index 00e9e07..1b273af 100644
  --- a/lite/core/optimizer.h
  +++ b/lite/core/optimizer.h
  @@ -55,7 +55,8 @@ class Optimizer {

       if (passes.empty()) {
         std::vector<std::string> passes_local{
  -          {"lite_quant_dequant_fuse_pass",     //
  +          {"graph_visualze",
  +           "lite_quant_dequant_fuse_pass",     //
              "lite_conv_elementwise_fuse_pass",  // conv-elemwise-bn
  ```
  - 提交代码：git add命令添加需要修改的文件，放弃提交可用git reset命令，放弃修改可使用git checkout -- [file_name]命令，每次代码提交时都需要填写说明，以便让他人知道这次提交做了哪些修改，可通过git commit命令完成，修改提交说明可通过git commit --amend命令；为了触发CI，提交说明最后结束前必须回车换行，然后添加test=develop，如果本次提交的Pull request仅修改doc目录下的文档，则额外加上test=document_fix加快CI流水线。
  ```
  $ git add lite/core/optimizer.h

  $ git status
  On branch hongming/print_ssa_graph
  Changes to be committed:
    (use "git reset HEAD <file>..." to unstage)

          modified:   lite/core/optimizer.h

  $ git commit -m "Add graph_visualze pass to output ssa graph
  > test=develop"
  CRLF end-lines remover...................................................Passed
  Check for added large files..............................................Passed
  Check for merge conflicts................................................Passed
  Check for broken symlinks................................................Passed
  Detect Private Key.......................................................Passed
  Fix End of Files.........................................................Passed
  clang-format.............................................................Passed
  cpplint..................................................................Passed
  copyright_checker........................................................Passed
  [hongming/print_ssa_graph 75ecdce] Add graph_visualze pass to output ssa graph test=develop
   1 file changed, 2 insertions(+), 1 deletion(-)
  ```
  - 同步本地仓库代码：在准备发起Pull Request前，需要将原仓库[https://github.com/PaddlePaddle/Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)的develop分支的最新代码同步到本地仓库的新建分支。首先通过git remote -v命令查看当前远程仓库的名字，然后通过git remote add 命令添加原Paddle Lite仓库地址，最后使用git fetch和git pull命令将本地分支更新到最新代码。
  ```
  $ git remote -v
  origin  https://github.com/UserName/Paddle-Lite.git (fetch)
  origin  https://github.com/UserName/Paddle-Lite.git (push)

  $ git remote add upstream https://github.com/PaddlePaddle/Paddle-Lite

  $ git remote
  origin
  upstream

  $ git fetch upstream
  remote: Enumerating objects: 105, done.
  remote: Counting objects: 100% (105/105), done.
  remote: Compressing objects: 100% (6/6), done.
  remote: Total 142 (delta 99), reused 100 (delta 99), pack-reused 37
  Receiving objects: 100% (142/142), 52.47 KiB | 2.00 KiB/s, done.
  Resolving deltas: 100% (103/103), completed with 45 local objects.
  From https://github.com/PaddlePaddle/Paddle-Lite
    a1527e8..d6cdb1e  develop    -> upstream/develop
    2136df9..17a58b6  gh-pages   -> upstream/gh-pages
    1091ab8..55be873  image-sr-v2 -> upstream/image-sr-v2
   * [new branch]      release/v2.2.0 -> upstream/release/v2.2.0
   * [new tag]         v2.2.0     -> v2.2.0

  $ git branch
  develop
  * hongming/print_ssa_graph

  $ git pull upstream develop
  From https://github.com/PaddlePaddle/Paddle-Lite
   * branch            develop    -> FETCH_HEAD
  Removing lite/kernels/npu/bridges/transpose_op_test.cc
  Removing lite/kernels/npu/bridges/batch_norm_op_test.cc
  Merge made by the 'recursive' strategy.
   lite/kernels/npu/bridges/batch_norm_op_test.cc | 168 ------------------------------------------------------------------------------------------------
   lite/kernels/npu/bridges/transpose_op.cc       |   2 +-
   lite/kernels/npu/bridges/transpose_op_test.cc  | 153 ---------------------------------------------------------------------------------------
   lite/tests/kernels/CMakeLists.txt              |   4 +--
   lite/tests/kernels/batch_norm_compute_test.cc  |   2 ++
   lite/tests/kernels/transpose_compute_test.cc   |  44 ++++++++++++-------------
   mobile/test/CMakeLists.txt                     |   6 ++++
   mobile/test/net/test_mobilenet_male2fe.cpp     |  66 ++++++++++++++++++++++++++++++++++++++
   8 files changed, 99 insertions(+), 346 deletions(-)
   delete mode 100644 lite/kernels/npu/bridges/batch_norm_op_test.cc
   delete mode 100644 lite/kernels/npu/bridges/transpose_op_test.cc
   create mode 100644 mobile/test/net/test_mobilenet_male2fe.cpp
  ```
  - Push到远程仓库：将本地的修改推送到自己账户下的Paddle Lite仓库，即https://github.com/UserName/Paddle-Lite 。
  ```
  $ git branch
  develop
  * hongming/print_ssa_graph

  $ git push origin hongming/print_ssa_graph
  Counting objects: 8, done.
  Delta compression using up to 2 threads.
  Compressing objects: 100% (8/8), done.
  Writing objects: 100% (8/8), 868 bytes | 0 bytes/s, done.
  Total 8 (delta 6), reused 0 (delta 0)
  remote: Resolving deltas: 100% (6/6), completed with 6 local objects.
  remote: 
  remote: Create a pull request for 'hongming/print_ssa_graph' on GitHub by visiting:
  remote:      https://github.com/UserName/Paddle-Lite/pull/new/hongming/print_ssa_graph
  remote: 
  To https://github.com/UserName/Paddle-Lite.git
   * [new branch]      hongming/print_ssa_graph -> hongming/print_ssa_graph
  ```
  - 发起Pull Request：登录github，在自己账户下找到并进入UserName/Paddle-Lite仓库，这时会自动提示创建Pull Request，点击Create Pull Request按钮，一般来说会自动选择比较更改的仓库和分支，如果需要手动设置，可将base repository选择为PaddlePaddle/Paddle-Lite，base分支为develop，然后将head repository选择为UserName/Paddle-Lite，compare分支为hongming/print_ssa_graph。PR（Pull Request）的标题必须用英文概括本次提交的修改内容，例如修复了什么问题，增加了什么功能。同时，为了便于其他人快速得知该PR影响了哪些模块，应该在标题前添加中括号+模块名称进行标识，例如"[HuaweiKirinNPU][BaiduXPU] Temporarily toggle printing ssa graph, test=develop"。 PR的描述必须详细描述本次修改的原因/背景、解决方法、对其它模块会产生何种影响（例如生成库的大小增量是多少），性能优化的PR需要有性能对比数据等。
  - 签署CLA协议：在首次向Paddle-Lite提交Pull Request时，您需要您签署一次CLA(Contributor License Agreement)协议，以保证您的代码可以被合入。
  - 等待CI测试完成：您在Pull Request中每提交一次新的commit后，都会触发一系列CI流水线（根据场景/硬件的不同，一般会有多个流水线），它将会在几个小时内完成，只需保证带有Required的流水线通过即可。例如下图所示，每项流水线测试通过后，都会在前面打勾，否则打叉，可点击Details查看日志定位错误原因：
  ![](https://user-images.githubusercontent.com/9973393/113404216-631e0f00-93da-11eb-8dad-fb47c8f512de.png)
  - PR Review：每个PR需要至少一个评审人apporve后才能进行代码合入，而且在请评审人review代码前，必须保证CI测试完成并通过全部测试项，否则评审人一般不做评审。根据PR修改的模块不同，代码评审人选择也不一样。例如：涉及到Core和API模块，需要@Superjomn进行Review，涉及到Subgraph相关的修改，需要@hong19860320或@zhupengyang进行Review。评审人的每个意见都必须回复，同意评审意见且按其修改完的，给个简单的Done即可，对评审意见不同意的，请给出您自己的反驳理由。
  - PR 合入：一般PR会有多次commit，原则上是尽量少的commit，且每个commit的内容不能太随意。在合入代码时，需要对多个commit进行squash commits after push，该PR在评审人approve且CI完全通过后，会出现"Squash and Merge"按钮，如上图所示，届时可以联系Paddle同学完成PR的合入。

## 硬件接入完成标志
  - 代码合入到develop分支
  - 提供完善的文档和Demo
    - 参考[ImaginationNNA](../demo_guides/imagination_nna)的格式编写文档并提供Demo压缩包（由Paddle同学上传到百度云）
    - 如果编译环境的docker镜像与PaddleLite所提供的不一致，需要额外提供构建docker镜像的docker file，保证用户能顺利编译获得产出
  - 厂商提供测试设备，增加CI流水线（由Paddle同学负责）
  - 双方兼容性认证
