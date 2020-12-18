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
  ![](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/images/architecture.png)
- **模型文件的加载和解析** Paddle模型由程序（Program）、块（Block）、算子（Operator）和变量（Variable）组成（如下图所示，程序由若干块组成，块由若干算子和变量组成，变量包括中间变量和持久化变量，如卷积的权值），经序列化保存后形成Combined和Non-combined两种形式的模型文件，Non-combined形式的模型由一个网络拓扑结构文件__model__和一系列以变量名命名的参数文件组成，Combined形式的模型由一个网络拓扑结构文件__model__和一个合并后的参数文件__params__组成，其中网络拓扑结构文件是基于[Protocol Buffers](https://github.com/protocolbuffers/protobuf)格式以[Paddle proto 文件](https://github.com/PaddlePaddle/Paddle/blob/c5f0293cf318a8d68b7b6c9bfab58cbd744000f7/paddle/fluid/framework/framework.proto)规则序列化后的文件。现在以Non-combined格式的Paddle模型为例，将网络拓扑结构文件拖入到[Netron](https://netron.app/)工具即可图形化显示整个网络拓扑结构。
  ![](https://user-images.githubusercontent.com/9973393/102584042-af518600-4140-11eb-8005-3109433ed7fd.png)
  该步骤的具体实现：[https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/model_parser](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/model_parser)
- **计算图的转化** 将程序中的每个块按照如下规则生成对应的计算图的过程：每个算子或变量都对应计算图的一个节点，节点间的有向边由算子的输入、输出决定（依赖关系确定边的方向），算子节点与变量节点相邻。为了方便调试，分析阶段的各个步骤都会将计算图的拓扑结构以[DOT](https://en.wikipedia.org/wiki/DOT_(graph_description_language))格式的文本随log打印，可以将DOT文本复制、粘贴到[webgraphviz](http://www.webgraphviz.com/)进行可视化，如下图所示，黄色矩形节点为算子，椭圆形节点为变量。
  ![](https://user-images.githubusercontent.com/9973393/102598007-5c82c900-4156-11eb-936a-260d8e5d7538.png)
  该步骤的具体实现：[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/ssa_graph.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/mir/ssa_graph.cc)
