# NNAdapter
飞桨推理AI硬件统一适配框架

## 背景
- 在[新增硬件](./add_hardware)章节中曾提到Paddle Lite的硬件适配主要分为算子和子图两种方式，特别是AI硬件，近两年来我们基于子图方式完成了华为麒麟NPU、瑞芯微NPU、联发科APU、颖脉NNA、寒武纪MLU和比特大陆NPU在Paddle Lite上的适配。但在与硬件厂商合作过程中，逐渐发现了该方案的不足之处，主要涉及以下两大方面：
  - 适配门槛高、周期长
    - 要求硬件厂商对Paddle Lite有较深的了解，涵盖框架运行机制、硬件接入方案、编译系统等方面。
    - 获取Paddle模型、算子定义、量化实现方式等信息所花费的沟通成本过高。
  - 适配代码与框架过度耦合，且存在重复开发、代码维护成本过高
    - 适配一个新的硬件并跑通一个分类模型，总文件修改数共48个，而框架文件的修改竟达到25个。
    - Paddle算子转硬件算子存在重复开发，且一旦Paddle算子发生升级，就需要对已支持的所有硬件的相关代码进行适配，维护成本过高。
    - 量化方式（Paddle仅支持对称量化，而大部分SoC类NPU支持非对称量化）、数据布局（例如联发科APU仅支持NHWC，而Paddle大部分模型为NCHW格式）的转换等模块存在重复实现，不利于各硬件间的共享达到缩减适配周期、降低维护成本的目的。

## 简介
- NNAdapter是什么？
  - 由一系列C接口组成的、支撑各种深度学习框架在各种硬件（特别是AI ASIC芯片）完成高效推理的通用接口，它是建立深度学习推理框架和硬件的桥梁，包含API、Runtime、HAL三层，以及模型中间表示层的标准算子定义。

  ![](https://paddlelite-demo.bj.bcebos.com/devices/generic/nnadapter_arch.png)

- NNAdapter的目的是什么？
  - 降低接入门槛，不要求硬件厂商深入了解Paddle Lite框架，只需了解NNAdapter的标准算子定义、HAL层标准接口定义、Runtime与HAL层的调用关系即可。
  - 减少适配工作量，缩短适配周期，只需完成硬件的HAL层库的开发即可。
  - 与推理框架解耦，降低维护成本。

- NNAdapter做了哪些工作？
  - 标准化向上（推理框架）的接口，包括设备管理、模型组网、生成和执行的一系列C接口。
  - 标准化算子定义，提供稳定的、文档丰富的中间表示层的算子定义（主要参考ONNX、Paddle、PyTorch和TensorFlow的算子），方便硬件厂商快速完成算子映射/转换。
  - 标准化向下（硬件）抽象层（HAL）的接口定义，实现对硬件设备的抽象和封装（屏蔽硬件细节），为NNAdapter在不同硬件设备提供统一的访问接口。

## 功能模块
![](https://paddlelite-demo.bj.bcebos.com/devices/generic/nnadapter_arch_detail.png)

用户视角下各功能模块的关系

![](https://paddlelite-demo.bj.bcebos.com/devices/generic/paddle_lite_and_nnadapter_dynamic_shared_library.png)

### NNAdapter API
- 设备管理
  - NNAdapterDevice_acquire、NNAdapterDevice_release、NNAdapterDevice_getName、NNAdapterDevice_getVendor、NNAdapterDevice_getType、NNAdapterDevice_getVersion
- 统一设备上下文
  - NNAdapterContext_create、NNAdapterContext_destroy
- 模型组网
  - NNAdapterModel_create、NNAdapterModel_destroy、NNAdapterModel_finish、NNAdapterModel_addOperand、NNAdapterModel_setOperandValue、NNAdapterModel_getOperandType、NNAdapterModel_addOperation、NNAdapterModel_identifyInputsAndOutputs
- 模型编译
  - NNAdapterCompilation_create、NNAdapterCompilation_destroy、NNAdapterCompilation_finish、NNAdapterCompilation_queryInputsAndOutputs
- 模型执行
  - NNAdapterExecution_create、NNAdapterExecution_destroy、NNAdapterExecution_setInput、NNAdapterExecution_setOutput、NNAdapterExecution_compute

注意：具体可以参考『附录』中的『NNAdapter API详细说明』。

### NNAdapter 标准算子
### NNAdapter Runtime
### NNAdapter HAL标准接口定义
- 示例

## NNAdapter在Paddle Lite的实现
### 实现方案
![](https://paddlelite-demo.bj.bcebos.com/devices/generic/paddle_lite_with_nnadapter.png)

### Paddle Lite为NNAdapter新增的接口
- 设备查询和设置
  - check_nnadapter_device_name
    ```c++
    bool check_nnadapter_device_name(const std::string& device_name)
    ```
    通过设备名称查询设备是否可用，设备名称包括`huawei_ascend_npu`, `huawei_kirin_npu`, `amlogic_npu`, `rockchip_npu`, `mediatek_apu`, `imagination_nna`等，已支持设备的最新列表可在[NNAdapter HAL](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/backends/nnadapter/nnadapter/driver)中查询。
    - 参数：
      - device_name：设备HAL层库的名称，例如：[huawei_ascend_npu](https://github.com/PaddlePaddle/Paddle-Lite/blob/34639deaf036e2daf4429205c1bc77958e0b1e0f/lite/backends/nnadapter/nnadapter/driver/huawei_ascend_npu/CMakeLists.txt#L15)。
    - 返回值：设备可用则返回true。

  - set_nnadapter_device_names
    ```c++
    void set_nnadapter_device_names(const std::vector<std::string>& device_names)
    ```
    设置模型在哪些设备中运行（当前版本只支持第一个设备）。
    - 参数：
      - device_names：设备名称列表。
    - 返回值：无。

- 设备上下文的参数设置
  - set_nnadapter_context_properties
    ```c++
    void set_nnadapter_context_properties(const std::string& context_properties)
    ```
    将设备参数传递给设备HAL库。
    - 参数：
      - context_properties：以Key-value字串的形式表示设备参数，例如：如果希望使用昇腾310卡的第0个核心，可以设置"HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0;"。
    - 返回值：无。

- 模型缓存
  - set_nnadapter_model_cache_dir
    ```c++
    void set_nnadapter_model_cache_dir(const std::string& model_cache_dir)
    ```
    启用模型编译缓存功能，设置编译后的设备程序的缓存文件（以.nnc为扩展名）的存储目录，它能够跳过每次进程启动且模型首次推理时的编译步骤，减少首次推理耗时。
    - 参数：
      - model_cache_dir：模型缓存目录。
    - 返回值：无。

  - set_nnadapter_model_cache_buffers
    ```c++
    void set_nnadapter_model_cache_buffers(const std::string& model_cache_token, const std::vector<char>& model_cache_buffer)
    ```
    设置模型缓存的标识和数据，子图在编译生成设备程序时，如果成功匹配到`model_cache_token`，则跳过模型编译步骤，直接使用缓存数据恢复设备程序（需要设备HAL层库的支持），该接口通常用于从内存中设置解密后的模型缓存数据。
    - 参数：
      - model_cache_token：根据子图输入、输出、设备信息按照一定规则生成的唯一标识子图的32个字符，它实现方式可以参考[相关代码](https://github.com/PaddlePaddle/Paddle-Lite/blob/9e16e8ee9a079f673d992351cdd9ec0f4d731575/lite/kernels/nnadapter/engine.cc#L49)。
      - model_cache_buffer：`model_cache_token`对应子图和设备的模型缓存数据。
    - 返回值：无。

- 自定义子图分割
  - set_nnadapter_subgraph_partition_config_path
    ```c++
    void set_nnadapter_subgraph_partition_config_path(const std::string& subgraph_partition_config_path)
    ```
    设置自定义子图分割配置文件路径，用于将某些算子强制异构到CPU，防止因切分成过多子图而导致的性能下降，内存增加。该配置文件的规则如下：

    1）每行记录用于唯一标识某一个或某一类需要被强制异构到CPU的算子。

    2）每行记录由『算子类型:输入张量名列表:输出张量名列表』组成，即以冒号分隔算子类型、输入和输出张量名列表，以逗号分隔输入、输出张量名列表中的每个张量名。

    3）可省略输入、输出张量名列表中的部分张量名，如果不设置任何输入、输出张量列表，则代表计算图中该类型的所有算子节点均被强制异构到CPU。

    用法举例：
    ```c++
    op_type0:var_name0,var_name1:var_name2    表示将类型为op_type0、输入张量为var_name0和var_name1、输出张量为var_name2的算子强制异构到CPU上
    op_type1::var_name3                       表示将类型为op_type1、任意输入张量、输出张量为var_name3的算子强制异构到CPU上
    op_type2:var_name4                        表示将类型为op_type2、输入张量为var_name4、任意输出张量的算子强制异构到CPU上
    op_type3                                  表示任意类型为op_type3的算子均被强制异构到CPU上
    ```

    为了方便唯一标识模型中的某一个算子，可以在使用cxxconfig加载Paddle模型进行nb模型转换或直接推理时，设置GLOG_v=5打印完整调试信息，然后以`subgraph operators`为关键字搜索，例如：[ssd_mobilenet_v1_relu_voc_fp32_300](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)模型运行在华为麒麟NPU时，将得到如下调试信息：

    ```
    subgraph clusters: 1
    digraph G {
    node_1150[label="batch_norm_0.tmp_3"]
    node_1154[label="batch_norm_1.tmp_3"]
    node_1190[label="batch_norm_10.tmp_3"]
    node_1194[label="batch_norm_11.tmp_3"]
    ...
    node_1426->node_1427
    node_1427->node_1428
    node_1428->node_1429
    } // end G
    subgraph operators:
    feed:feed:image
    conv2d:image,conv1_weights,conv1_bn_offset:batch_norm_0.tmp_3
    depthwise_conv2d:batch_norm_0.tmp_3,conv2_1_dw_weights,conv2_1_dw_bn_offset:batch_norm_1.tmp_3
    conv2d:batch_norm_1.tmp_3,conv2_1_sep_weights,conv2_1_sep_bn_offset:batch_norm_2.tmp_3
    ...
    box_coder:concat_0.tmp_0,concat_1.tmp_0,reshape2_0.tmp_0:box_coder_0.tmp_0
    multiclass_nms:box_coder_0.tmp_0,transpose_12.tmp_0:save_infer_model/scale_0.tmp_0
    fetch:save_infer_model/scale_0.tmp_0:fetch
    ```

    其中：

    1）`subgraph operators` 一行的后面是模型经过Paddle Lite各种优化Pass后的全部算子集合，可以非常方便的作为自定义子图分割配置文件的内容，这也将成为我们在硬件适配时快速调通目标模型的好帮手（即先将所有算子强制异构到CPU上，然后一行一行的删掉，让它们跑在目标设备上，这种方法可以快速定位问题算子，完成整个模型的调通）。

    2）`subgraph clusters` 一行的后面是经过子图检测后的子图个数，以下则是用于可视化子图检测后的模型拓扑结构的DOT格式字符串，可将其复制到[webgraphviz](http://www.webgraphviz.com/)进行可视化，其中不同颜色的算子代表所属不同的子图。
    
    同样的，以ssd_mobilenet_v1_relu_voc_fp32_300为例，下面两幅图展示了使用自定义子图分割配置文件前后的对比：

    1）未使用自定义子图分割配置：

    ![](https://paddlelite-demo.bj.bcebos.com/devices/generic/ssd_mobilenet_v1_relu_voc_fp32_300_auto_split_netron.jpg)

    2）使用如下自定义子图配置：

    ```
    transpose2:conv2d_22.tmp_1:transpose_0.tmp_0,transpose_0.tmp_1
    transpose2:conv2d_23.tmp_1:transpose_1.tmp_0,transpose_1.tmp_1
    ```

    ![](https://paddlelite-demo.bj.bcebos.com/devices/generic/ssd_mobilenet_v1_relu_voc_fp32_300_manual_split_netron.jpg)

    注意：该接口仅用于cxxconfig加载Paddle模型生成nb模型或直接推理时使用。

    - 参数：
      - model_cache_token：根据子图输入、输出、设备信息按照一定规则生成的唯一标识子图的32个字符，它实现方式可以参考[相关代码](https://github.com/PaddlePaddle/Paddle-Lite/blob/9e16e8ee9a079f673d992351cdd9ec0f4d731575/lite/kernels/nnadapter/engine.cc#L49)。
      - model_cache_buffer：`model_cache_token`对应子图和设备的模型缓存数据。
    - 返回值：无。

  - set_nnadapter_subgraph_partition_config_buffer
    ```c++
    void set_nnadapter_subgraph_partition_config_buffer(const std::string& subgraph_partition_config_buffer)
    ```
    设置自定义子图分割配置内容，该接口通常用于加、解密场景。
    - 参数：
      - subgraph_partition_config_buffer：自定义子图分割配置的内容，与`set_nnadapter_subgraph_partition_config_path`中阐述的一致。
    - 返回值：无。

### Paddle Lite与NNAdapter的一般调用过程
![](https://paddlelite-demo.bj.bcebos.com/devices/generic/nnadapter_call_flow.png)

### 应用程序、Paddle Lite、NNAdapter和硬件SDK之间的详细调用过程
![](https://paddlelite-demo.bj.bcebos.com/devices/generic/nnadapter_call_flow_detail_0.png)
![](https://paddlelite-demo.bj.bcebos.com/devices/generic/nnadapter_call_flow_detail_1.png)
![](https://paddlelite-demo.bj.bcebos.com/devices/generic/nnadapter_call_flow_detail_2.png)
![](https://paddlelite-demo.bj.bcebos.com/devices/generic/nnadapter_call_flow_detail_3.png)
![](https://paddlelite-demo.bj.bcebos.com/devices/generic/nnadapter_call_flow_detail_4.png)

## 附录

### NNAdapter API详细说明
- NNAdapter_getVersion
  ```c++
  int NNAdapter_getVersion(uint32_t* version)
  ```
  获取NNAdapter版本值。
  - 参数：
    - version：存储返回NNAdapter的版本值。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterDevice_acquire
  ```c++
  NNAdapterDevice_acquire(const char* name, NNAdapterDevice** device)
  ```
  通过名称获取设备实例。
  - 参数：
    - name：通过该名称加载并注册设备HAL库后（仅发生在进程首次调用时），创建一个设备实例。
    - device：存储创建后的设备实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterDevice_release
  ```c++
  NNAdapterDevice_release(NNAdapterDevice* device)
  ```
  释放设备实例（注意：只有进程退出时，才会释放设备HAL层库）。
  - 参数：
    - device：需要销毁的设备实例。
  - 返回值：无。

- NNAdapterDevice_getName
  ```c++
  int NNAdapterDevice_getName(const NNAdapterDevice* device, const char** name)
  ```
  获得设备名称。
  - 参数：
    - device：设备实例。
    - name：存储返回的设备名称。
  - 返回值：无。

- NNAdapterDevice_getVendor
  ```c++
  int NNAdapterDevice_getVendor(const NNAdapterDevice* device, const char** vendor)
  ```
  获得设备厂商名称。
  - 参数：
    - device：设备实例。
    - vendor：存储返回的设备厂商名称。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterDevice_getType
  ```c++
  int NNAdapterDevice_getType(const NNAdapterDevice* device, NNAdapterDeviceType* type)
  ```
  获得设备类型。
  - 参数：
    - device：设备实例。
    - type：存储返回的设备类型值，由`NNAdapterDeviceCode`定义，`NNADAPTER_CPU`代表CPU，`NNADAPTER_GPU`代表GPU，`NNADAPTER_ACCELERATOR`代表神经网络加速器。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterDevice_getVersion
  ```c++
  int NNAdapterDevice_getVersion(const NNAdapterDevice* device, int32_t* version)
  ```
  获取设备HAL动态链接库的版本值。
  - 参数：
    - device：设备实例。
    - version：存储返回的设备HAL层库的版本值。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterContext_create
  ```c++
  int NNAdapterContext_create(NNAdapterDevice** devices, uint32_t num_devices, const char* properties, NNAdapterContext** context)
  ```
  为多种设备创建一个统一设备上下文，并通过key-value字符串的形式将设备的参数信息传递给每一个设备HAL层库。
  - 参数：
    - devices：设备实例列表。
    - num_devices：`devices`中设备实例的个数。
    - properties：设备参数信息，按照key-value字符串的形式表示设备参数信息，例如："HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0"表示只使用昇腾310卡中第0个核心。
    - context：存储创建后的统一设备上下文实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterContext_destroy
  ```c++
  void NNAdapterContext_destroy(NNAdapterContext* context)
  ```
  销毁统一设备上下文实例。
  - 参数：
    - context：需要销毁的统一设备上下文实例。
  - 返回值：无。

- NNAdapterModel_create
  ```c++
  int NNAdapterModel_create(NNAdapterModel** model)
  ```
  创建一个空的、与设备无关的模型实例。
  - 参数：
    - model：存储创建后的模型实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterModel_destroy
  ```c++
  void NNAdapterModel_destroy(NNAdapterModel* model)
  ```
  销毁模型实例及相关资源。
  - 参数：
    - model：需要销毁的模型实例。
  - 返回值：无。

- NNAdapterModel_finish
  ```c++
  int NNAdapterModel_finish(NNAdapterModel* model)
  ```
  结束模型组网。
  - 参数：
    - model：模型实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterModel_addOperand
  ```c++
  int NNAdapterModel_addOperand(NNAdapterModel* model, const NNAdapterOperandType* type, NNAdapterOperand** operand)
  ```
  向模型中增加一个操作数，即神经网络模型中的张量。
  - 参数：
    - model：模型实例。
    - type：操作数的类型，由`NNAdapterOperandType`定义，包含精度类型、数据布局、生命周期、维度信息和量化信息。
    - operand：存储新增的操作数实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterModel_setOperandValue
  ```c++
  int NNAdapterModel_setOperandValue(NNAdapterOperand* operand, void* buffer, uint32_t length, bool copy)
  ```
  设置常量操作数的值。
  - 参数：
    - operand：操作数实例。
    - buffer：常量数据的内存地址。
    - lenght：常量数据的内存大小（字节）。
    - copy：是否创建常量数据的内存副本，否则将直接引用`buffer`。后者要求在模型编译前都不允许修改`buffer`指向的内容。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterModel_getOperandType
  ```c++
  int NNAdapterModel_getOperandType(NNAdapterOperand* operand,  NNAdapterOperandType** type)
  ```
  查询操作数的类型。
  - 参数：
    - operand：操作数实例。
    - type：存储返回的操作数类型。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterModel_addOperation
  ```c++
  int NNAdapterModel_addOperation(NNAdapterModel* model, NNAdapterOperationType type, uint32_t input_count, NNAdapterOperand** input_operands, uint32_t output_count, NNAdapterOperand** output_operands, NNAdapterOperation** operation)
  ```
  向模型中增加一个操作符，并设置它的输入、输出操作数，即神经网络模型中的算子。
  - 参数：
    - model：模型实例。
    - type：操作符类型，由`NNAdapterOperationCode`定义，包含二维卷积`NNADAPTER_CONV_2D`，最大值池化`NNADAPTER_AVERAGE_POOL_2D`，均值池化`NNADAPTER_MAX_POOL_2D`等操作符。
    - input_count：输入操作数的数量。
    - input_operands：输入操作数列表，需严格按照每一个操作符的定义依次将对应的输入操作数加入到列表中。
    - output_count：输出操作数的数量。
    - output_operands：输出操作数列表，需严格按照每一个操作符的定义依次将对应的输出操作数加入到列表中。
    - operation：存储新增的操作符实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterModel_identifyInputsAndOutputs
  ```c++
  int NNAdapterModel_identifyInputsAndOutputs(NNAdapterModel* model, uint32_t input_count, NNAdapterOperand** input_operands, uint32_t output_count, NNAdapterOperand** output_operands)
  ```
  标识模型的输入、输出操作数，其生命周期将被标记为`NNADAPTER_MODEL_INPUT`和`NNADAPTER_MODEL_OUTPUT`类型。
  - 参数：
    - model：模型实例。
    - input_count：输入操作数的数量。
    - input_operands：输入操作数列表，不约束每一个操作符顺序。
    - output_count：输出操作数的数量。
    - output_operands：输出操作数列表，不约束每一个操作符顺序。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterCompilation_create
  ```c++
  int NNAdapterCompilation_create(NNAdapterModel* model, const char* cache_token, void* cache_buffer, uint32_t cache_length, const char* cache_dir, NNAdapterContext* context, NNAdapterCompilation** compilation)
  ```
  创建一个编译实例，基于指定的统一设备上下文，为多种设备（当前版本仅支持一种设备）编译模型实例或直接加载模型缓存。如果同时设置模型实例和模型缓存参数，则优先加载模型缓存，因此存在以下三种情况：

  1）当设置`cache_token`，`cache_buffer`和`cache_length`时，则直接从内存中加载模型缓存，此时将忽略`model`参数。

  2）当设置`cache_token`和`cache_dir`时，将从<`cache_dir`>指定的目录中查找并尝试加载<`cache_token`>.nnc模型缓存文件，成功加载后将忽略`model`参数，否则在调用`NNAdapterCompilation_finish`完成模型实例`model`的在线编译后，在<`cache_dir`>目录中生成<`cache_token`>.nnc文件。

  3）当`cache_token`，`cache_buffer`，`cache_length`和`cache_dir`均未被设置时，则在调用`NNAdapterCompilation_finish`后完成模型实例`model`的在线编译。需要注意的是，由于未设置`cache_token`和`cache_dir`，在编译完成后将不会生成模型缓存文件，将使得在模型首次推理时都会进行模型的在线编译，导致首次推理耗时过长。

  - 参数：
    - model：模型实例。
    - cache_token：模型缓存唯一标识。
    - cache_buffer：模型缓存的内存地址。
    - cache_length：模型缓存的内存大小（字节），必须与`cache_buffer`成对使用。
    - cache_dir：模型缓存的目录。
    - context：统一设备上下文实例。
    - compilation：存储创建的编译实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterCompilation_destroy
  ```c++
  void NNAdapterCompilation_destroy(NNAdapterCompilation* compilation)
  ```
  销毁编译实例。
  - 参数：
    - compilation：需要销毁的编译实例。
  - 返回值：无。

- NNAdapterCompilation_finish
  ```c++
  int NNAdapterCompilation_finish(NNAdapterCompilation* compilation)
  ```
  结束编译配置的设置，调用设备HAL层库对`NNAdapterCompilation_create`中的模型实例`model`进行在线编译并生成设备程序。
  - 参数：
    - compilation：编译实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterCompilation_queryInputsAndOutputs
  ```c++
  int NNAdapterCompilation_queryInputsAndOutputs(NNAdapterCompilation* compilation, uint32_t* input_count, NNAdapterOperandType** input_types, uint32_t* output_count, NNAdapterOperandType** output_types)
  ```
  查询编译后的模型的输入、输出操作数的数量和类型，必须在`NNAdapterCompilation_finish`执行后才能调用，可以通过以下两次调用获得输入、输出操作数数量和类型信息。

  1）当`input_types`和`output_types`为NULL时，则仅查询输入、输出操作数的数量并将值存储在`input_count`和`output_count`。

  2）当`input_types`和`output_types`不为NULL时，则将输入、输出操作数的类型依次存储在`input_types`和`output_types`（要求调用方根据`input_count`和`output_count`分配它们的内存）。

  - 参数：
    - compilation：编译实例。
    - input_count：存储返回的输入操作数的数量，不允许为NULL。
    - input_types：存储返回的输入操作数列表。
    - output_count：存储返回的输出操作数的数量，不允许为NULL。
    - output_types：存储返回的输出操作数列表。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterExecution_create
  ```c++
  int NNAdapterExecution_create(NNAdapterCompilation* compilation, NNAdapterExecution** execution)
  ```
  基于编译实例创建一个执行计划实例。
  
  为了方便理解`NNAdapterCompilation`和`NNAdapterExecution`的区别，可以将`NNAdapterCompilation`简单理解为已经编译好的设备代码，而`NNAdapterExecution`代表如何执行它，可以是顺序依次执行，也可以并行执行，可以是同步执行，也可以是异步执行，但目前NNAdapter仅支持同步顺序执行。

  - 参数：
    - compilation：编译实例。
    - execution：存储创建的执行计划实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterExecution_destroy
  ```c++
  void NNAdapterExecution_destroy(NNAdapterExecution* execution)
  ```
  销毁执行计划实例。
  - 参数：
    - execution：需要销毁的执行计划实例。
  - 返回值：无。

- NNAdapterExecution_setInput
  ```c++
  int NNAdapterExecution_setInput(NNAdapterExecution* execution, int32_t index, void* memory, void* (*access)(void* memory, NNAdapterOperandType* type))
  ```
  设置执行计划输入操作数的内存实例和访问函数。

  为了能够让HAL层库更加灵活的访问推理框架的张量对象，在设置执行计划的输入时，要求设置内存实例`memory`和内存实例访问函数`access`，例如：

  ```c++
  typedef struct {
    NNAdapterOperandPrecisionCode precision;
    uint32_t dimensions_count;
    int32_t dimensions_data[NNADAPTER_MAX_SIZE_OF_DIMENSIONS];
    void* buffer;
    size_t length;
  } Memory;

  void* access_input_memory(void* memory, NNAdapterOperandType* type) {
    Memory* handle = reinterpret_cast<Memory*>(memory);
    // Return the dimensions and the host buffer to HAL
    memcpy(type->dimensions.data, handle->dimensions_data, handle->dimensions_count);
    return handle->buffer;
  }
  
  Memory input;
  NNAdapterExecution_setInput(execution, index, reinterpret_cast<void*>(&input), access_input_memory);
  ```

  - 参数：
    - execution：执行计划实例。
    - index：模型输入操作数的索引。
    - memory：模型输入操作数的内存实例，不限定为具体的缓存首地址，用户可自行封装后通过std::reinterpret_cast<void*>()强制转为void*类型。
    - access：内存实例访问函数，HAL层库将通过`access`函数访问`memory`获得host端缓存实际地址。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterExecution_setOutput
  ```c++
  int NNAdapterExecution_setOutput(NNAdapterExecution* execution, int32_t index, void* memory, void* (*access)(void* memory, NNAdapterOperandType* type))
  ```
  设置执行计划输出操作数的内存实例和访问函数。

  基于`NNAdapterExecution_setInput`示例中的`memory`的定义实现输出内存实例的访问函数`access`：

  ```c++
  void* access_output_memory(void* memory, NNAdapterOperandType* type) {
    Memory* handle = reinterpret_cast<Memory*>(memory);
    // Get the buffer length according to the type->precision and type->dimensions
    size_t request_length = GetBufferLength(type);
    if (request_length > handle->length) {
      free(handle->buffer);
      handle->buffer = malloc(request_length);
      assert(handle->buffer);
      handle->length = request_length;
    }
    // Tell the inference framework the output dimensions and return the host buffer to HAL
    memcpy(handle->dimensions_data, type->dimensions.data, type->dimensions.count);
    handle->dimensions_count = type->dimensions.count;
    return handle->buffer;
  }
  
  Memory output;
  NNAdapterExecution_setOutput(execution, index, reinterpret_cast<void*>(&output), access_output_memory);
  ```

  - 参数：
    - execution：执行计划实例。
    - index：模型输出操作数的索引。
    - memory：模型输出操作数的内存实例，不限定为具体的缓存首地址，用户可自行封装后通过std::reinterpret_cast<void*>()强制转为void*类型。
    - access：内存实例访问函数，HAL层库将通过`access`函数访问`memory`获得host端缓存实际地址。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

- NNAdapterExecution_compute
  ```c++
  int NNAdapterExecution_compute(NNAdapterExecution* execution)
  ```
  同步调度执行计划实例。
  - 参数：
    - execution：执行计划实例。
  - 返回值：无。

### NNAdapter 标准算子详细说明
- NNADAPTER_ABS

  Applies the abs activation to the input tensor element-wise. The output is calculated using this formula: output = abs(input)
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D

  Applies adaptive 2-D average pooling across the input according to input and output size.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in, H_in, W_in].
    - 1: output_shape, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, with shape [2], with value [H_out, H_out].
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_ADAPTIVE_MAX_POOL_2D

  Applies adaptive 2-D max pooling across the input according to input and output size.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in, H_in, W_in].
    - 1: output_shape, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, with shape [2], with value [H_out, H_out].
    - 2: return_indices, a NNADAPTER_BOOL8 scalar, whether to return index of output, default to false.
    - 3: return_indices_dtype, a NNADAPTER_INT32 scalar, must be one of NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64, specifies the dtype of the indices.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.
    - 1: indices, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, with the same shape as output, indicates the indices of the current feature map.

- NNADAPTER_ADD

  Performs element-wise binary addition(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
    - 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the result, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_ARG_MAX

  Computes the indices of the max elements of the input tensor’s element along the provided axis.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axis, a NNADAPTER_TENSOR_INT32 scalar, the axis in which to compute the arg indices, it should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
    - 2: keepdim, a NNADAPTER_BOOL8 scalar, keep the reduced dimension or not, If TRUE, keep the reduced dimension.
    - 3: dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64, specifies the dtype of the result,default to NNADAPTER_TENSOR_INT64.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor.

- NNADAPTER_ARG_MIN

  Computes the indices of the min elements of the input tensor’s element along the provided axis.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axis, a NNADAPTER_TENSOR_INT32 scalar. the axis in which to compute the arg indices, it should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
    - 2: keepdim, a NNADAPTER_BOOL8 scalar, keep the reduced dimension or not, If TRUE, keep the reduced dimension.
    - 3: dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64, specifies the dtype of the result, default to NNADAPTER_TENSOR_INT64.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor.

- NNADAPTER_ASSIGN

  Copy the input to the output.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_EQUAL

  Performs element-wise binary equal relational operation(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html). The output is calculated using this formula: output = input0 == input1
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_BOOL8, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64,NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_BOOL8 tensor.

- NNADAPTER_AVERAGE_POOL_2D

  Applies a 2-D average pooling across the input according to kernel sizes, stride sizes, and pad lengths.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in, H_in, W_in].
    - 1: auto_pad, a NNADAPTER_INT32 scalar. 0 means "EXPLICIT" so that paddings is used. 1 means "SAME". 2 means "VALID". It must be one of NNAdapterAutoPadCode values.
    - 2: pads, a NNADAPTER_TENSOR_INT32 tensor, with shape [4] and data {height_top, height_bottom, width_left, width_right}, or with shape[0] and no data.
    - 3: kernel_shape, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {kernel_height, kernel_width}.
    - 4: strides, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {height_stride, width_stride}.
    - 5: ceil_mode, a NNADAPTER_BOOL8 scalar, whether to use ceil or floor (default) to compute the output shape, default to false.
    - 6: count_include_pad, a NNADAPTER_BOOL8 scalar, whether include pad pixels when calculating values for the edges, default to false.
    - 7: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its type is the same as input.
      - When ceil_mode=false,

        H_out = floor((H_in + padding_height_top + padding_height_bottom - filter_height) / stride_height + 1)

        W_out = floor((W_in + padding_width_left + padding_width_right - filter_width) / stride_width + 1)
      - When ceil_mode=true,

        H_out = ceil((H_in + padding_height_top + padding_height_bottom - filter_height) / stride_height + 1)

        W_out = ceil((W_in + padding_width_left + padding_width_right - filter_width) / stride_width + 1)

- NNADAPTER_BATCH_NORMALIZATION

  Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N,C,...]
    - 1: scale, a 1-D tensor with shape [C]. 1) If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
    - 2: bias, a 1-D tensor with shape [C]. 1) If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
    - 3: mean, a 1-D tensor with shape [C]. 1) If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
    - 4: variance, a 1-D tensor with shape [C]. 1) If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
    - 5: epsilon, a NNADAPTER_FLOAT32 scalar. Defaults to 1e-5. The small value added to the variance to prevent division by zero.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_CAST

  The operator casts the elements of `input` to a data type specified by the `dtype` argument.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_BOOL8, NNADAPTER_TENSOR_INT8, NNADAPTER_TENSOR_UINT8, NNADAPTER_TENSOR_INT16, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64, NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_FLOAT64 tensor.
    - 1: dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_INT32, NNADAPTER_INT64, NNADAPTER_FLOAT32, NNADAPTER_FLOAT64 etc. Specifies the dtype of the result.
  - Outputs:
    - 0: output, a tensor with the same shape as input.

- NNADAPTER_CLIP

  Clip all elements in input into the range [min, max]. The output is calculated using this formula: output = MIN(MAX(input, min), max).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: min, a 1-D tensor with the same type as input with shape[1].
    - 2: max, a 1-D tensor with the same type as input with shape[1].
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_CONCAT

  Concatenates a list of tensors into a single tensor along the given dimension. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
  - Inputs:
    - 0 ~ n-1: input0 ~ inputn-1, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axis, a NNADAPTER_INT32 scalar. It represents the dimension along which axis to concat on. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
  - Outputs:
    - 0: output, the result with the same type as the inputs.

- NNADAPTER_CONV_2D

  Performs a normal or depthwise 2-D convolution operation. The CONV_2D op computes a 2-D convolution based on the input, filter, strides, paddings, dilations, groups and etc.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in, H_in, W_in].
    - 1: filter, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL 4-D tensor.
      - For a normal convolution, the filter's shape is [C_out, C_in, filter_height, filter_width], where C_out and C_in is the number of the channels of output and input, filter_height and filter_width is the filter's kernel size in the 'H' and 'W' dimension.
      - For a depthwise convolution, the filter's shape is [C_out, 1, filter_height, filter_width], where C_out is the number of the channels of output, filter_height and filter_width is the filter's kernel size in the 'H' and 'W' dimension.
    - 2: bias, a 1-D tensor with shape [C_out].
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
      - If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER, its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER, and bias_scale == input_scale * filter_scale.
      - If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, and bias_scale[i] = input_scale * filter_scale[i] for each output channel.
    - 3: auto_pad, a NNADAPTER_INT32 scalar. 0 means "EXPLICIT" so that paddings is used. 1 means "SAME". 2 means "VALID". It must be one of NNAdapterAutoPadCode.
    - 4: pads, a NNADAPTER_TENSOR_INT32 tensor, with shape [4] and data {height_top, height_bottom, width_left, width_right}, or with shape[0] and no data.
    - 5: strides, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {height_stride, width_stride}.
    - 6: group, a NNADAPTER_INT32 scalar.
      - For a normal convolution, group must be 1.
      - For a depthwise convolution, the formula should be satisfied: group=C_out=C_in.
    - 7: dilations, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {dilations_height, dilations_width}.
    - 8: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its type is the same as input.

      H_out = (H_in + padding_height_top + padding_height_bottom - (dilation_height * (filter_height - 1) + 1)) / stride_height + 1

      W_out = (W_in + padding_width_left + padding_width_right - (dilation_width * (filter_width - 1) + 1)) / stride_width + 1

- NNADAPTER_CONV_2D_TRANSPOSE

  Performs the transpose of 2-D convolution operation(also called deconvolution) based on the input, filter, strides, paddings, dilations, groups and etc.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in, H_in, W_in].
    - 1: filter, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL 4-D tensor. The filter's shape is [C_out, C_in, filter_height, filter_width], where C_out and C_in is the number of the channels of output and input, filter_height and filter_width is the filter's kernel size in the 'H' and 'W' dimension.
    - 2: bias, a 1-D tensor with shape [C_out].
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
      - If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER, its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER, and bias_scale == input_scale * filter_scale.
      - If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, and bias_scale[i] = input_scale * filter_scale[i] for each output channel.
    - 3: auto_pad, a NNADAPTER_INT32 scalar. 0 means "EXPLICIT" so that paddings is used. 1 means "SAME". 2 means "VALID". It must be one of NNAdapterAutoPadCode.
    - 4: pads, a NNADAPTER_TENSOR_INT32 tensor, with shape [4] and data {height_top, height_bottom, width_left, width_right}, or shape[0] and no data.
    - 5: strides, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {height_stride, width_stride}.
    - 6: group, a NNADAPTER_INT32 scalar.
      - For a normal convolution, group must be 1.
      - For a depthwise convolution, the formula should be satisfied: group=C_out=C_in.
    - 7: dilations, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {dilations_height, dilations_width}.
    - 8: output_padding, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {output_pad_height, output_pad_width}, or shape[0] and no data.
    - 9: output_shape, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, with shape [2] and data {output_height, output_width}, or shape[0] and no data.
    - 10: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its type is the same as input.

      H_out = (H_in - 1) * stride_height - padding_height_top - padding_height_bottom + (dilation_height * (filter_height - 1)) + 1 + output_padding_height

      W_out = (W_in - 1) * stride_width - padding_width_left - padding_width_right + (dilation_width * (filter_width - 1) + 1)) + 1 + output_padding_width

- NNADAPTER_CUM_SUM

  Performs cumulative sum of the input elements along the given axis.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axis, a NNADAPTER_INT32 scalar, default to -1. It represents the dimension along which softmax will be performed. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
    - 2: exclusive, a NNADAPTER_NOOL8 scalar. If set to true, the top element will not be include, default to false.
    - 3: reverse, a NNADAPTER_NOOL8 scalar, whether to perform the cumsum in the reversed direction, default to false.
  - Outputs:
    - 0: output, a tensor with the same type as input.

- NNADAPTER_DEFORMABLE_CONV_2D

  Compute 2-D deformable convolution on 4-D input.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in, H_in, W_in].
    - 1: offset, a tensor with the same type as input. It's shape is [N, 2 * deformable_groups * H_f * W_f, H_in, W_in].
    - 2: mask, a tensor with the same type as input. It's shape is [N, deformable_groups * H_f * W_f, H_in, W_in].
    - 3: filter, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL 4-D tensor.
      - For a normal convolution, the filter's shape is [C_out, C_in, filter_height, filter_width], where C_out and C_in is the number of the channels of output and input, filter_height and filter_width is the filter's kernel size in the 'H' and 'W' dimension.
      - For a depthwise convolution, the filter's shape is [C_out, 1, filter_height, filter_width], where C_out is the number of the channels of output, filter_height and filter_width is the filter's kernel size in the 'H' and 'W' dimension.
    - 4: bias, a 1-D tensor with shape [C_out].
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
      - If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER, its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER, and bias_scale == input_scale * filter_scale.
      -  If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, and bias_scale[i] = input_scale * filter_scale[i] for each output channel.
    - 5: pads, a NNADAPTER_TENSOR_INT32 tensor, with shape [4] and data {height_top, height_bottom, width_left, width_right}, or with shape[0] and no data.
    - 6: strides, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {height_stride, width_stride}.
    - 7: group, a NNADAPTER_INT32 scalar.
      - For a normal convolution, group must be 1.
      - For a depthwise convolution, the formula should be satisfied: group=C_out=C_in.
    - 8: deformable_group, a NNADAPTER_INT32 scalar. Specify the c-axis grouping number of input.
    - 9: dilations, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {dilations_height, dilations_width}.
    - 10: fuse_code, A NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its type is the same as input.

      H_out = (H_in + padding_height_top + padding_height_bottom - (dilation_height * (filter_height - 1) + 1)) / stride_height + 1

      W_out = (W_in + padding_width_left + padding_width_right - (dilation_width * (filter_width - 1) + 1)) / stride_width + 1

- NNADAPTER_DIV

  Performs element-wise binary division(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
    - 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the result, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_EXP

  Applies the exp activation to the input tensor element-wise. The output is calculated using this formula: output = e^input
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_EXPAND

  Broadcast the input tensor following the given shape(by Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: shape, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor. It indicates the shape you want to expand to, following the broadcast rule.
  - Outputs:
    - 0: output, a tensor with the same type as input.

- NNADAPTER_FILL

  Produces a tensor with the `shape` and `value`.
  - Inputs:
    - 0: shape, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor.
    - 1: value, a NNADAPTER_FLOAT32, NNADAPTER_INT32, NNADAPTER_INT64 or NNADAPTER_BOOL scalar.
  - Outputs:
    - 0: output, a tensor with the `shape` and `value`.

- NNADAPTER_FLATTEN

  Flattens the input tensor according to a contiguous range of axes from `start_axis` to `stop_axis`.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: start_axis, a NNADAPTER_INT32 scalar, the start axis to flatten.
    - 2: end_axis, a NNADAPTER_INT32 scalar, the end axis to flatten.
  - Outputs:
    - 0: output, a tensor with the same type as input.

- NNADAPTER_FULLY_CONNECTED

  Add a fully connected layer. The output is calculated using this formula: output = activation(input * weight' + bias).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor of at least rank 2, If its rank is greater than 2, it will be flattened to a 2-D Tensor with the shape [batch_size, input_size], where input_size represents the number of inputs, matching the second dimension of weight, and batch_size is calculated by dividing the number of elements by input_size.
    - 1: weight, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL 2-D tensor with shape [num_units, input_size], where the num_units represents the number of output units, which also means the feature size of output.
    - 2: bias, a 1-D tensor with shape [num_units].
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
      - If weight's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER, its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER, and bias_scale == input_scale * weight_scale.
      - If weight's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, and bias_scale[i] = input_scale * weight_scale[i] for each output channel.
    - 3: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, a 2-D tensor with shape [batch_size, num_units], and its type is the same as input.

- NNADAPTER_GATHER

  Gathers entries of axis dimension of `input` indexed by `indices`, and concatenates them together.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor, of any rank R.
    - 1: indices, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, of any rank Q. All index values are expected to be within bounds [-S, S-1] along axis of size S.
    - 2: axis, A NNADAPTER_INT32 scalar. It represents the dimension along which gather will be performed. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
  - Outputs:
    - 0: output, a tensor with the same type as input, of rank with rank Q + (R - 1).

- NNADAPTER_GELU

  Applies the Gaussian Error Linear Units activation to the input tensor element-wise. Refer to https://arxiv.org/abs/1606.08415 for more details.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: approximate, a NNADAPTER_BOOL8 scalar, whether to enable approximation.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_GREATER

  Performs element-wise binary greater relational operation(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html): output = input0 > input1.
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_BOOL8, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64,NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_BOOL8 tensor.

- NNADAPTER_GREATER_EQUAL

  Performs element-wise binary greater_equal relational operation(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html): output = input0 >= input1.
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_BOOL8, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64,
NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_BOOL8 tensor.

- NNADAPTER_HARD_SIGMOID

  Applies the hard-sigmoid activation to the input tensor element-wise. The output is calculated using this formula: output = max(0, min(1, alpha * input + beta)).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: alpha, a NNADAPTER_FLOAT32 scalar.
    - 2: beta, a NNADAPTER_FLOAT32 scalar.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_HARD_SWISH

  Applies the hard-swish activation to the input tensor element-wise. The output is calculated using this formula: output = input * max(0, min(1, alpha * input + beta)).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: alpha, a NNADAPTER_FLOAT32 scalar.
    - 2: beta, a NNADAPTER_FLOAT32 scalar.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_INSTANCE_NORMALIZATION

  Applies Instance Normalization over a N-D input (N>2) as described in the paper https://arxiv.org/abs/1607.08022. output = scale * (input - mean) / sqrt(variance + epsilon) + bias, where mean and variance are computed per instance per channel.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N,C,...].
    - 1: scale, a tensor, with shape [C].
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
    - 2: bias, a tensor with the same shape as scale.
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
    - 3: epsilon, a NNADAPTER_FLOAT32 scalar, the small value added to the variance to prevent division by zero, default to 1e-5.
    - 4: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_LAYER_NORMALIZATION

  Applies Layer Normalization over a N-D input described in the paper Layer Normalization: <https://arxiv.org/pdf/1607.06450v1.pdf>.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N,C,...].
    - 1: scale, a tensor, shape is performed along the input dimension from begin_norm_axis to the rank of input.
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
    - 2: bias, a tensor with the same shape as scale.
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
    - 3: begin_norm_axis, a NNADAPTER_INT32 scalar, indicates that the normalization will be performed along the dimension from begin_norm_axis to the rank of input, default to 1.
    - 4: epsilon, a NNADAPTER_FLOAT32 scalar, default to 1e-5.
    - 5: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_LEAKY_RELU

  Applies the Leaky ReLU activation to the input tensor element-wise. The output is calculated using this formula: output = input, if input >=0; output = alpha * input, if input < 0.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: alpha, a NNADAPTER_FLOAT32 scalar.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_LESS

  Performs element-wise binary less relational operation(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html): output = input0 < input1.
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_BOOL8, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_BOOL8 tensor.

- NNADAPTER_LESS_EQUAL

  Performs element-wise binary less_equal relational operation(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html): output = input0 <= input1.
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_BOOL8, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64,NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_BOOL8 tensor.

- NNADAPTER_LOG

  Applies the log activation to the input tensor element-wise. The output is calculated using this formula: output = log(input).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_LP_NORMALIZATION

  Applies the Lp Normalization to the input tensor element-wise. The output is calculated using this formula: output = sum(abs(input)), if p = 1; output = sqrt(sum(input^2)), if p = 2.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axis, an 1-D NNADAPTER_TENSOR_INT32, default to [1]. It represents the dimension along which softmax will be performed. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
    - 2: p, a NNADAPTER_INT32 scalar. The exponent value in the norm formulation, only 1 or 2 are supported, default to 2.
    - 3: epsilon, a NNADAPTER_FLOAT32 scalar, specifying the lower limit of normalization.
    - 4: keepdim, a NNADAPTER_BOOL8 scalar, keep the reduced dimension or not, default to true.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_MAT_MUL

  Matrix product that behaves like numpy.matmul.
  - Inputs:
    - 0: input0, A NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
    - 2: transpose_input0, a NNADAPTER_BOOL8 scalar, whether to transpose the last two dimensions of input0 before multiplication.
    - 3: transpose_input1, a NNADAPTER_BOOL8 scalar, whether to transpose the last two dimensions of input1 before multiplication.
  - Outputs:
    - 0: output, a tensor with the same type as two inputs.

- NNADAPTER_MAX

  Performs element-wise binary maximum(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
    - 2: fuse_code, a NNADAPTER_INT32 scalar, specifies the activation to the result, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_MAX_POOL_2D

  Applies a 2-D max pooling across the input according to kernel sizes, stride sizes, and pad lengths.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in, H_in, W_in].
    - 1: auto_pad, a NNADAPTER_INT32 scalar. 0 means 'EXPLICIT' so that paddings is used. 1 means 'SAME'. 2 means 'VALID'. It must be one of NNAdapterAutoPadCode values.
    - 2: pads, a NNADAPTER_TENSOR_INT32 tensor, with shape [4] and data {height_top, height_bottom, width_left, width_right}, or with shape[0] and no data.
    - 3: kernel_shape, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {kernel_height, kernel_width}.
    - 4: strides, a NNADAPTER_TENSOR_INT32 tensor, with shape [2] and data {height_stride, width_stride}.
    - 5: ceil_mode, a NNADAPTER_BOOL8 scalar, whether to use ceil(true) or floor(false) to compute the output shape, default to false.
    - 6: return_indices, A NNADAPTER_BOOL8 scalar, whether to return index of output, default to false.
    - 7: return_indices_dtype, a NNADAPTER_INT32 scalar, must be one of NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64, specifies the dtype of the indices.
    - 8: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its type is the same as input.
      - When ceil_mode=false,

        H_out = floor((H_in + padding_height_top + padding_height_bottom - filter_height) / stride_height + 1)

        W_out = floor((W_in + padding_width_left + padding_width_right - filter_width) / stride_width + 1)

      - When ceil_mode=true,

        H_out = ceil((H_in + padding_height_top + padding_height_bottom - filter_height) / stride_height + 1)

        W_out = ceil((W_in + padding_width_left + padding_width_right - filter_width) / stride_width + 1)
    
    - 1: indices, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, with the same shape as output, indicates the indices of the current feature map.

- NNADAPTER_MIN

  Performs element-wise binary minimum(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
    - 2: fuse_code, a NNADAPTER_INT32 scalar, specifies the activation to the result, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_MUL

  Performs element-wise binary multiplication(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
    - 2: fuse_code, a NNADAPTER_INT32 scalar, specifies the activation to the result, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_NOT_EQUAL

  Performs element-wise binary not_equal relational operation(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html). The output is calculated using this formula: output = input0 != input1.
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_BOOL8, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_BOOL8 tensor.

- NNADAPTER_PAD

  Pads input according to the specified `pads`, `mode` and `value`.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: pads, a NNADAPTER_TENSOR_INT32 1-D tensor, with shape [2 * input_rank], with value [x0_begin, x0_end, x1_begin, x1_end,...].
    - 2: mode, a NNADAPTER_INT32 scalar, supported pad modes: `constant`(default), `reflect`, `edge`, should be a value of NNAdapterPadModeCode.
    - 3: value, a scalar with the same type as input, only be used if the mode is `constant`.
  - Outputs:
    - 0: output, the result with the same type as input.

- NNADAPTER_POW

  Performs element-wise binary pow(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html). The output is calculated using this formula: output = input0^input1.
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 2: fuse_code, a NNADAPTER_INT32 scalar, specifies the activation to the result, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the result with the same type as input.

- NNADAPTER_PRELU

  Applies the prelu activation to the input tensor. The output is calculated using this formula: output = input, if input >= 0; output = slope * input, if input < 0.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32 or NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N, C, ...].
    - 1: slope, a tensor, with shape [1] or [C].
      - If input's type is NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_RANGE

  Produces a 1-D tensor with values from `start` to `end` with step `step`.
  - Inputs:
    - 0: start, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor with shape[1].
    - 1: end, a tensor with the same shape and type as `start`.
    - 2: step, a tensor with the same shape and type as `start`.
  - Outputs:
    - 0: output, a 1-D tensor with the same type as `start`.

- NNADAPTER_REDUCE_MEAN

  Computes the mean of the input’s elements along the specified axis. If axis has no data, mean is calculated over all elements of input. If keepdims equal 0, the resulted tensor have the reduced dimension pruned.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axes, a NNADAPTER_TENSOR_INT32 tensor, indicates the dimensions to perform mean calculations. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R. If axis has no data, mean is calculated over all elements of input.
    - 2: keepdim, a NNADAPTER_BOOL8 scalar, keeps the reduced dimension or not, default to true.
  - Outputs:
    - 0: output, a tensor with the same type as input.

- NNADAPTER_RELU

  Applies rectified linear activation to the input tensor element-wise. The output is calculated using this formula: output = max(0, input).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_RELU6

  Applies rectified linear 6 activation to the input tensor element-wise. The output is calculated using this formula: output = min(6, max(0, input)).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_RESHAPE

  Reshapes a tensor similar to numpy.reshape. The output tensor has the same data as the input tensor but with a new shape.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: shape, an 1-D NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 shape tensor which specifies the new shape, At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions. a dimension could also be 0, in which case the actual dimension value is unchanged.
  - Outputs:
    - 0: output, a tensor with a new shape, and its type and data is same as input.

- NNADAPTER_RESIZE_NEAREST

  Resizes the input tensor using the nearest interpolation.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N, C, ...].
    - 1: shape, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, indicates the target shape of output exclude dim_N and dim_C.
    - 2: scales, a NNADAPTER_TENSOR_FLOAT32 tensor, indicates the scale of the output's shape exclude dim_N and dim_C.
    - 3: align_corners, a NNADAPTER_BOOL scalar. If True, the centers of the 4 corner pixels of the input and output tensors are aligned, preserving the values at the corner pixels.
  - Outputs:
    - 0: output, a tensor with the same type as input.

- NNADAPTER_RESIZE_LINEAR

  Resizes the input tensor using the linear interpolation.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N, C, ...].
    - 1: shape, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, indicates the target shape of output exclude dim_N and dim_C.
    - 2: scales, a NNADAPTER_TENSOR_FLOAT32 tensor, indicates the scale of the output's shape exclude dim_N and dim_C.
    - 3: align_corners, NNADAPTER_BOOL scalar. If True, the centers of the 4 corner pixels of the input and output tensors are aligned, preserving the values at the corner pixels.
    - 4: align_mode, a NNADAPTER_INT32 scalar, optional for linear interpolation. It can be ‘0’ for src_idx = scale_factor * (dst_indx + 0.5) - 0.5, can be ‘1’ for src_idx = scale_factor * dst_index.
  - Outputs:
    - 0: output, a tensor with the same type as input.

- NNADAPTER_SHAPE

  Outputs an 1D tensor containing the shape of the input tensor.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_INT32 tensor.
    - 1: dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64, specifies the dtype of the result.
  - Outputs:
    - 0: output, a NNADAPTER_TENSOR_INT32 tensor.

- NNADAPTER_SIGMOID

  Applies sigmoid activation to the input tensor element-wise. The output is calculated using this formula: output = 1 / (1 + exp(-input)).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_SLICE

  This operator produces a slice of input along multiple axes. Similar to numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html Slice uses `axes`, `starts`, `ends` and `steps` to specify the start and end dimension and step for each axis in the list of axes, it uses this information to slice the input data tensor. If a negative value is passed to starts or ends such as −i, it represents the reverse position of the axis i−1 (here 0 is the initial position). If the value passed to starts or ends is greater than n (the number of elements in this dimension), it represents n. For slicing to the end of a dimension with unknown size, it is recommended to pass in INT_MAX. The size of axes must be equal to starts and ends.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axes, An optional NNADAPTER_TENSOR_INT32 tensor that `starts` and `ends` apply to, will be treated as [0, 1, ..., len(`starts`) - 1] if it is empty.
    - 2: starts, starts indices of corresponding axis in `axes`, a NNADAPTER_TENSOR_INT32 tensor.
    - 3: ends, ends indices of corresponding axis in `axes`, a NNADAPTER_TENSOR_INT32 tensor.
    - 4: steps, a NNADAPTER_TENSOR_INT32 1-D tensor, 1-D tensor of slice step of corresponding axis in `axes`. Negative value means slicing backward. `steps` cannot be 0. Defaults to 1.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_SOFTMAX

  Computes the normalized exponential values for the input tensor element-wise. The output is calculated using this formula: output = exp(input) / reduce_sum(exp(input), axis=axis, keepdims=true).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axis, a NNADAPTER_INT32 scalar. Defaults to 1. It represents the dimension along which softmax will be performed. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_SPLIT

  Split a tensor into a list of tensors along the specified axis.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axis, a NNADAPTER_INT32 scalar. It represents the dimension along which axis to split. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
    - 2: split, An 1-D NNADAPTER_TENSOR_INT32 tensor, each of values indicates the length of each output. Sum of the values must be equal to the dimension at `axis` specified.
  - Outputs:
    - 0 ~ n-1: output0 ~ outputn-1, the results with the same type as the input.

- NNADAPTER_SQUEEZE

  Returns a tensor with all the dimensions of input of size 1 removed.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axes, a NNADAPTER_TENSOR_INT32 tensor, indicates the dimensions to be squeezed, default to None. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
  - Outputs:
    - 0: output, a tensor with the same type as input.

- NNADAPTER_STACK

  Concatenates a sequence of tensors into a single tensor along the specified axis. All input tensors must have the same shape.
  - Inputs:
    - 0 ~ n-1: input0 ~ inputn-1, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - n: axis, a NNADAPTER_INT32 scalar. It represents the dimension along which axis to concatenate. It should be in range [-R-1, R+1), where R is the rank of input, negative value works the same way as axis+R+1.
  - Outputs:
    - 0: output, the result with the same type as the inputs.

- NNADAPTER_SUB

  Performs element-wise binary subtraction(with Numpy-style broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: input1, a tensor with the same type as input0.
    - 2: fuse_code, a NNADAPTER_INT32 scalar, specifies the activation to the result, must be one of NNAdapterFuseCode values.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_SWISH

  Applies the Swish activation to the input tensor element-wise. The output is calculated using this formula: output = input / (1 + e ^ (-input)).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_TANH

  Applies the hyperbolic tangent activation to the input tensor element-wise. The output is calculated using this formula: output = tanh(input).
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.

- NNADAPTER_TOP_K

  Retrieve the top-K largest elements along a specified axis.
  - Inputs:
    - input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_INT32, NNADAPTER_TENSOR_INT64, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: k, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor, the number of top elements to look for along the axis.
    - 2: axis, a NNADAPTER_INT32 scalar, represents the dimension along which top_k will be performed. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R.
    - 3: largest, a NNADAPTER_BOOL8 scalar, whether to return the top-K largest or smallest elements.
    - 4: sorted, a NNADAPTER_BOOL8 scalar, whether to return the elements in sorted order.
    - 5: return_indices_dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64, specifies the dtype of the indices.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input, top K values from the input tensor.
    - 1: indices, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, the corresponding input tensor indices for the top K values.

- NNADAPTER_TRANSPOSE

  Transposes the input according to the perm, similar to numpy.transpose https://numpy.org/doc/stable/reference/generated/numpy.transpose.html. For example, the input with shape (1, 2, 3) and perm=(1, 0, 2), the shape of output will be (2, 1, 3).
  - Inputs:
    - 0: input0, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: perm, An optional 1-D NNADAPTER_TENSOR_INT32 tensor, reverse the dimensions of input if perm is not given, otherwise permute the axes according to the values given.
  - Outputs:
    - 0: output, a tensor with the same type as input.

- NNADAPTER_UNSQUEEZE

  Inserts a dimension of size 1 at the specified axis of the dimensions of input.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
    - 1: axes, A NNADAPTER_TENSOR_INT32 tensor, indicates the dimensions to be inserted. It should be in range [-R, R), where R is the rank of input, negative value works the same way as axis+R+1.
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.
