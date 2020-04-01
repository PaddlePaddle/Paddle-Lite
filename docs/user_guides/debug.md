# 调试

## Profiler工具

Basic profiler 用于 CPU 上kernel 耗时的统计。

### 开启方法:

参照 [编译安装](../user_guides/source_compile) 中的**full_publish**部分进行环境配置，在 cmake 时添加 `-DLITE_WITH_PROFILE=ON` ，就可以开启相应支持。

### 使用示例：

在模型执行完毕后，会自动打印类似如下 profiler 的日志

```
                        kernel   average       min       max     count
                feed/def/1/4/2         0         0         0         1
              conv2d/def/4/1/1      1175      1175      1175         1
              conv2d/def/4/1/1      1253      1253      1253         1
    depthwise_conv2d/def/4/1/1       519       519       519         1
              conv2d/def/4/1/1       721       721       721         1
     elementwise_add/def/4/1/1        18        18        18         1
              conv2d/def/4/1/1      2174      2174      2174         1
    depthwise_conv2d/def/4/1/1       380       380       380         1
              conv2d/def/4/1/1       773       773       773         1
     elementwise_add/def/4/1/1         2         2         2         1
              conv2d/def/4/1/1      1248      1248      1248         1
    depthwise_conv2d/def/4/1/1       492       492       492         1
              conv2d/def/4/1/1      1150      1150      1150         1
     elementwise_add/def/4/1/1        33        33        33         1
     elementwise_add/def/4/1/1         3         3         3         1
              conv2d/def/4/1/1      1254      1254      1254         1
    depthwise_conv2d/def/4/1/1       126       126       126         1
```

## Debug工具

**Lite Model Debug Tool** 是用来检查Paddle-Lite框架与Paddle-Fluid框架运行时tensor(包括variable与weight)之间diff信息的基础工具。

### 编译方法:

1. 参照 [编译安装](../user_guides/source_compile) 中的**full_publish**部分进行环境配置和编译。
2. 在生成的`build`目录下，执行`make lite_model_debug_tool`，`lite_model_debug_tool`产出在编译目录的`lite/tools/debug`目录下。

### 工作流程:

1. 运行 `/bin/bash check_model.sh --model_dir=<your_model_path> --build_root_dir=<your_cmake_root_dir> debug_cpp_stage` 获得模型在Paddle-Lite框架下的运行拓扑信息、varibles信息和weights信息。运行后拓扑信息将会存储在默认名为 `topo_file.txt` 的文件中，variables和weights信息将会存储在默认名为 `tensor_cpp.txt` 的文件中。
2. 运行 `/bin/bash check_model.sh --model_dir=<your_model_path> --build_root_dir=<your_cmake_root_dir> debug_py_stage`执行fluid框架预测以获取相同模型在fluid框架下的variable与weight信息(注意：我们使用fluid的python api运行fluid模型，因此您在运行此步之前应确保已正确安装fluid的python api)。然后debug tool将会自动比较Paddle-Lite框架输出的信息和Paddle-Fluid框架输出的信息来检查是否存在运行时diff。 执行Paddle-Fluid框架，输出的信息将会存储在默认名为 `tensor_py.txt` 的文件中，相应的diff信息将会存储在默认名为 `diff.txt`的文件中(默认情况下，只会输出执行拓扑序中第一个有diff的variable相关的信息)。

### 注意事项:

1. 输出的结果是在**执行完一次预测后**输出的相应变量/权重的最终值，因此如果您在预测过程进行过诸如变量复用/子图融合等优化方法，则相应的输出可能会出现偏差。
2. 默认情况下debug tools将以全1作为输入进行比对。
3. 默认情况下，为了保证与Paddle-Fluid框架的结果可比对，debug tool将会禁用掉所有的Paddle-Lite的优化策略。
4. Paddle-Lite框架的执行环境由与您的编译选项有关，比如您开启了LITE_WITH_ARM编译选项，那debug tool的`debug_cpp_stage`也需要在ARM平台下运行。

### Diff信息输出：

如果debug tool检测到diff信息，那么在`diff.txt`中将会输出类似以下结构信息

```c++
>>>>>>>>>>>>>>>>>>DIFF VARIABLE: dropout_0.tmp_0<<<<<<<<<<<<<<<<<<<
dropout	(X:pool2d_7.tmp_0)	(Mask:dropout_0.tmp_1 Out:dropout_0.tmp_0)
--------------- Tensor File info ---------------
pool2d_7.tmp_0	{1,1536,1,1}	0.749892 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0150336 0.621641 0.147099 0.636727 0.0 0.0 0.00410917 0.784708 0.0 0.0704846 0.233599 0.840123 0.239201 0.112878 0.0 0.155352 0.306906 0.0 0.0 0.860938 0.221037 0.787316 0.256585 ... 
dropout_0.tmp_0	{1,1536,1,1}	0.749892 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0150336 0.621641 0.147099 0.636727 0.0 0.0 0.00410917 0.784708 0.0 0.0704846 0.233599 0.840123 0.239201 0.112878 0.0 0.155352 0.306906 0.0 0.0 0.860938 0.221037 0.787316 0.256585 ...
--------------- Fluid Tensor info ---------------
pool2d_7.tmp_0	{1,1536,1,1}	0.7498912 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.015033395 0.6216395 0.14709876 0.63672537 0.0 0.0 0.0041093696 0.7847073 0.0 0.07048465 0.23359808 0.8401219 0.23919891 0.1128789 0.0 0.1553514 0.3069055 0.0 0.0 0.8609365 0.22103554 ...
dropout_0.tmp_0	{1,1536,1,1}	0.599913 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.012026716 0.4973116 0.117679015 0.5093803 0.0 0.0 0.0032874958 0.62776583 0.0 0.056387722 0.18687847 0.67209756 0.19135913 0.090303116 0.0 0.12428112 0.2455244 0.0 0.0 0.68874925 ... 
```

其中第二行为op相关信息，标明了执行哪个op出现了diff及其对应的输入输出变量名。Tensor File info为Paddle-Lite框架的输出信息，而Fluid Tensor info为Paddle-Fluid框架的相应输出信息。
示例中的`dropout_0.tmp_1`没有相应的tensor信息是因为工具检测到其在预测的后序流程中未被使用，因此不会对预测结果造成影响，从而将其自动屏蔽掉以保证输出尽量简洁。

### 其他选项：

| Option                      | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| --input_file                | 输入文件名，不同field以逗号分隔，相同field内以空格分隔, 只有文件中的第一行输入信息会被使用. 如果您不指定input_file，那么所有输入将会被置为1。注意：`debug_py_stage`目前不支持多field输入。 |
| --cpp_topo_file             | 存储运行时拓扑信息，由`debug_cpp_stage`写入并且由`debug_py_stage`读取使用。 默认为`topo_file.txt` 。 |
| --cpp_tensor_file           | 存储`debug_cpp_stage` 在运行拓扑序下的输出信息，默认为 `tensor_cpp.txt` 。 |
| --tensor_names              | 如果此选项不为空，那么只输出由此选项中指定名字的variable/weight信息，名字间用逗号分隔。 |
| --tensor_output_length      | 输出数据的长度，默认为全部输出。                             |
| --py_threshold              | 判断diff发生的阈值，默认为 `1e-5` 。                         |
| --py_tensor_file            | 存储`debug_py_stage` 在运行拓扑序下的输出信息，默认为`tensor_py.txt`. |
| --py_output_file            | diff信息的存储文件，默认为`diff.txt`。                       |
| --py_only_output_first_diff | 是否只输出运行时拓扑序中第一个有diff的var/op信息，默认为true |

您可以参考 `check_model.sh` 脚本中的代码以获得更多细节.
