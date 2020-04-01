
# C++ API

## CreatePaddlePredictor

```c++
template <typename ConfigT>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT&);
```

`CreatePaddlePredictor`用来根据`MobileConfig`构建预测器。

示例：

```c++
// 设置MobileConfig
MobileConfig config;
config.set_model_dir(FLAGS_model_dir);

// 根据MobileConfig创建PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
```

参数：

- `config(MobileConfig)` - 用于构建Predictor的配置信息。

返回：`PaddlePredictor`指针

返回类型：`std::shared_ptr<PaddlePredictor>`

## CxxConfig

```c++
class CxxConfig;
```

`CxxConfig`用来配置构建CxxPredictor的配置信息，如protobuf格式的模型地址、能耗模式、工作线程数、place信息等等。

示例：

```c++
config = CxxConfig()
# 设置模型目录，加载非combined模型时使用
config.set_model_dir(<your_model_dir_path>)
# 设置工作线程数
config.set_threads(4);
# 设置能耗模式
config.set_power_mode(PowerMode.LITE_POWER_NO_BIND)
# 设置valid places
places = [Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据CxxConfig创建CxxPredictor
predictor = create_paddle_predictor(config)
```

### `set_model_dir(model_dir)`

设置模型文件夹路径，当需要从磁盘加载非combined模型时使用。

参数：

- `model_dir(str)` - 模型文件夹路径

返回：`None`

返回类型：`None`



### `model_dir()`

返回设置的模型文件夹路径。

参数：

- `None`

返回：模型文件夹路径

返回类型：`str`



### `set_model_file(model_file)`

设置模型文件路径，加载combined形式模型时使用。

参数：

- `model_file(str)` - 模型文件路径

返回类型：`None`



### `model_file()`

获取设置模型文件路径，加载combined形式模型时使用。

参数：

- `None`

返回：模型文件路径

返回类型：`str`



### `set_param_file(param_file)`

设置模型参数文件路径，加载combined形式模型时使用。

参数：

- `param_file(str)` - 模型文件路径

返回类型：`None`



### `param_file()`

获取设置模型参数文件路径，加载combined形式模型时使用。

参数：

- `None`

返回：模型参数文件路径

返回类型：`str`



### `set_valid_places(valid_places)`

设置可用的places列表。

参数：

- `valid_places(list)` - 可用place列表。

返回类型：`None`

示例：

```c++
config = CxxConfig()
# 设置模型目录，加载非combined模型时使用
config.set_model_dir(<your_model_dir_path>)
# 设置valid places
# 注意，valid_places列表中Place的排序表明了用户对Place的偏好程度，如用户想优先使用ARM上Int8精度的
# kernel，则应把Place(TargetType.ARM, PrecisionType.INT8)置于valid_places列表的首位。
places = [Place(TargetType.ARM, PrecisionType.INT8),
          Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据CxxConfig创建CxxPredictor
predictor = create_paddle_predictor(config)
```



### `set_power_mode(mode)`

设置CPU能耗模式。若不设置，则默认使用`PowerMode.LITE_POWER_HIGH`。

*注意：只在开启`OpenMP`时生效，否则系统自动调度。此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `mode(PowerMode)` - CPU能耗模式

返回：`None`

返回类型：`None`



### `power_mode()`

获取设置的CPU能耗模式。

*注意：此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `None`

返回：设置的CPU能耗模式

返回类型：`PowerMode`



### `set_threads(threads)`

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启`OpenMP`的模式下生效，否则只使用单线程。此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `threads(int)` - 工作线程数

返回：`None`

返回类型：`None`



### `threads()`

获取设置的工作线程数。

*注意：此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `None`

返回：工作线程数

返回类型：`int`


### `set_x86_math_library_num_threads(threads)`

设置CPU Math库线程数，CPU核心数支持情况下可加速预测。默认为1，并且仅在x86下有效。

参数：

- `threads(int)` - CPU Math库线程数。

返回：`None`

返回类型：`None`


### `x86_math_library_num_threads()`

返回CPU Math库线程数，CPU核心数支持情况下可加速预测。仅在x86下有效。

参数：

- `None`

返回：CPU Math库线程数。

返回类型：`int`

## MobileConfig

```c++
class MobileConfig;
```

`MobileConfig`用来配置构建轻量级PaddlePredictor的配置信息，如NaiveBuffer格式的模型地址、模型的内存地址(从内存加载模型时使用)、能耗模式、工作线程数等等。

*注意：输入的模型需要使用[Model Optimize Tool](../user_guides/model_optimize_tool)转化为NaiveBuffer格式的优化模型。*

示例：

```c++
MobileConfig config;
// 设置NaiveBuffer格式模型目录，从文件加载模型时使用
config.set_model_from_file(<your_model_path>);
// 设置工作线程数
config.set_threads(4);
// 设置能耗模式
config.set_power_mode(LITE_POWER_HIGH);

// 根据MobileConfig创建PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
```

### `set_model_from_file(model_file)`

设置模型文件，当需要从磁盘加载模型时使用。

参数：

- `model_file(std::string)` - 模型文件路径

返回：`None`

返回类型：`void`

### `set_model_dir(model_dir)`

**注意**：Lite模型格式在release/v2.3.0之后修改，本接口为加载老格式模型的接口，将在release/v3.0.0废弃。建议替换为`set_model_from_file`接口。

设置模型文件夹路径，当需要从磁盘加载模型时使用。

参数：

- `model_dir(std::string)` - 模型文件夹路径

返回：`None`

返回类型：`void`



### `model_dir()`

返回设置的模型文件夹路径。

参数：

- `None`

返回：模型文件夹路径

返回类型：`std::string`

### `set_model_from_buffer(model_buffer)`

设置模型的内存数据，当需要从内存加载模型时使用。

参数：

- `model_buffer(std::string)` - 内存中的模型数据

返回：`None`

返回类型：`void`

### `set_model_buffer(model_buffer, model_buffer_size, param_buffer, param_buffer_size)`

**注意**：Lite模型格式在release/v2.3.0之后修改，本接口为加载老格式模型的接口，将在release/v3.0.0废弃。建议替换为`set_model_from_buffer`接口。

设置模型、参数的内存地址，当需要从内存加载模型时使用。

示例：

```c++
// 读取模型文件到内存
std::string model_buffer = ReadFile(FLAGS_model_path);
std::string params_buffer = lite::ReadFile(FLAGS_params_path);

// 设置MobileConfig
lite_api::MobileConfig config;
config.set_model_buffer(model_buffer.c_str(), model_buffer.size(), 
                        params_buffer.c_str(), params_buffer.size());

// 根据MobileConfig创建PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
```

参数：

- `model_buffer(const char*)` - 内存中模型结构数据。
- `model_buffer_size(size_t)` - 内存中模型结构数据的大小。
- `param_buffer(const char*)` - 内存中模型参数数据。
- `param_buffer_size(size_t)` - 内存中模型参数数据的大小。

返回：`None`

返回类型：`Void`



### `model_from_memory()`

是否从内存中加载模型，当使用`set_model_buffer`接口时返回`true`

参数：

- `None`

返回：是否从内存加载模型

返回类型：`bool`



### `model_buffer()`

获取内存中模型结构数据。

参数：

- `None`

返回：内存中模型结构数据

返回类型：`const std::string&`



### `param_buffer()`

获取内存中模型参数数据。

参数：

- `None`

返回：内存中模型结构数据

返回类型：`const std::string&`



### `set_power_mode(mode)`

设置CPU能耗模式。若不设置，则默认使用`LITE_POWER_HIGH`。

*注意：只在开启`OpenMP`时生效，否则系统自动调度。*

参数：

- `mode(PowerMode)` - CPU能耗模式

返回：`None`

返回类型：`void`



### `power_mode()`

获取设置的CPU能耗模式。

参数：

- `None`

返回：设置的CPU能耗模式

返回类型：`PowerMode`



### `set_threads(threads)`

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启`OpenMP`的模式下生效，否则只使用单线程。*

参数：

- `threads(int)` - 工作线程数

返回：`None`

返回类型：`void`



### `threads()`

获取设置的工作线程数。

参数：

- `None`

返回：工作线程数

返回类型：`int`

## PaddlePredictor

```c++
class PaddlePredictor
```

`PaddlePredictor`是Paddle-Lite的预测器，由`CreatePaddlePredictor`根据`MobileConfig`进行创建。用户可以根据PaddlePredictor提供的接口设置输入数据、执行模型预测、获取输出以及获得当前使用lib的版本信息等。

示例：

```c++
int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

// 设置MobileConfig
MobileConfig config;
config.set_model_dir(FLAGS_model_dir);

// 根据MobileConfig创建PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

// 获得模型的输入和输出名称
std::vector<std::string> input_names = predictor->GetInputNames();
for (int i = 0; i < input_names.size(); i ++) {
  printf("Input name[%d]: %s\n", i, input_names[i].c_str());
}
std::vector<std::string> output_names = predictor->GetOutputNames();
for (int i = 0; i < output_names.size(); i ++) {
  printf("Output name[%d]: %s\n", i, output_names[i].c_str());
}

// 准备输入数据
// (1)根据index获取输入Tensor
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
// (2)根据名称获取输入Tensor
// std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInputByName(input_names[0])));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}

// 执行预测
predictor->Run();

// 获取输出
// (1)根据index获取输出Tensor
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
// (2)根据名称获取输出Tensor
// std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(output_names[0])));
printf("Output dim: %d\n", output_tensor->shape()[1]);
for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
  printf("Output[%d]: %f\n", i, output_tensor->data<float>()[i]);
}
```

### `GetInput(index)`

获取输入Tensor指针，用来设置模型的输入数据。

参数：

- `index(int)` - 输入Tensor的索引

返回：第`index`个输入`Tensor`的指针

返回类型：`std::unique_ptr<Tensor>`



### `GetOutput(index)`

获取输出Tensor的指针，用来获取模型的输出结果。

参数：

- `index(int)` - 输出Tensor的索引

返回：第`index`个输出Tensor`的指针

返回类型：`std::unique_ptr<Tensor>`

### `GetInputNames()`

获取所有输入Tensor的名称。

参数：

- `None` 

返回：所有输入Tensor的名称

返回类型：`std::vector<std::string>`

### `GetOutputNames()`

获取所有输出Tensor的名称。

参数：

- `None`

返回：所有输出Tensor的名称

返回类型：`std::vector<std::string>`

### `GetInputByName(name)`

根据名称获取输出Tensor的指针，用来获取模型的输出结果。

参数：

- `name(const std::string)` - 输入Tensor的名称

返回：输入Tensor`的指针

返回类型：`std::unique_ptr<Tensor>`

### `GetTensor(name)`

根据名称获取输出Tensor的指针。

**注意**：`GetTensor`接口是为开发者设计的调试接口，可以输出[转化](../user_guides/model_optimize_tool)后模型中的任一节点。如果出现`GetTensor(InputName)`返回值为空`Tensor`，可能原因是以该`InputName`命名的Tensor在模型转化的**子图融合**过程被融合替换了。

参数：

- `name(const std::string)` - Tensor的名称

返回：指向`const Tensor`的指针

返回类型：`std::unique_ptr<const Tensor>`

### `Run()`

执行模型预测，需要在***设置输入数据后***调用。

参数：

- `None`

返回：`None`

返回类型：`void`



### `GetVersion()`

用于获取当前lib使用的代码版本。若代码有相应tag则返回tag信息，如`v2.0-beta`；否则返回代码的`branch(commitid)`，如`develop(7e44619)`。

参数：

- `None`

返回：当前lib使用的代码版本信息

返回类型：`std::string`

## TargetType

```c++
class TargetType;
```
`TargetType`为目标设备硬件类型，用户可以根据应用场景选择硬件平台类型。

枚举型变量`TargetType`的所有可能取值包括：

`{X86, CUDA, ARM, OpenCL, FPGA, NPU}`


## PrecisionType
```c++
class PrecisionType {FP32};
```
`PrecisionType`为模型中Tensor的数据精度，默认值为FP32(float32)。

枚举型变量`PrecisionType`的所有可能取值包括：

`{FP32, INT8, INT32, INT64}`




## DataLayoutType

```c++
class DataLayoutType {NCHW};
```
`DataLayoutType`为Tensor的数据格式，默认值为NCHW（number, channel, height, weigth）。

枚举型变量`DataLayoutType`的所有可能取值包括：

` {NCHW, NHWC}`



## Place
```c++
class Place{
  TargetType target;
  PrecisionType precision{FP32};
  DataLayoutType layout{NCHW}
}
```
`Place`是`TargetType`、`PrecisionType`和`DataLayoutType`的集合，说明运行时的设备类型、数据精度和数据格式。

示例：
```C++
Place{TargetType(ARM), PrecisionType(FP32), DataLayoutType(NCHW)}
```

## PowerMode

```c++
enum PowerMode;
```

`PowerMode`为ARM CPU能耗模式，用户可以根据应用场景设置能耗模式获得最优的能效比。

示例：

```c++
MobileConfig config;
// 设置NaiveBuffer格式模型目录
config.set_model_dir(FLAGS_model_dir);
// 设置能耗模式
config.set_power_mode(LITE_POWER_HIGH);

// 根据MobileConfig创建PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
```

PowerMode详细说明如下：

|         选项         | 说明                                                         |
| :------------------: | ------------------------------------------------------------ |
|   LITE_POWER_HIGH    | 绑定大核运行模式。如果ARM CPU支持big.LITTLE，则优先使用并绑定Big cluster。如果设置的线程数大于大核数量，则会将线程数自动缩放到大核数量。如果系统不存在大核或者在一些手机的低电量情况下会出现绑核失败，如果失败则进入不绑核模式。 |
|    LITE_POWER_LOW    | 绑定小核运行模式。如果ARM CPU支持big.LITTLE，则优先使用并绑定Little cluster。如果设置的线程数大于小核数量，则会将线程数自动缩放到小核数量。如果找不到小核，则自动进入不绑核模式。 |
|   LITE_POWER_FULL    | 大小核混用模式。线程数可以大于大核数量。当线程数大于核心数量时，则会自动将线程数缩放到核心数量。 |
|  LITE_POWER_NO_BIND  | 不绑核运行模式（推荐）。系统根据负载自动调度任务到空闲的CPU核心上。 |
| LITE_POWER_RAND_HIGH | 轮流绑定大核模式。如果Big cluster有多个核心，则每预测10次后切换绑定到下一个核心。 |
| LITE_POWER_RAND_LOW  | 轮流绑定小核模式。如果Little cluster有多个核心，则每预测10次后切换绑定到下一个核心。 |



## Tensor

```c++
class Tensor
```

Tensor是Paddle-Lite的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置Shape、数据、LoD信息等。

*注意：用户应使用`PaddlePredictor`的`GetInput`和`GetOuput`接口获取输入/输出的`Tensor`。*

示例：

```c++
int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

// 设置MobileConfig
MobileConfig config;
config.set_model_dir(FLAGS_model_dir);

// 根据MobileConfig创建PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

// 准备输入数据, 获取输入Tensor
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
// 设置输入Tensor维度信息
input_tensor->Resize({1, 3, 224, 224});
// 设置输入数据
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}

// 执行预测
predictor->Run();

// 获取输出Tensor
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
// 获取输出Tensor维度
printf("Output dim: %d\n", output_tensor->shape()[1]);
// 获取输出Tensor数据
for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
  printf("Output[%d]: %f\n", i, output_tensor->data<float>()[i]);
}
```

### `Resize(shape)`

设置Tensor的维度信息。

参数：

- `shape(std::vector<int64_t>)` - 维度信息

返回：`None`

返回类型：`void`



### `shape()`

获取Tensor的维度信息。

参数：

- `None`

返回：Tensor的维度信息

返回类型：`std::vector<int64_t>`



### `data<T>()`

```c++
template <typename T>
const T* data() const;
```

获取Tensor的底层数据的常量指针，根据传入的不同模型类型获取相应数据。用于读取Tensor数据。

示例：

```c++
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
// 如果模型中输出为float类型
output_tensor->data<float>()
```

参数：

- `None`

返回：`Tensor`底层数据常量指针

返回类型：`const T*`



### `mutable_data<T>()`

```c++
template <typename T>
T* mutable_data() const;
```

获取Tensor的底层数据的指针，根据传入的不同模型类型获取相应数据。用于设置Tensor数据。

示例：

```c++
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
// 如果模型中输出为float类型
auto* data = input_tensor->mutable_data<float>();
// 设置Tensor数据
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}
```

参数：

- `None`

返回：`Tensor`底层数据指针

返回类型：`T*`



### `SetLoD(lod)`

设置Tensor的LoD信息。

参数：

- `lod(std::vector<std::vector<uint64_t>>)` - Tensor的LoD信息

返回：`None`

返回类型：`void`



### `lod()`

获取Tensor的LoD信息

参数：

- `None`

返回：`Tensor`的LoD信息

返回类型：`std::vector<std::vector<uint64_t>>`
