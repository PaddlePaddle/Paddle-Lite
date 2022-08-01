
# C++ API

## CreatePaddlePredictor

 \#include &lt;[paddle\_api.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_api.h)&gt;

```c++
template <typename ConfigT>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT&);
```

`CreatePaddlePredictor` 用来根据 `MobileConfig` 构建预测器。

示例：

```c++
// 设置 MobileConfig
MobileConfig config;
config.set_model_dir(FLAGS_model_dir);

// 根据 MobileConfig 创建 PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
```

- 参数

    - `config`: 用于构建 Predictor 的配置信息。

- 返回值

  `PaddlePredictor` 指针

## CxxConfig

 \#include &lt;[paddle\_api.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_api.h)&gt;

```c++
class CxxConfig;
```

`CxxConfig` 用来配置构建 CxxPredictor 的配置信息，如 protobuf 格式的模型地址、能耗模式、工作线程数、place 信息等等。

示例：

```c++
config = CxxConfig()
# 设置模型目录，加载非 combined 模型时使用
config.set_model_dir(<your_model_dir_path>)
# 设置工作线程数
config.set_threads(4);
# 设置能耗模式
config.set_power_mode(PowerMode.LITE_POWER_NO_BIND)
# 设置 valid places
places = [Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据 CxxConfig 创建 CxxPredictor
predictor = create_paddle_predictor(config)
```

### 公有成员函数

### `set_model_dir`

```c++
void set_model_dir(const std::string& x);
```

设置模型文件夹路径，当需要从磁盘加载非 combined 模型时使用。

- 参数

    - `x`：模型文件夹路径


### `model_dir`

```c++
const std::string& model_dir();
```

返回设置的模型文件夹路径。

- 返回值

  模型文件夹路径


### `set_model_file`

```c++
void set_model_file(const std::string& path);
```

设置模型文件路径，加载 combined 形式模型时使用。

- 参数

    - `path`：模型文件路径


### `model_file`

```c++
std::string model_file();
```

获取设置模型文件路径，加载 combined 形式模型时使用。

- 返回值

  模型文件路径


### `set_param_file`

```c++
void set_param_file(const std::string& path);
```

设置模型参数文件路径，加载 combined 形式模型时使用。

- 参数

    - `path`: 模型参数文件路径

### `param_file`

```c++
std::string param_file() const;
```

获取设置模型参数文件路径，加载 combined 形式模型时使用。

- 返回值 

  模型参数文件路径

### `set_valid_places`

```c++
void set_valid_places(const std::vector<Place>& x);
```

设置可用的 places 列表。

- 参数

    - `x`：可用 place 列表。

示例：

```c++
config = CxxConfig()
# 设置模型目录，加载非 combined 模型时使用
config.set_model_dir(<your_model_dir_path>)
# 设置 valid places
# 注意，valid_places 列表中 Place 的排序表明了用户对 Place 的偏好程度，如用户想优先使用 ARM 上 Int8 精度的
# kernel，则应把 Place(TargetType.ARM, PrecisionType.INT8) 置于 valid_places 列表的首位。
places = [Place(TargetType.ARM, PrecisionType.INT8),
          Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据 CxxConfig 创建 CxxPredictor
predictor = create_paddle_predictor(config)
```

### `set_power_mode`

```c++
void set_power_mode(PowerMode mode);
```

设置 CPU 能耗模式。若不设置，则默认使用 `PowerMode.LITE_POWER_HIGH`。

*注意：只在开启 `OpenMP` 时生效，否则系统自动调度。此函数只在使用 `LITE_WITH_ARM` 编译选项下生效。*

- 参数

    - `mode(PowerMode)`：CPU 能耗模式


### `power_mode`

```c++
PowerMode power_mode() const;
```

获取设置的 CPU 能耗模式。

*注意：此函数只在使用 `LITE_WITH_ARM` 编译选项下生效。*

- 返回值

  设置的 CPU 能耗模式

### `set_threads`

```c++
void set_threads(int threads);
```

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启 `OpenMP` 的模式下生效，否则只使用单线程。此函数只在使用 `LITE_WITH_ARM` 编译选项下生效。*

- 参数

    - `threads`：工作线程数


### `threads`

```c++
int threads() const;
```

获取设置的工作线程数。

*注意：此函数只在使用 `LITE_WITH_ARM` 编译选项下生效。*

- 返回值

  工作线程数


### `set_x86_math_num_threads`

```c++
void set_x86_math_num_threads(int threads);
```

设置 CPU Math 库线程数，CPU 核心数支持情况下可加速预测。默认为 1，并且仅在 x86 下有效。

- 参数

    - `threads`：CPU Math 库线程数


### `x86_math_num_threads`

```c++
int x86_math_num_threads() const;
```

返回 CPU Math 库线程数，CPU 核心数支持情况下可加速预测。仅在 x86 下有效。

- 返回值

  CPU Math 库线程数

## MobileConfig

 \#include &lt;[paddle\_api.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_api.h)&gt;

```c++
class MobileConfig;
```

`MobileConfig` 用来配置构建轻量级 PaddlePredictor 的配置信息，如 NaiveBuffer 格式的模型地址、模型的内存地址（从内存加载模型时使用）、能耗模式、工作线程数等等。

*注意：输入的模型需要使用 [Model Optimize Tool](https://paddle-lite.readthedocs.io/zh/develop/user_guides/model_optimize_tool.html) 转化为 NaiveBuffer 格式的优化模型。*

示例：

```c++
MobileConfig config;
// 判断设备是否支持 FP16 指令集(或者是否是 armv8.2 架构的 arm 设备)
bool suppor_fp16 = config.check_fp16_valid();
// 设置 NaiveBuffer 格式模型目录，从文件加载模型时使用
config.set_model_from_file(<your_model_path>);
// 设置工作线程数
config.set_threads(4);
// 设置能耗模式
config.set_power_mode(LITE_POWER_HIGH);

// 根据 MobileConfig 创建 PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
```

### `check_fp16_valid`

```c++
bool check_fp16_valid();
```

判断当前设备是否支持FP16 指令集(或者是否是 armv8.2 架构的 arm 设备)，且该方法仅对 **arm CPU** 有效

### `set_model_from_file`

```c++
void set_model_from_file(const std::string& x);
```

设置模型文件，当需要从磁盘加载模型时使用。

- 参数

    - `x`: 模型文件路径

### `set_model_dir`

```c++
void set_model_dir(const std::string& x);
```

**注意**：Lite 模型格式在 release/v2.3.0 之后修改，本接口为加载老格式模型的接口，将在 release/v3.0.0 废弃。建议替换为 `set_model_from_file` 接口。

设置模型文件夹路径，当需要从磁盘加载模型时使用。

- 参数

    - `x`: 模型文件夹路径

### `model_dir`

```c++
const std::string& model_dir() const;
```
获取设置的模型文件夹路径。

- 返回值

  模型文件夹路径


### `set_model_from_buffer`

```c++
void set_model_from_buffer(const std::string& x);
```

设置模型的内存数据，当需要从内存加载模型时使用。

- 参数

    - `x`: 内存中的模型数据

### `set_model_buffer`

```c++
void set_model_buffer(const char* model_buffer,
                      size_t model_buffer_size,
                      const char* param_buffer,
                      size_t param_buffer_size);
```

**注意**：Lite 模型格式在 release/v2.3.0 之后修改，本接口为加载老格式模型的接口，将在 release/v3.0.0 废弃。建议替换为 `set_model_from_buffer` 接口。

设置模型、参数的内存地址，当需要从内存加载模型时使用。

示例：

```c++
// 读取模型文件到内存
std::string model_buffer = ReadFile(FLAGS_model_path);
std::string params_buffer = lite::ReadFile(FLAGS_params_path);

// 设置 MobileConfig
lite_api::MobileConfig config;
config.set_model_buffer(model_buffer.c_str(), model_buffer.size(),
                        params_buffer.c_str(), params_buffer.size());

// 根据 MobileConfig 创建 PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
```

- 参数

    - `model_buffer`: 内存中模型结构数据。
    - `model_buffer`: 内存中模型结构数据的大小。
    - `param_buffer`: 内存中模型参数数据。
    - `param_buffer`: 内存中模型参数数据的大小。

### `is_model_from_memory`

```c++
bool is_model_from_memory() const;
```

是否从内存中加载模型，当使用 `set_model_buffer` 接口时返回 `true`。

- 返回值

  是否从内存加载模型

### `model_buffer`

```c++
const std::string& model_buffer() const;
```

获取内存中模型结构数据。

- 返回值

  内存中模型结构数据

### `param_buffer`

```c++
const std::string& param_buffer() const;
```

获取内存中模型参数数据。


- 返回值

  内存中模型参数数据


### `set_power_mode`

```c++
void set_power_mode(PowerMode mode);
```

设置 CPU 能耗模式。若不设置，则默认使用 `LITE_POWER_HIGH`。

*注意：只在开启 `OpenMP` 时生效，否则系统自动调度。*

- 参数

    - `PowerMode`: CPU 能耗模式

### `power_mode`

```c++
PowerMode power_mode() const;
```

获取设置的 CPU 能耗模式。

- 返回值

  设置的 CPU 能耗模式


### `set_threads`

```c++
void set_threads(int threads);
```

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启 `OpenMP` 的模式下生效，否则只使用单线程。*

- 参数

    - `threads`: 工作线程数

### `threads`

```c++
int threads() const;
```

获取设置的工作线程数。

- 返回值

  工作线程数


### `set_metal_lib_path`

```c++
void set_metal_lib_path(const std::string& path);
```

用于 iOS 设备上使用 Metal 进行 GPU 预测时，配置 metallib 加载路径。

- 参数

    - `str`：metallib 库文件路径


### `set_metal_use_mps`

```c++
void set_metal_use_mps(bool flag);
```

设置 iOS 设备上使用 Metal 进行 GPU 预测时，是否启用 [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)。若不设置，默认不使用（建议启用）。

- 参数

    - `flag`：是否使用 MPS

### `metal_use_mps`

```c++
bool metal_use_mps() const; 
```

- 返回值

  是否使用 Metal Performance Shaders


## PaddlePredictor

 \#include &lt;[paddle\_api.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_api.h)&gt;

```c++
class PaddlePredictor;
```

`PaddlePredictor` 是 Paddle Lite 的预测器，由 `CreatePaddlePredictor` 根据 `MobileConfig` 进行创建。用户可以根据 PaddlePredictor 提供的接口设置输入数据、执行模型预测、获取输出以及获得当前使用 lib 的版本信息等。

示例：

```c++
int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

// 设置 MobileConfig
MobileConfig config;
config.set_model_dir(FLAGS_model_dir);

// 根据 MobileConfig 创建 PaddlePredictor
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
// (1) 根据 index 获取输入 Tensor
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
// (2) 根据名称获取输入 Tensor
// std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInputByName(input_names[0])));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}

// 执行预测
predictor->Run();

// 获取输出
// (1) 根据 index 获取输出 Tensor
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
// (2) 根据名称获取输出 Tensor
// std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(output_names[0])));
printf("Output dim: %d\n", output_tensor->shape()[1]);
for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
  printf("Output[%d]: %f\n", i, output_tensor->data<float>()[i]);
}
```

### `GetInput`

```c++
virtual std::unique_ptr<Tensor> GetInput(int i) = 0;
```

获取输入 Tensor 指针，用来设置模型的输入数据。

- 参数

    - `i`: 输入 `Tensor` 的索引

- 返回值

  第 `i` 个输入 `Tensor` 的指针

### `GetOutput`

```c++
virtual std::unique_ptr<const Tensor> GetOutput(int i) const = 0;
```

获取输出 Tensor 的指针，用来获取模型的输出结果。

- 参数

    - `i`: 输出 Tensor 的索引

- 返回值

  第 `i` 个输出 `Tensor` 的指针


### `GetInputNames`

```c++
virtual std::vector<std::string> GetInputNames() = 0;
```

获取所有输入 Tensor 的名称。

- 返回值

  所有输入 Tensor 的名称


### `GetOutputNames`

```c++
virtual std::vector<std::string> GetOutputNames() = 0;
```

获取所有输出 Tensor 的名称。

- 返回值

  所有输出 Tensor 的名称


### `GetInputByName`

```c++
virtual std::unique_ptr<Tensor> GetInputByName(const std::string& name) = 0;
```

根据名称获取输出 Tensor 的指针，用来获取模型的输出结果。

- 参数

    - `name`: 输入 Tensor 的名称

- 返回值

  输入`Tensor` 的指针


### `GetTensor`

```c++
virtual std::unique_ptr<const Tensor> GetTensor(const std::string& name) const = 0;
```

根据名称获取输出Tensor的指针。

**注意**：`GetTensor` 接口是为开发者设计的调试接口，可以输出[转化](https://paddle-lite.readthedocs.io/zh/develop/user_guides/model_optimize_tool.html)后模型中的任一节点。如果出现 `GetTensor(InputName)` 返回值为空 `Tensor`，可能原因是以该 `InputName` 命名的 Tensor 在模型转化的**子图融合**过程被融合替换了。


- 参数

    - `name`: Tensor 的名称

- 返回值

  指向 `const Tensor` 的指针

### `Run`

```c++
virtual void Run() = 0;
```

执行模型预测，需要在设置输入数据后调用。


### `GetVersion`

```c++
virtual std::string GetVersion() const = 0;
```

用于获取当前库使用的代码版本。若代码有相应 tag 则返回 tag 信息，如 `v2.0-beta`；否则返回代码的 `branch(commitid)`，如 `develop(7e44619)`。

- 返回值

  当前库使用的代码版本信息

## TargetType

 \#include &lt;[paddle\_place.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_place.h)&gt;

```c++
class TargetType;
```
`TargetType` 为目标设备硬件类型，用户可以根据应用场景选择硬件平台类型。

枚举型变量 `TargetType` 的所有可能取值包括：`{kX86, kCUDA, kARM, kOpenCL, kFPGA, kNPU}`

## PrecisionType

 \#include &lt;[paddle\_place.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_place.h)&gt;

```c++
class PrecisionType;
```

`PrecisionType` 为模型中 Tensor 的数据精度，默认值为 FP32 （float32）。

枚举型变量 `PrecisionType` 的所有可能取值包括: `{kFloat, kInt8, kInt32, kFP16, kBool, kInt64, kInt16, kUInt8, kFP64, kAny}`


## DataLayoutType

 \#include &lt;[paddle\_place.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_place.h)&gt;

```c++
class DataLayoutType;
```

`DataLayoutType` 为 Tensor 的数据格式，默认值为 NCHW（number, channel, height, weigth）。

枚举型变量 `DataLayoutType` 的所有可能取值包括：`{kNCHW, kNHWC, kImageDefault, kImageFolder, kImageNW, kMetalTexture2DArray, kMetalTexture2D, kAny}`


## Place

 \#include &lt;[paddle\_place.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_place.h)&gt;

```C++
struct Place;
```

`Place` 是 `TargetType`、`PrecisionType` 和 `DataLayoutType` 的集合，说明运行时的设备类型、数据精度和数据格式。

示例：
```C++
Place{TargetType(ARM), PrecisionType(FP32), DataLayoutType(NCHW)}
```

## PowerMode

 \#include &lt;[paddle\_place.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_place.h)&gt;

```c++
enum PowerMode;
```

`PowerMode` 为 ARM CPU 能耗模式，用户可以根据应用场景设置能耗模式获得最优的能效比。

示例：

```c++
MobileConfig config;
// 设置 NaiveBuffer 格式模型目录
config.set_model_dir(FLAGS_model_dir);
// 设置能耗模式
config.set_power_mode(LITE_POWER_HIGH);

// 根据 MobileConfig 创建 PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
```

PowerMode详细说明如下：

|         选项         | 说明                                                         |
| :------------------: | ------------------------------------------------------------ |
|   LITE_POWER_HIGH    | 绑定大核运行模式。如果 ARM CPU 支持 big.LITTLE，则优先使用并绑定 Big cluster，如果设置的线程数大于大核数量，则会将线程数自动缩放到大核数量。如果系统不存在大核或者在一些手机的低电量情况下会出现绑核失败，如果失败则进入不绑核模式。 |
|    LITE_POWER_LOW    | 绑定小核运行模式。如果 ARM CPU 支持 big.LITTLE，则优先使用并绑定 Little cluster，如果设置的线程数大于小核数量，则会将线程数自动缩放到小核数量。如果找不到小核，则自动进入不绑核模式。 |
|   LITE_POWER_FULL    | 大小核混用模式。线程数可以大于大核数量，当线程数大于核心数量时，则会自动将线程数缩放到核心数量。 |
|  LITE_POWER_NO_BIND  | 不绑核运行模式（推荐）。系统根据负载自动调度任务到空闲的 CPU 核心上。 |
| LITE_POWER_RAND_HIGH | 轮流绑定大核模式。如果 Big cluster 有多个核心，则每预测 10 次后切换绑定到下一个核心。 |
| LITE_POWER_RAND_LOW  | 轮流绑定小核模式。如果 Little cluster 有多个核心，则每预测 10 次后切换绑定到下一个核心。 |


## Tensor
 \#include &lt;[paddle\_api.h](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/api/paddle_api.h)&gt;

```c++
struct Tensor
```

Tensor 是 Paddle Lite 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。

*注意：用户应使用 `PaddlePredictor` 的 `GetInput` 和 `GetOuput` 接口获取输入 / 输出的 `Tensor`。*

示例：

```c++
int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

// 设置 MobileConfig
MobileConfig config;
config.set_model_dir(FLAGS_model_dir);

// 根据 MobileConfig 创建 PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

// 准备输入数据, 获取输入 Tensor
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
// 设置输入 Tensor 维度信息
input_tensor->Resize({1, 3, 224, 224});
// 设置输入数据
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}

// 执行预测
predictor->Run();

// 获取输出 Tensor
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
// 获取输出 Tensor 维度
printf("Output dim: %d\n", output_tensor->shape()[1]);
// 获取输出 Tensor 数据
for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
  printf("Output[%d]: %f\n", i, output_tensor->data<float>()[i]);
}
```

### `Resize`

```c++
void Resize(const shape_t& shape);
```

设置 Tensor 的维度信息。

- 参数
    - `shape`: 维度信息

### `shape`

```c++
shape_t shape() const;
```

获取 Tensor 的维度信息。

- 返回值

  Tensor 的维度信息

### `data`

```c++
template <typename T> const T* data() const;
```

获取 Tensor 的底层数据的常量指针，根据传入的不同模型类型获取相应数据。用于读取 Tensor 数据。

示例：

```c++
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
// 如果模型中输出为 float 类型
output_tensor->data<float>()
```

- 返回值

  `Tensor` 底层数据常量指针

### `mutable_data`

```c++
template <typename T> T* mutable_data() const;
```

获取 Tensor 的底层数据的指针，根据传入的不同模型类型获取相应数据。用于设置 Tensor 数据。

示例：

```c++
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
// 如果模型中输出为 float 类型
auto* data = input_tensor->mutable_data<float>();
// 设置 Tensor 数据
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}
```

- 返回值

  `Tensor` 底层数据指针


### `ShareExternalMemory`

```c++
void ShareExternalMemory(void* data, size_t memory_size, TargetType target);
```

设置 Tensor 共享用户数据指针。注意：请保证数据指针在预测过程中处于有效状态。

示例：

```c++
lite_api::CxxConfig config
config.set_model_dir(FLAGS_model_dir);
config.set_valid_places({
  Place{TARGET(kX86), PRECISION(kFloat)},
  Place{TARGET(kARM), PRECISION(kFloat)},
});
auto predictor = lite_api::CreatePaddlePredictor(config);
auto inputs = predictor->GetInputNames();
auto outputs = predictor->GetOutputNames();

std::vector<float> external_data(100 * 100, 0);
auto input_tensor = predictor->GetInputByName(inputs[0]);
input_tensor->Resize(std::vector<int64_t>({100, 100}));
size_t memory_size = external_data.size() * sizeof(float);

input_tensor->ShareExternalMemory(static_cast<void*>(external_data.data()),
                                  memory_size,
                                  config.valid_places()[0].target);
predictor->Run();
```

- 参数

    - `data`: 外部数据指针，请确保在预测过程中数据处于有效状态
    - `memory_size`: 外部数据所占字节大小
    - `target`: 目标设备硬件类型，即数据所处设备类型

### `SetLoD`

```c++
void SetLoD(const lod_t& lod);
```

设置 Tensor 的 LoD 信息。

- 参数

    - `lod`: Tensor 的 LoD 信息, 类型为 `std::vector<std::vector<uint64_t>>`

### `lod`

```c++
lod_t lod() const;
```

获取Tensor的 LoD 信息

- 返回值
  `Tensor`的 LoD 信息, 类型为 `std::vector<std::vector<uint64_t>>`


### `precision`

```c++
PrecisionType precision() const;
```

获取Tensor的精度信息

- 返回值

  `Tensor` 的 precision 信息, 类型为 `PrecisionType`

### `SetPrecision`

```c++
void SetPrecision(PrecisionType precision);
```

设置Tensor的精度信息

- 参数

    - `precision`: Tensor 的 precision 信息


### `target`

```c++
TargetType target() const;
```

获取 Tensor 的数据所处设备信息

- 返回值

  `Tensor` 的 target 信息
