# Java API

## MobileConfig

```java
public class MobileConfig extends ConfigBase;
```

`MobileConfig`用来配置构建轻量级PaddlePredictor的配置信息，如NaiveBuffer格式的模型地址、能耗模式、工作线程数等等。

*注意：输入的模型需要使用Model Optimize Tool转化为NaiveBuffer格式的优化模型。*

示例：

```java
MobileConfig config = new MobileConfig();
// 设置NaiveBuffer格式模型目录
config.setModelFromFile(modelfile);
// 设置能耗模式
config.setPowerMode(PowerMode.LITE_POWER_HIGH);
// 设置工作线程数
config.setThreads(1);

// 根据MobileConfig创建PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);
```

### ``setModelFromFile(model_file)``

设置模型文件夹路径。

参数：

- `model_file(String)` - 模型文件路径

返回：`None`

返回类型：`void`



### ``setModelDir(model_dir)``

**注意**：Lite模型格式在release/v2.3.0之后修改，本接口为加载老格式模型的接口，将在release/v3.0.0废弃。建议替换为`setModelFromFile`接口。

设置模型文件夹路径。

参数：

- `model_dir(String)` - 模型文件夹路径

返回：`None`

返回类型：`void`



### ``setModelFromBuffer(model_buffer)``

设置模型的内存数据，当需要从内存加载模型时使用。

参数：

- `model_buffer(str)` - 内存中的模型数据

返回：`None`

返回类型：`void`



### `getModelDir()`

返回设置的模型文件夹路径。

参数：

- `None`

返回：模型文件夹路径

返回类型：`String`



### `setPowerMode(mode)`

设置CPU能耗模式。若不设置，则默认使用`LITE_POWER_HIGH`。

*注意：只在开启`OpenMP`时生效，否则系统自动调度。*

参数：

- `mode(PowerMode)` - CPU能耗模式。

返回：`None`

返回类型：`void`



### `getPowerMode()`

获取设置的CPU能耗模式。

参数：

- `None`

返回：设置的CPU能耗模式

返回类型：`PowerMode`



### `setThreads(threads)`

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启`OpenMP`的模式下生效，否则只使用单线程。*

参数：

- `threads(int)` - 工作线程数。默认为1。

返回：`None`

返回类型：`void`



### `getThreads()`

获取设置的工作线程数。

参数：

- `None`

返回：工作线程数

返回类型：`int`

## PaddlePredictor

```java
public class PaddlePredictor;
```

`PaddlePredictor`是Paddle-Lite的预测器。用户可以根据PaddlePredictor提供的接口使用MobileConfig创建新的预测器、设置输入数据、执行模型预测、获取输出以及获得当前使用lib的版本信息等。

示例：

```java
// 设置MobileConfig
MobileConfig config = new MobileConfig();
config.setModelDir(modelPath);

// 创建PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);

// 设置输入数据
long[] dims = {100, 100};
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
    inputBuffer[i] = i;
}
Tensor input = predictor.getInput(0);
input.resize(dims);
input.setData(inputBuffer);

// 执行预测
predictor.run();

// 获取输出数据
Tensor output = predictor.getOutput(0);
float[] output = result.getFloatData();
for (int i = 0; i < 1000; ++i) {
    System.out.println(output[i]);
}
```



### `CreatePaddlePredictor(config)`

```java
public static PaddlePredictor createPaddlePredictor(ConfigBase config);
```

`CreatePaddlePredictor`用来根据`ConfigBase`动态创建预测器，目前Java API支持使用MobileConfig`。框架会根据您在config中指定的模型路径、能耗模型、工作线程数等自动创建一个预测器。

参数：

- `config(ConfigBase，目前应使用MobileConfig)` - 创建预测器的配置信息

返回：根据config创建完成的预测器

返回类型：`PaddlePredictor`



### `getInput(index)`

获取输入Tensor，用来设置模型的输入数据。

参数：

- `index(int)` - 输入Tensor的索引

返回：第`index`个输入`Tensor`

返回类型：`Tensor`



### `getOutput(index)`

获取输出Tensor，用来获取模型的输出结果。

参数：

- `index(int)` - 输出Tensor的索引

返回：第`index`个输出Tensor

返回类型：`Tensor`



### `run()`

执行模型预测，需要在***设置输入数据后***调用。

参数：

- `None`

返回：预测执行状态，成功返回`true`，否则返回`false`

返回类型：`boolean`



### `getVersion()`

用于获取当前lib使用的代码版本。若代码有相应tag则返回tag信息，如`v2.0-beta`；否则返回代码的`branch(commitid)`，如`develop(7e44619)`。

参数：

- `None`

返回：当前lib使用的代码版本信息

返回类型：`String`

## PowerMode

```java
public enum PowerMode;
```

`PowerMode`为ARM CPU能耗模式，用户可以根据应用场景设置能耗模式获得最优的能效比。

示例：

```java
MobileConfig config = new MobileConfig();
// 设置NaiveBuffer格式模型目录
config.setModelDir(modelPath);
// 设置能耗模式
config.setPowerMode(PowerMode.LITE_POWER_HIGH);

// 根据MobileConfig创建PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);
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
public class Tensor;
```

Tensor是Paddle-Lite的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置维度、数据等。

*注意：用户应使用`PaddlePredictor`的`getInput`和`getOuput`接口获取输入/输出的`Tensor`。*

示例：

```java
// 导入Java API
import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.Predictor;
import com.baidu.paddle.lite.PowerMode;

// 设置MobileConfig
MobileConfig config = new MobileConfig();
config.setModelDir(modelPath);

// 创建PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);

// 设置输入数据
long[] dims = {100, 100};
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
    inputBuffer[i] = i;
}
// 获取输入Tensor
Tensor input = predictor.getInput(0);
// 设置输入维度
input.resize(dims);
// 设置输入数据
input.setData(inputBuffer);

// 执行预测
predictor.run();

// 获取输出Tensor
Tensor result = predictor.getOutput(0);
// 获取输出数据
float[] output = result.getFloatData();
for (int i = 0; i < 1000; ++i) {
    System.out.println(output[i]);
}
```

### `resize(dims)`

设置Tensor的维度信息。

参数：

- `dims(long[])` - 维度信息

返回：设置成功返回`true`，否则返回`false`

返回类型：`boolean`



### `shape()`

获取Tensor的维度信息。

参数：

- `None`

返回：Tensor的维度信息

返回类型：`long[]`



### `setData(data)`

设置Tensor数据。

参数：

- `data(float[])` - 需要设置的数据

返回：成功则返回`true`，否则返回`false`

返回类型：`boolean`



### `getFloatData()`

获取Tensor的底层float型数据。

参数：

- `None`

返回：`Tensor`底层数据

返回类型：`float[]`
