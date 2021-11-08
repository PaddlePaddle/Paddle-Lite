# Java API

## MobileConfig

import [com.baidu.paddle.lite.MobileConfig](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.10/lite/api/android/jni/src/com/baidu/paddle/lite/MobileConfig.java);

```java
public class MobileConfig extends ConfigBase;
```

`MobileConfig` 用来配置构建轻量级 PaddlePredictor 的配置信息，如 NaiveBuffer 格式的模型地址、能耗模式、工作线程数等等。

*注意：输入的模型需要使用 Model Optimize Tool 转化为 NaiveBuffer 格式的优化模型。*

示例：

```java
MobileConfig config = new MobileConfig();
// 设置 NaiveBuffer 格式模型目录
config.setModelFromFile(modelfile);
// 设置能耗模式
config.setPowerMode(PowerMode.LITE_POWER_HIGH);
// 设置工作线程数
config.setThreads(1);

// 根据 MobileConfig 创建 PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);
```

### `setModelFromFile`

```java
void setModelFromFile(String x);
```

设置模型文件，当需要从磁盘加载模型时使用。

- 参数
    - `x`: 模型文件路径


### `setModelFromBuffer`

```java
void setModelFromBuffer(String x);
```

设置模型的内存数据，当需要从内存加载模型时使用。

- 参数

    - `x`： 内存中的模型数据



### `setPowerMode`

```java
void void setPowerMode(PowerMode mode);
```

设置 CPU 能耗模式。若不设置，则默认使用 `LITE_POWER_HIGH`。

*注意：只在开启 `OpenMP` 时生效，否则系统自动调度。*

- 参数

    - `PowerMode`： CPU 能耗模式。



### `getPowerMode`

```java
PowerMode getPowerMode();
```

获取设置的 CPU 能耗模式。


- 返回值：

  设置的 CPU 能耗



### `setThreads`

```java
void setThreads(int i);
```

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启 `OpenMP` 的模式下生效，否则只使用单线程。*

- 参数

    - `i`： 工作线程数，默认为1。



### `getThreads`

```java
int getThreads();
```

获取设置的工作线程数。


- 返回值

  工作线程数


## PaddlePredictor

import [com.baidu.paddle.lite.PaddlePredictor](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.10/lite/api/android/jni/src/com/baidu/paddle/lite/PaddlePredictor.java);

```java
public class PaddlePredictor;
```

`PaddlePredictor` 是 Paddle Lite 的预测器。用户可以根据 `PaddlePredictor` 提供的接口使用 `MobileConfig` 创建新的预测器、设置输入数据、执行模型预测、获取输出以及获得当前使用 lib 的版本信息等。

示例：

```java
// 设置 MobileConfig
MobileConfig config = new MobileConfig();
config.setModelDir(modelPath);

// 创建 PaddlePredictor
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



### `CreatePaddlePredictor`

```java
public static PaddlePredictor createPaddlePredictor(ConfigBase config);
```

`CreatePaddlePredictor` 用来根据 `MobileConfig` 动态创建预测器。框架会根据您在 config 中指定的模型路径、能耗模型、工作线程数等自动创建一个预测器。

- 参数

    - `config(MobileConfig)` : 用于构建 Predictor 的配置信息。

- 返回值

  `PaddlePredictor` 指针



### `getInput`

```java
Tensor getInput(int i);
```

获取输入 `Tensor`，用来设置模型的输入数据。

- 参数

    - `i`: 输入 `Tensor` 的索引

- 返回值

  第 `i` 个输入 `Tensor` 的指针



### `getOutput`

```java
Tensor getOutput(int i);
```

获取输出 `Tensor`，用来获取模型的输出结果。

参数

- 参数

    - `i`: 输出 `Tensor` 的索引

- 返回值

  第 `i` 个输出 `Tensor` 的指针



### `run`

```java
boolean run();
```

执行模型预测，需要在**设置输入数据后**调用。




### `getVersion`

```java
String getVersion();
```

用于获取当前 lib 使用的代码版本。若代码有相应 tag 则返回 tag 信息，如 `v2.0-beta`；否则返回代码的 `branch(commitid)`，如 `develop(7e44619)`。

- 返回值：

  当前库使用的代码版本信息



## PowerMode

import [com.baidu.paddle.lite.PowerMode](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.10/lite/api/android/jni/src/com/baidu/paddle/lite/PowerMode.java);

```java
public enum PowerMode;
```

`PowerMode` 为 ARM CPU 能耗模式，用户可以根据应用场景设置能耗模式获得最优的能效比。

示例：

```java
MobileConfig config = new MobileConfig();
// 设置 NaiveBuffer 格式模型目录
config.setModelDir(modelPath);
// 设置能耗模式
config.setPowerMode(PowerMode.LITE_POWER_HIGH);

// 根据 MobileConfig 创建 PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);
```

PowerMode详细说明如下：

|         选项         | 说明                                                         |
| :------------------: | ------------------------------------------------------------ |
|   LITE_POWER_HIGH    | 绑定大核运行模式。如果 ARM CPU 支持 big.LITTLE，则优先使用并绑定 Big cluster，如果设置的线程数大于大核数量，则会将线程数自动缩放到大核数量。如果系统不存在大核或者在一些手机的低电量情况下会出现绑核失败，如果失败则进入不绑核模式。 |
|    LITE_POWER_LOW    | 绑定小核运行模式。如果 ARM CPU 支持 big.LITTLE，则优先使用并绑定 Little cluster，如果设置的线程数大于小核数量，则会将线程数自动缩放到小核数量。如果找不到小核，则自动进入不绑核模式。 |
|   LITE_POWER_FULL    | 大小核混用模式。线程数可以大于大核数量，当线程数大于核心数量时，则会自动将线程数缩放到核心数量。 |
|  LITE_POWER_NO_BIND  | 不绑核运行模式（推荐）。系统根据负载自动调度任务到空闲的 CPU 核心上。 |
| LITE_POWER_RAND_HIGH | 轮流绑定大核模式。如果 Big cluster 有多个核心，则每预测10次后切换绑定到下一个核心。 |
| LITE_POWER_RAND_LOW  | 轮流绑定小核模式。如果 Little cluster 有多个核心，则每预测10次后切换绑定到下一个核心。 |


## Tensor

import [com.baidu.paddle.lite.Tensor](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.10/lite/api/android/jni/src/com/baidu/paddle/lite/Tensor.java);

```c++
public class Tensor;
```

Tensor 是 Paddle Lite 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置维度、数据等。

*注意：用户应使用 `PaddlePredictor` 的 `getInput` 和 `getOuput` 接口获取输入/输出的 `Tensor`。*

示例：

```java
// 导入 Java API
import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.Predictor;
import com.baidu.paddle.lite.PowerMode;

// 设置 MobileConfig
MobileConfig config = new MobileConfig();
config.setModelDir(modelPath);

// 创建 PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);

// 设置输入数据
long[] dims = {100, 100};
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
    inputBuffer[i] = i;
}
// 获取输入 Tensor
Tensor input = predictor.getInput(0);
// 设置输入维度
input.resize(dims);
// 设置输入数据
input.setData(inputBuffer);

// 执行预测
predictor.run();

// 获取输出 Tensor
Tensor result = predictor.getOutput(0);
// 获取输出数据
float[] output = result.getFloatData();
for (int i = 0; i < 1000; ++i) {
    System.out.println(output[i]);
}
```

### `resize`

```java
boolean resize(long[] shape);
```

设置 Tensor 的维度信息。

- 参数

    - `shape`： 维度信息



### `shape`

```java
long[] shape();
```

获取 Tensor 的维度信息。


- 返回值

  Tensor 的维度信息




### `setData`

```java
boolean setData(float[] data);
```

设置 Tensor 数据。

- 参数

    - `data`： 需要设置的数据




### `getFloatData`

```java
float[] getFloatData();
```

获取 Tensor 的底层 float 型数据。


- 返回值：

  `Tensor` 底层 float 型数据
