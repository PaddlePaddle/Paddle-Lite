# Python API


### [create_paddle_predictor](./python_api/create_paddle_predictor)

创建预测执行器[`CxxPredictor`](./python_api/CxxPredictor)或者[`LightPredictor`](./python_api/LightPredictor)

### [Opt](./python_api/opt)

```python
class Opt;
```

`Opt`模型离线优化接口，Paddle原生模型需经`opt`优化图结构后才能在Paddle-Lite上运行。

### [CxxConfig](./python_api/CxxConfig)
```python
class CxxConfig;
```

`CxxConfig`用来配置构建CxxPredictor的配置信息，如protobuf格式的模型地址、能耗模式、工作线程数、place信息等等。


### [MobileConfig](./python_api/MobileConfig)

```python
class MobileConfig;
```

`MobileConfig`用来配置构建LightPredictor的配置信息，如NaiveBuffer格式的模型地址、能耗模式、工作线程数等等。


### [CxxPredictor](./python_api/CxxPredictor)

```python
class CxxPredictor
```

`CxxPredictor`是Paddle-Lite的预测器，由`create_paddle_predictor`根据`CxxConfig`进行创建。用户可以根据CxxPredictor提供的接口设置输入数据、执行模型预测、获取输出以及获得当前使用lib的版本信息等。



### [TargetType 、PrecisionType、DataLayoutType、Place](./python_api/TypePlace)

`TargetType`为目标设备硬件类型，用户可以根据应用场景选择硬件平台类型。

`PrecisionType`为模型中Tensor的数据精度，默认值为FP32(float32)。

`DataLayoutType`为Tensor的数据格式，默认值为NCHW（number, channel, height, weigth）。

`Place`是`TargetType`、`PrecisionType`和`DataLayoutType`的集合，说明运行时的设备类型、数据精度和数据格式。




### [PowerMode](./python_api/PowerMode)

```python
class PowerMode;
```

`PowerMode`为ARM CPU能耗模式，用户可以根据应用场景设置能耗模式获得最优的能效比。



### [Tensor](./python_api/Tensor)

```c++
class Tensor
```

Tensor是Paddle-Lite的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置Shape、数据、LoD信息等。

*注意：用户应使用`CxxPredictor`或`LightPredictor`的`get_input`和`get_output`接口获取输入/输出的`Tensor`。*
