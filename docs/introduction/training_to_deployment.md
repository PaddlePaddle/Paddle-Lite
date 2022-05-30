# 训练推理示例说明

本文档将向您介绍如何使用 Paddle 新接口训练和推理一个模型, 保存训练的模型后，使用 Paddle Lite 的 c++ 接口，在 andriod arm 上部署这个模型。
在这之前，可以参考[ Paddle 安装指南 ](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)完成 Paddle 的安装。

## 一、使用 Paddle 新接口训练一个简单模型

我们参考[ LeNet 的 MNIST 数据集图像分类 ](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/cv_case/image_classification/image_classification.html#lenetmnist)，使用 Paddle 接口训练一个简单的模型并存储为预测部署格式。我们将着重介绍如何生成模型文件。

- 依赖包导入

```
import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor
```

- 查看 Paddle 版本

```
print(paddle.__version__)
```

要求 paddle 版本号 >= 2.0.0

- 数据集准备

```
train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())
```

- 构建 LeNet 网络

```
class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
```

- 模型训练

```
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
model = LeNet()
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
def train(model, optim):
    model.train()
    epochs = 2
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()
train(model, optim)
```


## 二、训练部署场景的模型&参数保存载入

- 存储为预测部署模型：实际部署时，您需要使用预测格式的模型，预测格式模型相对训练格式模型而言，在拓扑上进行了裁剪，去除了预测不需要的算子。您可以参考[ InputSpec ](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/input_spec_cn.html)来完成动转静功能。只需 InputSpec 标记模型的输入，调用 `paddle.jit.to_static` 和 `paddle.jit.save` 即可得到预测格式的模型。

```
net = to_static(model, input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
paddle.jit.save(net, 'inference_model/lenet')
```
或者直接写为
```
paddle.jit.save(model, 'inference_model/lenet', input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
```

### 参考代码

```
import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor


class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1,
                                      out_channels=6,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6,
                                      out_channels=16,
                                      kernel_size=5,
                                      stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16 * 5 * 5,
                                        out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # x = x.reshape((-1, 1, 28, 28))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


def train(model, optim):
    model.train()
    epochs = 2
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # calc loss
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
                    epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()


if __name__ == '__main__':
    # paddle version
    print(paddle.__version__)

    # prepare datasets
    train_dataset = MNIST(mode='train', transform=ToTensor())
    test_dataset = MNIST(mode='test', transform=ToTensor())

    # load dataset
    train_loader = paddle.io.DataLoader(train_dataset,
                                        batch_size=64,
                                        shuffle=True)

    # build network
    model = LeNet()

    # prepare optimizer
    optim = paddle.optimizer.Adam(learning_rate=0.001,
                                  parameters=model.parameters())

    # train network
    train(model, optim)

    # save inferencing format model
    net = to_static(model,
                    input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
    paddle.jit.save(net, 'inference_model/lenet')
```

Paddle 2.0 及之后的版本默认保存的权重格式为 `*.pdiparams` 后缀的文件, 在当前路径 inference_model/ 下生成了三个文件 `lenet.pdiparams`, `lenet.pdmodel`, `lenet.pdiparams.info`, 其中 `lenet.pdmodel` 为模型文件， `lenet.pdiparams` 为权重文件。Paddle 1.8 是以权重分离的方式保存模型，即权重
参数信息分开保存在多个参数文件中， 模型保存在文件 `__model__` 中。


## 三、使用 Paddle Python 接口预测部署

我们使用存储好的预测部署模型，借助 Python 接口执行预测部署。

### 加载预测模型并进行预测配置

首先，我们加载预测模型，并配置预测时的一些选项，根据配置创建预测引擎：

```python
config = Config( "inference_model/lenet/lenet.pdmodel", "inference_model/lenet/lenet.pdiparams" ) # 通过模型和参数文件路径加载
config.disable_gpu() # 使用 cpu 预测
predictor = create_predictor(config) # 根据预测配置创建预测引擎 predictor
```

注意：如果是以权重分离的方式保存模型的模型，config 按如下方式设置：

```python
config = Config("inference_model/") # 通过路径加载，路径下保存着模型文件和多个权重信息文件
config.disable_gpu() # 使用 cpu 预测
predictor = create_predictor(config) # 根据预测配置创建预测引擎 predictor
```

更多配置选项可以参考[官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/python_infer_cn.html#config)。

### 设置输入

我们先通过获取输入 Tensor 的名称，再根据名称获取到输入 Tensor 的句柄。

```python
# 获取输入变量名称
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])
```

下面我们准备输入数据，并将其拷贝至待预测的设备上。这里我们使用了随机数据，您在实际使用中可以将其换为需要预测的真实图片。

```python
### 设置输入
fake_input = np.random.randn(1, 1, 28, 28).astype("float32")
input_handle.reshape([1, 1, 28, 28])
input_handle.copy_from_cpu(fake_input)
```

### 运行预测

```python
predictor.run()
```

### 获取输出

```python
# 获取输出变量名称
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu()
```
获取输出句柄的方式与输入类似，我们最后获取到的输出是 numpy.ndarray 类型，方便使用 numpy 对其进行后续的处理。

### 完整可运行代码
```python
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor

config = Config("inference_model/lenet.pdmodel", "inference_model/lenet.pdiparams")
config.disable_gpu()

# 创建 PaddlePredictor
predictor = create_predictor(config)

# 获取输入的名称
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

# 设置输入
fake_input = np.random.randn(1, 1, 28, 28).astype("float32")
input_handle.reshape([1, 1, 28, 28])
input_handle.copy_from_cpu(fake_input)

# 运行 predictor
predictor.run()

# 获取输出
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu() # numpy.ndarray 类型

print(output_data)
```

## 四、使用 Paddle Lite 预测库和 C++ 接口预测部署

存储好的模型可以使用 Paddle-Lite C++ 接口执行预测部署，具体可以参考文档 [ c++ 完整示例](../user_guides/cpp_demo.md)

## 附录
PaddlePaddle 提供了丰富的计算单元，使得用户可以采用模块化的方法解决各种学习问题，一些常见的模型可在此仓库获 [ PaddlePaddle/models ](https://github.com/PaddlePaddle/models)
