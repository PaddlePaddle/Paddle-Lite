# Python Demo

## 1. 下载最新版本python预测库

```shell
python -m pip install paddlelite
```

## 2. 转化模型

PaddlePaddle的原生模型需要经过[opt]()工具转化为Paddle-Lite可以支持的naive_buffer格式。

以`mobilenet_v1`模型为例：

（1）下载[mobilenet_v1模型](http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz)后解压：

```shell
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxf mobilenet_v1.tar.gz
```

（2）使用opt工具：

 从磁盘加载模型时，根据模型和参数文件存储方式不同，加载模型和参数的路径有两种形式。

- Linux环境
  - 非combined形式：模型文件夹model_dir下存在一个模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为__model__。

  ```shell
  paddle_lite_opt --model_dir=./mobilenet_v1  \
                  --optimize_out=mobilenet_v1_opt \
                  --optimize_out_type=naive_buffer \
                  --valid_targets=x86
  ```
  - combined形式：模型文件夹model_dir下只有一个模型文件__model__和一个参数文件__params__时，传入模型文件和参数文件路径

  ```shell
  paddle_lite_opt --model_file=./mobilenet_v1/__model__ \
                  --param_file=./mobilenet_v1/__params__  \
                  --optimize_out=mobilenet_v1_opt \
                  --optimize_out_type=naive_buffer \
                  --valid_targets=x86
  ```

- windows环境

windows 暂不支持命令行方式直接运行模型转换器，需要编写python脚本

```python
import paddlelite.lite as lite

a=lite.Opt()
# 非combined形式
a.set_model_dir("D:\\YOU_MODEL_PATH\\mobilenet_v1")

# conmbined形式
# a.set_model_file("D:\\YOU_MODEL_PATH\\mobilenet_v1\\__model__")
# a.set_param_file("D:\\YOU_MODEL_PATH\\mobilenet_v1\\__params__")

a.set_optimize_out("mobilenet_v1_opt")
a.set_valid_places("x86")

a.run()
```

- MAC 环境

Opt工具使用方式同Linux（MAC环境暂不支持python端预测，下个版本会修复该问题）

## 3. 编写预测程序

准备好预测库和模型，我们便可以编写程序来执行预测。我们提供涵盖图像分类、目标检测等多种应用场景的C++示例demo可供参考，创建文件mobilenetV1_light_api.py，
python demo 完整代码位于 [demo/python](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/python/mobilenetv1_light_api.py) 。

(1) 设置config信息
```python
from paddlelite.lite import *

config = MobileConfig()
config.set_model_from_file(/YOU_MODEL_PATH/mobilenet_v1_opt.nb)
```

(2) 创建predictor

```python
predictor = create_paddle_predictor(config)
```

(3) 设置输入数据
```python
input_tensor = predictor.get_input(0)
input_tensor.resize([1, 3, 224, 224])
input_tensor.set_float_data([1.] * 3 * 224 * 224)
```

(4) 执行预测
```python
predictor.run()
```

(5) 得到输出数据
```python
output_tensor = predictor.get_output(0)
print(output_tensor.shape())
print(output_tensor.float_data()[:10])
```

## 4. 运行文件
```shell
python mobilenetV1_light_api.py
```
