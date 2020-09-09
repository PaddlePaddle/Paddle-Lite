# Python 完整示例

Python仅支持服务器端预测，目前支持 Windows / Mac / Linux (x86 | ARM)。

本章节包含2部分内容：(1) [Python 示例程序](python_demo.html#id1)；(2) [Python 应用开发说明](python_demo.html#id6)。

## Python 示例程序

本章节展示的所有Python 示例代码位于 [demo/python](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/python) 。

### 1. 环境准备

要编译和运行Android Python 示例程序，你需要准备一台可以编译运行PaddleLite的电脑。

### 2. 安装python预测库

```shell
python -m pip install paddlelite
```

**注意：** PyPI源目前仅提供Windows / Mac / Linux (x86) 三个平台pip安装包，如果您需要使用AMRLinux平台的Python预测功能，请参考[源码编译(ARMLinux)](../source_compile/compile_linux)。

### 3. 准备预测部署模型

(1) 模型下载：下载[mobilenet_v1](http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz)模型后解压，得到Paddle非combined形式的模型，位于文件夹 `mobilenet_v1` 下。可通过模型可视化工具[Netron](https://lutzroeder.github.io/netron/)打开文件夹下的`__model__`文件，查看模型结构。


```shell
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxf mobilenet_v1.tar.gz
```

(2) 模型转换：Paddle的原生模型需要经过[opt](../user_guides/model_optimize_tool)工具转化为Paddle-Lite可以支持的naive_buffer格式。

- Linux环境：通过pip安装paddlelite，即可获得paddle_lite_opt命令工具

  ```shell
  paddle_lite_opt --model_dir=./mobilenet_v1  \
                  --optimize_out=mobilenet_v1_opt \
                  --optimize_out_type=naive_buffer \
                  --valid_targets=x86
  ```

- windows环境：windows 暂不支持命令行方式直接运行模型转换器，需要编写python脚本

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

- MAC环境: paddle_lite_opt工具使用方式同Linux。

以上命令执行成功之后将在同级目录生成名为`mobilenet_v1_opt.nb`的优化后模型文件。

### 4. 下载和运行预测示例程序

从[demo/python](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/python)下载预测示例文件`mobilenetv1_light_api.py`和`mobilenetv1_full_api.py`，并运行Python预测程序。

```shell
# light api的输入为优化后模型文件mobilenet_v1_opt.nb
python mobilenetv1_light_api.py --model_dir=mobilenet_v1_opt.nb

# full api的输入为优化千的模型文件夹mobilenet_v1
python mobilenetv1_full_api.py --model_dir=./mobilenet_v1

# 运行成功后，将在控制台输出如下内容
[1L, 1000L]
[0.00019130950386170298, 0.0005920541007071733, 0.00011230241216253489, 6.27333574811928e-05, 0.0001275067188544199, 0.0013214796781539917, 3.138116153422743e-05, 6.52207963867113e-05, 4.780858944286592e-05, 0.0002588215284049511]
```
## Python 应用开发说明

Python代码调用Paddle-Lite执行预测库仅需以下六步：

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

(3) 从图片读入数据

```python
image = Image.open('./example.jpg')
resized_image = image.resize((224, 224), Image.BILINEAR)
image_data = np.array(resized_image).flatten().tolist()
```

(4) 设置输入数据

```python
input_tensor = predictor.get_input(0)
input_tensor.resize([1, 3, 224, 224])
input_tensor.set_float_data(image_data)
```

(5) 执行预测
```python
predictor.run()
```

(6) 得到输出数据
```python
output_tensor = predictor.get_output(0)
print(output_tensor.shape())
print(output_tensor.float_data()[:10])
```

详细的Python API说明文档位于[Python API](../api_reference/python_api_doc)。更多Python应用预测开发可以参考位于位于[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)的工程示例代码。
