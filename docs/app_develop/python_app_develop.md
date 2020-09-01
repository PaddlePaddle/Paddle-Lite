# Python 应用开发

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

详细的Python API说明文档位于[Python API](../api_reference/python_api_doc)。更多Python应用预测开发可以参考位于 [demo/python](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/python) 下的示例代码，或者位于[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)的工程示例代码。
