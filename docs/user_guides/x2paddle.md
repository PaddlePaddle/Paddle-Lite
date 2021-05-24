# 模型转换工具 X2Paddle

X2Paddle可以将caffe、tensorflow、onnx模型转换成Paddle支持的模型。目前支持版本为caffe 1.0；tensorflow 1.x，推荐1.4.0；ONNX 1.6.0，OpSet支持 9, 10, 11版本。如果您使用的是PyTorch框架，请先转换为ONNX模型之后再使用X2Paddle工具转化为Paddle模型。

[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)支持将Caffe/TensorFlow/ONNX模型转换为PaddlePaddle模型。



## 安装

- 环境依赖
  - python >= 3.5
  - paddlepaddle >= 2.0.0
```
pip install x2paddle
```

安装最新版本，可使用如下安装方式

```
pip install git+https://github.com/PaddlePaddle/X2Paddle.git@develop
```

## 使用

### Caffe

```
x2paddle --framework=caffe \
         --prototxt=deploy.prototxt \
         --weight=deploy.caffemodel \
         --save_dir=pd_model
```

### TensorFlow

```
x2paddle --framework=tensorflow \
         --model=tf_model.pb \
         --save_dir=pd_model
```

### ONNX

```
x2paddle --framework=onnx \
         --model=onnx_model.onnx \
         --save_dir=pd_model
```

## 转换结果说明

在指定的`save_dir`下生成两个目录  
1. inference_model : 模型结构和参数均序列化保存的模型格式
2. model_with_code : 保存了模型参数文件和模型的python代码

## 问题反馈

受限于不同框架的差异，部分模型可能会存在目前无法转换的情况，如TensorFlow中包含控制流的模型，NLP模型等。对于CV常见的模型，如若您发现无法转换或转换失败，存在较大差异等问题，欢迎您将问题或Bug报告以[Github Issues](https://github.com/PaddlePaddle/X2Paddle/issues)的形式提交给我们，我们会实时跟进。
