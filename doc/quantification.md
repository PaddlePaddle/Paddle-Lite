# Quantification 模型量化、反量化

## 背景故事
部分网络如AlexNet训练出的模型体积较大，不适宜在移动设备上使用。


## 解决模型过大办法
1. 选用适合移动端的模型结构如：mobilenet、googlenet、 yolo、squeezenet 等；
2. 使用我们提供的量化工具，可以在几乎不影响精度的情况下将float32模型减小至原模型的 1/4；

- - - - - 
## 量化工具介绍

### 模型转化工具目录：

- [量化工具目录](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/tools/quantification)

- [模型转化工具](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/tools/quantification/convert.cpp)

#### 使用说明
- [工具使用](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/tools/quantification/README.md)

## 如何读取量化后的模型
load方法中添加了 quantification 参数，默认为false。 如果需要load量化后的模型，按需传参即可。

[我是源代码](https://github.com/PaddlePaddle/paddle-mobile/blob/55302b33ea3bd68c9797d8f65e527544792b8095/src/io/paddle_mobile.h)

```c++
bool Load(const std::string &dirname, bool optimize = false,
            bool quantification = false, int batch_size = 1);
```

- - - - - 






