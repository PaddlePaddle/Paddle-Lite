# PaddleJS Operators 支持表

Operators表示神经网络每层对应的算子，参考具体的算法实现，表格显示了百度人工智能算子支持情况，PadderJS目前支持GPU操作计算版本。

受paddle.js支持的列表，请参阅兼容性，此文件会随着Operator数量增加和支持情况做相应的变更。



## 演示

| Operator      | Gpu Backend    | desc     |
| ------------- | ------------- | ------------- |
| conv2d_transpose   |  webGL1、 webGL2   | |
| conv2d   |  webGL1、 webGL2   | |
| conv2d_depthwise   |  webGL1、 webGL2   | |
| conv2d_elementwise_add   |  webGL1、 webGL2   | |
| conv2d_elementwise_add_winograd   |  webGL1、 webGL2   | |
| dynamic   |  webGL1、 webGL2   | |
| scale   |  webGL1、 webGL2   | |
| pool2d   |  webGL1、 webGL2   | |
| pool2d_max   |  webGL1、 webGL2   | |
| pool2d_winograd   |  webGL1、 webGL2   | |
| elementwise_add   |  webGL1、 webGL2   | |
| mul   |  webGL1、 webGL2   | |
| relu   |  webGL1、 webGL2   | |
| relu6   |  webGL1、 webGL2   | |
| softmax   |  webGL1、 webGL2   | |
| batchnorm   |  webGL1、 webGL2   | |
| reshape   |  webGL1、 webGL2   | |
| transpose   |  webGL1、 webGL2   | |



## 浏览器覆盖面

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser


