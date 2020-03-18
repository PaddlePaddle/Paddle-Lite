[中文版](./README_cn.md)
# PaddleJS Operators Support Table

Operators represent the operators corresponding to each layer of the neural network. Refer to the specific algorithm implementation, the table shows the support of Baidu artificial intelligence operators. Padderjs currently supports GPU operation calculation version.

See Compatibility for a list of the supported platforms.

Please refer to compatibility for the list supported by paddle.js. This file will change as the number of operators increases and the support situation changes.

Baidu paddlejs uses the ready-made JavaScript model or transforms the paddle model to run in the browser.

## Demonstration


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


## Browser coverage

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser

