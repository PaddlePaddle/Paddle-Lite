# Road map

这篇文档会介绍 Paddle-Lite 近期对外的开源版本和计划。

其中包含的 feature 为最小集合，按最终发布的版本为准。


## 2.0.0-beta1-prerelease

预计发布 *2019-8-26 ~ 2days*

- 完善编译和 benchmark 文档
- 增加第三方依赖代码的离线下载功能，加速编译过程
- 去掉 `tiny_publish` 模式下无关的第三方代码下载，可以不依赖任何第三方

## 2.0.0-beta1

预计发布 *2019-9-1~2days*

- `model_optimize_tool` 从 ARM 上执行修改为 Host 上执行，只从 kernel 分布来确定计算图优化；后续硬件针对优化会发布新的工具；
- Paddle 模型支持参数 composed  的格式
- 增加分层编译来控制常用模型的部署库的大小，分两个模式 `basic`, `extra`；默认 `basic` 模式只发布核心的op 和kernel；将控制流相关的Op和kernel 折叠进 `extra` 按需编译
- 增加 INT8 量化，从 PaddleSlim 训练到 PaddleLite 部署完整案例
- 支持内存中加载模型，以支持 APP 的简易加密

## 2.3

[v2.3 project](https://github.com/PaddlePaddle/Paddle-Lite/milestone/3?closed=1)

## 2.6

[v2.6 project](https://github.com/PaddlePaddle/Paddle-Lite/milestones/v2.6)
