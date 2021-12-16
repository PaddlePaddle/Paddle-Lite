# Road map

这篇文档会介绍 Paddle Lite 近期对外的开源版本和计划。

其中包含的 feature 为最小集合，按最终发布的版本为准。

以下计划按时间排序，最新的计划在末尾


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

## 2.7
[v2.7 project](https://github.com/PaddlePaddle/Paddle-Lite/milestones/v2.7)

## 2.8
- 框架升级
    - opt 工具功能增强：+量化模型压缩功能
    - 版本间的兼容性增减：+算子版本控制方法
    - 编译系统优化：减少编译耗时
    - 文档易用性增强
- 硬件平台增强：昆仑 XPU、RK NPU、ARM OPENCL
- 性能增强：ARM模型性能提升
- 支持Paddle2.0： 支持更多Paddle2.0模型和算子

## 2.9 (目前最新版本)

- ARM CPU
  - 支持 FP16 执行
  
- OpenCL 性能增强

## 2.9.1 (in plan)
- ARM CPU FP32  和 Int8 在重点模型上性能优化
- OpenCL 重点模型，高低端硬件性能优化
- 库体积进一步压缩，根据模型裁剪算子效果会更明显

## 2.10 (in plan)
- FP16 支持执行时，根据芯片能力决定是否切换回 FP32 执行
