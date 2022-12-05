# Road map

这篇文档会介绍 Paddle Lite 近期对外的开源版本和计划。

其中包含的 feature 为最小集合，按最终发布的版本为准。

## 2.12
- 易用性提升: 支持同一 FP32 模型在不同 Arm CPU 架构下运行期间动态支持 FP32 和 FP16 精度的推理，初步完成框架与 Arm CPU 计算库编译解耦。
- 量化推理: 支持 PaddleSlim 量化新格式模型，降低在不同硬件的迁移成本；新增 Armv9 和 SVE 指令支持，MobileNetV1 和 MobileNetV2 模型性能分别提升 21% 和 10% ，其它模型上均有不同程度的性能提升。
- 新硬件支持: 新增支持高通 QNN 及 SA8295P 芯片，支持 Linux、Android、QNX 操作系统，支持 HTP 后端 INT8、FP16、INT8 和 FP16 混合精度。

## 2.11
- 新增非结构化 1x1 稀疏卷积实现，非结构化稀疏卷积 相对于稠密卷积，在 75% 稀疏下，性能有20%-40% 提升（支持int8/fp32精度计算）
- 新增非结构化 1x1 稀疏卷积实现，非结构化稀疏卷积 相对于稠密卷积，在 75% 稀疏下，性能有20%-40% 提升（支持int8/fp32精度计算）
- 新增 “全流程/多后端” 稳定性主动验证方法 AutoScanTester

## 2.10
- 新增 Apple Metal 后端支持
- 新增 NNAdapter: 飞桨推理 AI 硬件统一适配框架
- ARM CPU 性能增强
- 编译策略升级
- benchmark 工具升级

## 2.9.1
- ARM CPU FP32  和 Int8 在重点模型上性能优化
- OpenCL 重点模型，高低端硬件性能优化
- 库体积进一步压缩，根据模型裁剪算子效果会更明显

## 2.9 
- ARM CPU
  - 支持 FP16 执行
  
- OpenCL 性能增强

## 2.8
- 框架升级
    - opt 工具功能增强：+量化模型压缩功能
    - 版本间的兼容性增减：+算子版本控制方法
    - 编译系统优化：减少编译耗时
    - 文档易用性增强
- 硬件平台增强：昆仑 XPU、RK NPU、ARM OPENCL
- 性能增强：ARM模型性能提升
- 支持Paddle2.0： 支持更多Paddle2.0模型和算子

## 2.7
[v2.7 project](https://github.com/PaddlePaddle/Paddle-Lite/milestones/v2.7)

## 2.6
[v2.6 project](https://github.com/PaddlePaddle/Paddle-Lite/milestones/v2.6)

## 2.3
[v2.3 project](https://github.com/PaddlePaddle/Paddle-Lite/milestone/3?closed=1)

## 2.0.0-beta1
- `model_optimize_tool` 从 ARM 上执行修改为 Host 上执行，只从 kernel 分布来确定计算图优化；后续硬件针对优化会发布新的工具；
- Paddle 模型支持参数 composed  的格式
- 增加分层编译来控制常用模型的部署库的大小，分两个模式 `basic`, `extra`；默认 `basic` 模式只发布核心的op 和kernel；将控制流相关的Op和kernel 折叠进 `extra` 按需编译
- 增加 INT8 量化，从 PaddleSlim 训练到 PaddleLite 部署完整案例
- 支持内存中加载模型，以支持 APP 的简易加密

## 2.0.0-beta1-prerelease
- 完善编译和 benchmark 文档
- 增加第三方依赖代码的离线下载功能，加速编译过程
- 去掉 `tiny_publish` 模式下无关的第三方代码下载，可以不依赖任何第三方
