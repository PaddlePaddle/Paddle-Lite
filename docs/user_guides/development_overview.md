# 程序开发概览

以下介绍了该工作流程的每一个步骤，并提供了进一步说明的链接：

### 1. 创建 Paddle Lite 模型

您可以通过以下方式生成 Paddle Lite 模型：

- 将 Paddle 模型转换为 Paddle Lite 模型：使用 [Paddle Lite opt 工具](model_optimize_tool) 将 Paddle 模型转换为 Paddle Lite 模型。在转换过程中，您可以应用量化等优化措施，以缩减模型大小和缩短延时，并最大限度降低或完全避免准确率损失。

### 2. 运行推断

推断是指在设备上执行 Paddle Lite 模型，以便根据输入数据进行预测的过程。您可以通过以下方式运行推断：

- 使用 Paddle Lite API，在多个平台和语言中均受支持（如 [Java](../user_guides/java_demo)、[C++](../user_guides/cpp_demo)、[Python](../user_guides/python_demo)）
  - 配置参数（`MobileConfig`），设置模型来源等
  - 创建推理器（`Predictor`），调用 `CreatePaddlePredictor` 接口即可创建
  - 设置模型输入，通过 `predictor->GetInput(i)` 获取输入变量，并为其指定大小和数值
  - 执行预测，只需要调用 `predictor->Run()`
  - 获得输出，使用 `predictor->GetOutput(i)` 获取输出变量，并通过 `data<T>` 取得输出值
