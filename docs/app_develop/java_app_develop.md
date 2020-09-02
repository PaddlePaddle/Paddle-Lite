# Java 应用开发

Java代码调用Paddle-Lite执行预测库仅需以下五步：

(1) 设置config信息

```java
MobileConfig config = new MobileConfig();
config.setModelDir(modelPath);
config.setPowerMode(PowerMode.LITE_POWER_HIGH);
config.setThreads(1);
```

(2) 创建predictor

```java
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);
```

(3) 设置模型输入 (下面以全一输入为例)

```java
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
    inputBuffer[i] = i;
}
Tensor input = predictor.getInput(0);
input.resize({100, 100});
input.setData(inputBuffer);
```

(4) 执行预测

```java
predictor.run();
```

(5) 获得预测结果

```java
Tensor output = predictor.getOutput(0);
```

详细的Java API说明文档位于[Java API](../api_reference/java_api_doc)。更多Java应用预测开发可以参考位于 [demo/java](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/java) 下的示例代码，或者位于[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)的工程示例代码。
