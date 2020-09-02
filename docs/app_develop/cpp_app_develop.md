# C++ 应用开发

C++代码调用Paddle-Lite执行预测库仅需以下五步：

(1) 引用头文件和命名空间

```c++
#include "paddle_api.h"
using namespace paddle::lite_api;
```

(2) 指定模型文件，创建Predictor

```C++
// 1. Set MobileConfig, model_file_path is 
// the path to model model file. 
MobileConfig config;
config.set_model_from_file(model_file_path);
// 2. Create PaddlePredictor by MobileConfig
std::shared_ptr<PaddlePredictor> predictor =
    CreatePaddlePredictor<MobileConfig>(config);
```

(3) 设置模型输入 (下面以全一输入为例)

```c++
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}
```

(4) 执行预测

```c++
predictor->Run();
```

(5) 获得预测结果

```c++
std::unique_ptr<const Tensor> output_tensor(
    std::move(predictor->GetOutput(0)));
// 转化为数据
auto output_data=output_tensor->data<float>();
```

详细的C++ API说明文档位于[C++ API](../api_reference/cxx_api_doc)。更多C++应用预测开发可以参考位于 [demo/c++](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/cxx) 下的示例代码，或者位于[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)的工程示例代码。
