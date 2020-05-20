
## create_paddle_predictor

```python
CxxPredictor create_paddle_predictor(config); # config为CxxConfig类型
LightPredictor create_paddle_predictor(config); # config为MobileConfig类型
```

`create_paddle_predictor`函数用来根据`CxxConfig`或`MobileConfig`构建预测器。

示例：

```python
from paddlelite.lite import *

# 设置CxxConfig
config = CxxConfig()
config.set_model_dir(<your_model_dir_path>)
places = [Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据CxxConfig创建CxxPredictor
predictor = create_paddle_predictor(config)
```

参数：

- `config(CxxConfig或MobileConfig)` - 用于构建Predictor的配置信息。

返回：预测器`predictor`

返回类型：`CxxPredictor`或`LightPredictor`
