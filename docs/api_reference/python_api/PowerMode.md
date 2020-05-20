## PowerMode

```python
class PowerMode;
```

`PowerMode`为ARM CPU能耗模式，用户可以根据应用场景设置能耗模式获得最优的能效比。

示例：

```python
from paddlelite.lite import *

config = MobileConfig()
# 设置NaiveBuffer格式模型目录
config.set_model_dir(<your_model_dir_path>)
# 设置能耗模式
config.set_power_mode(PowerMode.LITE_POWER_NO_BIND)

# 根据MobileConfig创建LightPredictor
predictor = create_paddle_predictor(config)
```

PowerMode详细说明如下：

|         选项         | 说明                                                         |
| :------------------: | ------------------------------------------------------------ |
|   LITE_POWER_HIGH    | 绑定大核运行模式。如果ARM CPU支持big.LITTLE，则优先使用并绑定Big cluster。如果设置的线程数大于大核数量，则会将线程数自动缩放到大核数量。如果系统不存在大核或者在一些手机的低电量情况下会出现绑核失败，如果失败则进入不绑核模式。 |
|    LITE_POWER_LOW    | 绑定小核运行模式。如果ARM CPU支持big.LITTLE，则优先使用并绑定Little cluster。如果设置的线程数大于小核数量，则会将线程数自动缩放到小核数量。如果找不到小核，则自动进入不绑核模式。 |
|   LITE_POWER_FULL    | 大小核混用模式。线程数可以大于大核数量。当线程数大于核心数量时，则会自动将线程数缩放到核心数量。 |
|  LITE_POWER_NO_BIND  | 不绑核运行模式（推荐）。系统根据负载自动调度任务到空闲的CPU核心上。 |
| LITE_POWER_RAND_HIGH | 轮流绑定大核模式。如果Big cluster有多个核心，则每预测10次后切换绑定到下一个核心。 |
| LITE_POWER_RAND_LOW  | 轮流绑定小核模式。如果Little cluster有多个核心，则每预测10次后切换绑定到下一个核心。 |
