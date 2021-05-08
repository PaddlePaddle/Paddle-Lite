# 性能数据

大家可以参考[测试方法文档](benchmark_tools)对模型进行测试。

## ARM测试环境

* 测试模型
    * fp32模型
        * mobilenet_v1
        * mobilenet_v2
        * squeezenet_v1.1
        * mnasnet
        * shufflenet_v1
    
    * int8模型
        * mobilenet_v1
        * mobilenet_v2

* 测试机器(android ndk ndk-r17c)
   *  骁龙855
      * xiaomi mi9, snapdragon 855 (enable sdot instruction)
      * 4xA76(1@2.84GHz + 3@2.4GHz) + 4xA55@1.78GHz

   *  骁龙845
      * xiaomi mi8, 845
      * 2.8GHz（大四核），1.7GHz（小四核）

   *  骁龙835
      * xiaomi mix2, snapdragon 835
      * 2.45GHz（大四核），1.9GHz（小四核）
 
* 测试说明
    * branch: release/v2.8
    * warmup=10, repeats=100，统计平均时间，单位是ms
    * 当线程数为1时，```DeviceInfo::Global().SetRunMode```设置LITE_POWER_HIGH，否者设置LITE_POWER_NO_BIND
    * 模型的输入图像的维度是{1, 3, 224, 224}，输入图像的每一位数值是1
    
## ARM测试数据


### fp32模型测试数据

#### paddlepaddle model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |34.21 |19.78 |11.53 |29.93 |17.34 |10.04 |
mobilenet_v2 |23.59 |14.07 |8.47 |21.30 |12.89 |7.81 |
shufflenet_v1 |4.09 |2.88 |2.04 |3.96 |2.67 |2.08 |
squeezenet_v1.1 |18.98 |12.50 |8.18 |16.63 |11.49 |7.48 |
mnasnet |29.47 |12.75 |7.26 |22.92 |11.85 |6.71 |

骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |64.26 |36.71 |18.32 |62.19 |32.08 |16.89 |
mobilenet_v2 |43.28 |24.48 |13.69 |40.31 |22.43 |12.72 |
shufflenet_v1 |7.39 |4.56 |3.18 |7.18 |4.63 |3.24 |
squeezenet_v1.1 |35.21 |22.38 |12.91 |32.71 |20.41 |12.07 |
mnasnet |38.33 |26.26 |12.21 |37.42 |20.61 |11.57 |

骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |91.63 |50.36 |29.94 |86.86 |46.39 |26.43 |
mobilenet_v2 |62.3 |35.29 |22.01 |57.64 |32.83 |19.25 |
shufflenet_v1 |9.81 |5.99 |4.19 |9.20 |5.77 |4.05 |
squeezenet_v1.1 |51.22 |32.70 |19.86 |47.23 |30.59 |18.11 |
mnasnet |57.17 |32.60 |19.67 |53.74 |30.02 |17.74 |

#### caffe model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |32.23 |18.60 |10.61 |30.94 |18.19 |9.94 |
mobilenet_v2 |29.89 |17.46 |10.81 |27.03 |16.30 |9.73 |
shufflenet_v1 |4.86 |2.94 |2.10 |3.89 |2.82 |2.11 |

骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |65.20 |35.11 |18.92 |61.25 |32.15 |17.32 |
mobilenet_v2 |55.53 |30.83 |17.56 |51.62 |28.92 |15.95 |
shufflenet_v1 |7.38 |4.55 |3.19 |7.16 |4.35 |3.30 |

骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |92.31 |50.94 |30.72 |87.47 |46.41 |26.19 |
mobilenet_v2 |81.32 |45.10 |28.12 |75.57 |42.47 |25.71 |
shufflenet_v2 |9.91 |5.98 |4.20 |9.59 |5.76 |4.06 |

#### int8量化模型测试数据

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |21.25 |10.88 |5.43 | 13.19 |7.66 |3.95 |
mobilenet_v2 |16.99 |10.23 |5.68 | 12.63 |7.59 |4.34 |

骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |51.43 |28.14 |14.37 | 45.17 |33.12 |12.60 |
mobilenet_v2 |38.98 |21.64 |11.80 | 33.12 |18.44 |10.02 |

骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |61.91 |32.75 |16.60 |57.46 |30.03 |15.37 |
mobilenet_v2 |48.87 |26.15 |13.74 |42.61 |22.63 |11.79 |


## 华为麒麟NPU的性能数据
请参考[PaddleLite使用华为麒麟NPU预测部署](../demo_guides/huawei_kirin_npu)的最新性能数据

## 瑞芯微NPU的性能数据
请参考[PaddleLite使用瑞芯微NPU预测部署](../demo_guides/rockchip_npu)的最新性能数据

## 联发科APU的性能数据
请参考[PaddleLite使用联发科APU预测部署](../demo_guides/mediatek_apu)的最新性能数据

## 颖脉NNA的性能数据
请参考[PaddleLite使用颖脉NNA预测部署](../demo_guides/imagination_nna)的最新性能数据
