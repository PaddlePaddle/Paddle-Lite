# 性能数据

可以参考[benchmark_tools](benchmark_tools)，推荐**一键benchmark**。

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


## 华为麒麟NPU测试环境

* 测试模型
    * fp32模型
        * mobilenet_v1
        * mobilenet_v2
        * squeezenet_v1.1
        * mnasnet

* 测试机器(android ndk ndk-r17c)
   *  麒麟810
      * HUAWEI Nova5, Kirin 810
      * 2xCortex A76 2.27GHz + 6xCortex A55 1.88GHz

   *  麒麟990
      * HUAWEI Mate 30, Kirin 990
      * 2 x Cortex-A76 Based 2.86 GHz + 2 x Cortex-A76 Based 2.09 GHz + 4 x Cortex-A55 1.86 GHz

   *  麒麟990 5G
      * HUAWEI P40, Kirin 990 5G
      * 2 x Cortex-A76 Based 2.86GHz + 2 x Cortex-A76 Based 2.36GHz + 4 x Cortex-A55 1.95GHz

* HIAI ddk 版本： 310 or 320
 
* 测试说明
    * branch: release/v2.6.1
    * warmup=10, repeats=30，统计平均时间，单位是ms
    * 线程数为1，```DeviceInfo::Global().SetRunMode```设置LITE_POWER_HIGH
    * 模型的输入图像的维度是{1, 3, 224, 224}，输入图像的每一位数值是1
    
## 华为麒麟NPU测试数据

#### paddlepaddle model

- ddk 310

|Kirin |810||990||990 5G||
|---|---|---|---|---|---|---|
|  |cpu(ms) | npu(ms) |cpu(ms) | npu(ms) |cpu(ms) | npu(ms) |
|mobilenet_v1|	 41.20|  12.76|  31.91|  4.07|  33.97|  3.20|
|mobilenet_v2|	 29.57|  12.12|  22.47|  5.61|  23.17|  3.51|
|squeezenet|  23.96|  9.04|  17.79|  3.82|	 18.65|  3.01|
|mnasnet|  26.47|  13.62|  19.54|  5.17|	 20.34|  3.32|


- ddk 320

|模型 |990||990-5G||
|---|---|---|---|---|
||cpu(ms) | npu(ms) |cpu(ms) | npu(ms) |
|ssd_mobilenetv1|  65.67|  18.21|  71.8|	16.6|


*说明：ssd_mobilenetv1的npu性能为npu、cpu混合调度运行的总时间*
