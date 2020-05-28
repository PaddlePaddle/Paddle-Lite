# 性能数据

可以参考[benchmark_tools](benchmark_tools)，推荐**一键benchmark**。

## 测试环境

* 测试模型
    * fp32模型
        * mobilenet_v1
        * mobilenet_v2
        * squeezenet_v1.1
        * mnasnet
        * shufflenet_v2
    
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

   * 麒麟970
      * HUAWEI Mate10
 
* 测试说明
    * branch: release/v2.6.0
    * warmup=10, repeats=30，统计平均时间，单位是ms
    * 当线程数为1时，```DeviceInfo::Global().SetRunMode```设置LITE_POWER_HIGH，否者设置LITE_POWER_NO_BIND
    * 模型的输入图像的维度是{1, 3, 224, 224}，输入图像的每一位数值是1
    
## 测试数据


### fp32模型测试数据

#### paddlepaddle model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |35.11 |20.67 |11.83 |30.56 |18.59 |10.44 |
mobilenet_v2 |26.36 |15.83 |9.29 |21.64 |13.25 |7.95 |
shufflenet_v2 |4.56 |3.14 |2.35 |4.07 |2.89 |2.28 |
squeezenet_v1.1 |21.27 |13.55 |8.49 |18.05 |11.51 |7.83 |
mnasnet |21.40 |13.18 |7.63 |18.84 |11.40 |6.80 |


骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |65.56 |37.17 |19.65 |63.23 |32.98 |17.68 |
mobilenet_v2 |45.89 |25.20 |14.39 |41.03 |22.94 |12.98 |
shufflenet_v2 |7.31 |4.66 |3.27 |7.08 |4.71 |3.41 |
squeezenet_v1.1 |36.98 |22.53 |13.45 |34.27 |20.96 |12.60 |
mnasnet |39.85 |23.64 |12.25 |37.81 |20.70 |11.81 |


骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |92.77 |51.56 |30.14 |87.46 |48.02 |26.42 |
mobilenet_v2 |65.78 |36.52 |22.34 |58.31 |33.04 |19.87 |
shufflenet_v2 |10.39 |6.26 |4.46 |9.72 |6.19 |4.41 |
squeezenet_v1.1 |53.59 |33.16 |20.13 |51.56 |31.81 |19.10 |
mnasnet |57.44 |32.62 |19.47 |54.99 |30.69 |17.98 |

#### caffe model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |32.38 |18.65 |10.69 |30.75 |18.11 |9.88 |
mobilenet_v2 |29.45 |17.86 |10.81 |26.61 |16.26 |9.67 |
shufflenet_v2 |5.04 |3.14 |2.20 |4.09 |2.85 |2.25 |


骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |65.26 |35.19 |19.11 |61.42 |33.15 |17.48 |
mobilenet_v2 |55.59 |31.31 |17.68 |51.54 |29.69 |16.00 |
shufflenet_v2 |7.42 |4.73 |3.33 |7.18 |4.75 |3.39 |


骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |95.38 |52.16 |30.37 |92.10 |46.71 |26.31 |
mobilenet_v2 |82.89 |45.49 |28.14 |74.91 |41.88 |25.25 |
shufflenet_v2 |10.25 |6.36 |4.42 |9.68 |6.20 |4.42 |

#### int8量化模型测试数据

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |37.18 |21.71 |11.16 | 14.41 |8.34 |4.37 |
mobilenet_v2 |27.95 |16.57 |8.97 | 13.68 |8.16 |4.67 |


骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |61.63 |32.60 |16.49 |57.36 |29.74 |15.50 |
mobilenet_v2 |47.13 |25.62 |13.56 |41.87 |22.42 |11.72 |


麒麟970|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |63.13 |32.63 |16.85 |58.92 |29.96 |15.42 |
mobilenet_v2 |48.60 |25.43 |13.76 |43.06 |22.10 |12.09 |
