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
      * xiaomi mi9, snapdragon 855 
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
    * branch: release/v2.3.0
    * warmup=10, repeats=30，统计平均时间，单位是ms
    * 当线程数为1时，```DeviceInfo::Global().SetRunMode```设置LITE_POWER_HIGH，否者设置LITE_POWER_NO_BIND
    * 模型的输入图像的维度是{1, 3, 224, 224}，输入图像的每一位数值是1
    
## 测试数据


### fp32模型测试数据

#### paddlepaddle model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |33.27 |19.52 |11.14 |31.72 |18.76 |10.24 |
mobilenet_v2 |29.08 |15.79 |9.25 |25.89 |14.17 |8.38 |
shufflenet_v2 |4.40 |3.09 |2.30 |4.28 |3.02 |2.35 |
squeezenet_v1.1 |19.96 |12.61 |8.76 |18.25 |11.46 |7.97 |
mnasnet |21.00 |12.54 |7.28 |19.65 |11.65 |6.96 |


骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |66.36 |35.97 |19.45 |62.66 |33.87 |17.85 |
mobilenet_v2 |45.86 |25.53 |14.6 |41.58 |23.24 |13.39 |
shufflenet_v2 |7.58 |4.89 |3.41 |7.44 |4.91 |3.58 |
squeezenet_v1.1 |37.15 |22.74 |13.51 |34.69 |21.27 |12.74 |
mnasnet |40.09 |21.73 |11.91 |38.19 |21.02 |12.11 |


骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |96.98 |53.92 |32.24 |89.31 |48.02 |27.58 |
mobilenet_v2 |67.72 |37.66 |23.82 |60.10 |34.36 |21.05 |
shufflenet_v2 |10.72 |6.62 |4.63 |10.10 |6.44 |4.63 |
squeezenet_v1.1 |53.89 |33.28 |20.73 |50.83 |32.31 |19.51 |
mnasnet |59.55 |33.53 |20.32 |56.21 |31.58 |19.06 |

#### caffe model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |33.36 |19.45 |11.26 |31.63 |18.74 |10.31 |
mobilenet_v2 |31.63 |19.21 |11.61 |28.34 |17.14 |10.16 |
shufflenet_v2 |4.46 |3.08 |2.32 |4.26 |2.98 |2.35 |


骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |66.32 |35.83 |19.56 |62.52 |33.79 |17.91 |
mobilenet_v2 |58.46 |32.69 |18.56 |53.72 |29.86 |16.80 |
shufflenet_v2 |7.65 |4.82 |3.46 |7.55 |4.97 |3.62 |


骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |95.38 |54.09 |32.03 |95.05 |48.33 |27.54 |
mobilenet_v2 |88.46 |48.98 |30.23 |79.28 |44.64 |27.10 |
shufflenet_v2 |10.07 |6.51 |4.61 |10.31 |6.50 |4.66 |

#### int8量化模型测试数据

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |36.80 |21.58 |11.12 | 14.01 |8.13 |4.32 |
mobilenet_v2 |28.72 |19.08 |12.49 | 17.24 |11.55 |7.82 |


骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |60.76 |32.25 |16.66 |56.57 |29.84 |15.24 |
mobilenet_v2 |49.38 |31.10 |22.07 |47.52 |28.18 |19.24 |


麒麟970|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |65.95 |34.39 |18.68 |60.86 |30.98 |16.31 |
mobilenet_v2 |68.87 |39.39 |24.43 |65.57 |37.31 |20.87 |
