# Benchmark 数据

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
        * resnet50

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
 
   *  骁龙625
      * oppo R9s, snapdragon625
      * A53 x 8, big core@2.0GHz
 
   * 骁龙653
      * 360 N5, snapdragon 653
      * 4 x A73@2.0GHz + 4 x A53@1.4GHz
 
   * 麒麟970
      * HUAWEI Mate10
 
* 测试说明
    * branch: release/2.0.0
    * warmup=10, repeats=30，统计平均时间，单位是ms
    * 当线程数为1时，```DeviceInfo::Global().SetRunMode```设置LITE_POWER_HIGH，否者设置LITE_POWER_NO_BIND
    * 模型的输入图像的维度是{1, 3, 224, 224}，输入图像的每一位数值是1
    
## 测试数据


### fp32模型测试数据

#### paddlepaddle model


骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |32.19 |18.81 |10.90 |30.92 |18.31 |10.15 
mobilenet_v2 |22.91 |13.75 |8.64 |21.15 |12.79 |7.84 
shufflenet_v2 |4.67 |3.37 |2.65 |4.43 |3.15 |2.66 
squeezenet_v1.1 |25.10 |15.93 |9.68 |23.28 |14.61 |8.71 
mnasnet |21.84 |13.14 |7.96 |19.61 |11.88 |7.55



骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |94.13 |52.17 |30.68 |88.28 |47.58 |26.64 
mobilenet_v2 |61.24 |34.64 |22.36 |56.66 |32.19 |19.63 
shufflenet_v2 |10.87 |6.92 |5.12 |10.41 |6.76 |4.97 
squeezenet_v1.1 |73.61 |42.25 |24.44 |64.87 |38.43 |23.06 
mnasnet |58.22 |33.43 |20.44 |53.43 |30.20 |18.09 


麒麟980|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |55.11 |28.24 |13.27 |34.24 |17.74 |12.41 
mobilenet_v2 |37.03 |19.80 |51.94 |23.64 |12.98 |9.38 
shufflenet_v2 |7.26 |4.94 |15.06 |5.32 |3.33 |2.82 
squeezenet_v1.1 |42.73 |23.66 |57.39 |26.03 |14.53 |13.66 
mnasnet |36.87 |20.15 |46.04 |21.85 |12.06 |8.68 

麒麟970|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |97.80 |52.64 |34.46 |94.51 |49.36 |28.43 
mobilenet_v2 |66.55 |38.52 |23.19 |62.89 |34.93 |21.53 
shufflenet_v2 |13.78 |8.11 |5.93 |11.95 |7.90 |5.91 
squeezenet_v1.1 |77.64 |43.67 |25.72 |69.91 |40.66 |24.62 
mnasnet |61.86 |34.62 |22.68 |59.61 |32.79 |19.56 

#### caffe model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |32.42 |18.68 |10.86 |30.92 |18.35 |10.07 |
mobilenet_v2 |29.53 |17.76 |10.89 |27.19 |16.53 |9.75 |
shufflenet_v2 |4.61 |3.29 |2.61 |4.36 |3.11 |2.51 |


骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |92.52 |52.34 |30.37 |88.31 |49.75 |27.29 |
mobilenet_v2 |79.50 |45.67 |28.79 |76.13 |44.01 |26.13 |
shufflenet_v2 |10.94 |7.08 |5.16 |10.64 |6.83 |5.01 |


麒麟980|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |55.36 |28.18 |13.31 |34.42 |17.93 |12.52 |
mobilenet_v2 |49.17 |26.10 |65.49 |30.50 |16.66 |11.72 |
shufflenet_v2 |8.45 |5.00 |15.65 |4.58 |3.14 |2.83 |


麒麟970|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 |
mobilenet_v1 |97.85 |53.38 |33.85 |94.29 |49.42 |28.29 |
mobilenet_v2 |87.40 |50.25 |31.85 |85.55 |48.11 |28.24 |
shufflenet_v2 |12.16 |8.39 |6.21 |12.21 |8.33 |6.32 |

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
