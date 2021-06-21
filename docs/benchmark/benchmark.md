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
    * branch: release/v2.9
    * warmup=10, repeats=100，统计平均时间，单位是ms
    * 当线程数为1时，```DeviceInfo::Global().SetRunMode```设置LITE_POWER_HIGH，否者设置LITE_POWER_NO_BIND
    * 模型的输入图像的维度是{1, 3, 224, 224}，输入图像的每一位数值是1
## ARM测试数据


### fp32模型测试数据

#### paddlepaddle model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |32.19 |18.75 |11.02 |29.50 |17.50 |9.58 
mobilenet_v2 |23.77 |14.23 |8.52 |19.98 |12.19 |7.44 
shufflenet_v2 |10.63 |6.60 |4.24 |9.74 |6.02 |3.99 
squeezenet |17.44 |11.39 |7.50 |15.33 |10.04 |6.91 
mnasnet |20.54 |12.30 |7.04 |17.62 |10.62 |6.34 



骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |65.28 |36.37 |22.88 |59.27 |32.62 |19.57 
mobilenet_v2 |43.40 |24.33 |15.43 |38.15 |21.77 |13.81 
shufflenet_v2 |20.09 |11.55 |7.57 |18.45 |10.91 |7.16 
squeezenet |32.89 |21.24 |13.46 |30.20 |19.30 |12.83 
mnasnet |39.22 |21.41 |12.92 |34.79 |19.39 |12.05 



骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |92.40 |51.29 |33.14 |86.65 |48.89 |27.06 
mobilenet_v2 |63.60 |36.32 |23.82 |61.17 |33.08 |19.89 
shufflenet_v2 |27.54 |16.75 |11.05 |24.02 |14.24 |8.74 
squeezenet |47.71 |31.51 |20.51 |43.30 |27.07 |16.74 
mnasnet |59.17 |32.38 |20.71 |51.29 |28.32 |16.95 

#### caffe model

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |32.14 |18.70 |10.91 |29.49 |17.48 |9.60 
mobilenet_v2 |29.87 |17.64 |10.68 |25.82 |15.58 |9.29 
shufflenet_v1 |3.96 |2.80 |2.05 |3.64 |2.70 |2.04 



骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |65.04 |36.13 |21.99 |58.55 |32.89 |19.18 
mobilenet_v2 |55.35 |31.56 |19.63 |49.06 |27.87 |17.36 
shufflenet_v1 |7.20 |4.44 |3.21 |6.75 |4.50 |3.26 



骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |94.00 |52.42 |31.61 |85.96 |45.89 |49.02 
mobilenet_v2 |81.01 |46.32 |29.10 |81.07 |43.46 |42.66 
shufflenet_v1 |10.22 |6.23 |4.60 |10.04 |6.11 |4.13 

#### int8量化模型测试数据

骁龙855|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |19.00 |10.93 |5.97 | 13.08 |7.68 |3.98 
mobilenet_v2 |17.68 |10.49 |5.93 | 12.76 |7.70 |4.36 

骁龙845|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |51.37 |28.11 |15.50 | 45.06 |24.47 |13.80 
mobilenet_v2 |38.90 |21.64 |12.33 | 33.03 |18.71 |10.77 

骁龙835|armv7 | armv7 |  armv7 |armv8 | armv8 |armv8 
----| ---- | ---- | ---- | ----  |----  |----
threads num|1 |2 |4 |1 |2 |4 
mobilenet_v1 |60.48 |31.94 |16.53 |56.70 |29.73 |15.22 
mobilenet_v2 |47.02 |25.34 |13.57 |41.75 |22.27 |11.94 


## 华为麒麟NPU的性能数据
请参考[PaddleLite使用华为麒麟NPU预测部署](../demo_guides/huawei_kirin_npu)的最新性能数据

## 瑞芯微NPU的性能数据
请参考[PaddleLite使用瑞芯微NPU预测部署](../demo_guides/rockchip_npu)的最新性能数据

## 联发科APU的性能数据
请参考[PaddleLite使用联发科APU预测部署](../demo_guides/mediatek_apu)的最新性能数据

## 颖脉NNA的性能数据
请参考[PaddleLite使用颖脉NNA预测部署](../demo_guides/imagination_nna)的最新性能数据
