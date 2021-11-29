# 性能数据

请参考[性能测试文档](benchmark_tools)对模型进行测试。

## 测试环境

* 模型
    * fp32 浮点模型
        * [MobileNetV1](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz)
        * [MobileNetV2](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV2.tar.gz)
        * [MobileNetV3_large_x1_0](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_large_x1_0.tar.gz)
        * [MobileNetV3_small_x1_0](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_small_x1_0.tar.gz)
        * [ResNet50](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/ResNet50.tar.gz)
        * [SSD_MobileNetV3_large](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/ssdlite_mobilenet_v3_large.tar.gz)
        * [HRNet_w18](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/HRNet_18_voc.tar.gz)

    * int8 量化模型
        * [MobileNetV1_quant](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1_quant.tar.gz)
        * [MobileNetV2_quant](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV2_quant.tar.gz)
        * [MobileNetV3_large_x1_0_quant](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_large_x1_0_quant.tar.gz)
        * [MobileNetV3_small_x1_0_quant](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_small_x1_0_quant.tar.gz)
        * [ResNet50_quant](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/ResNet50_quant.tar.gz)
        * [SSD_MobileNetV3_large_quant](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/SSD_MobileNetV3_large_quant.tar.gz)
        * [HRNet_w18_quant](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/HRNet_18_voc_quant.tar.gz)

* 测试机器
   *  骁龙 865
      * Xiaomi MI10, Snapdragon 865 (enable sdot instruction)
      * CPU: 1xA77 @2.84GHz + 3xA77 @2.42GHz + 4xA55 @1.8GHz
      * GPU: Adreno 650

   *  骁龙 855
      * Xiaomi MI9, Snapdragon 855 (enable sdot instruction)
      * CPU: 1xA76 @2.84GHz + 3xA76 @2.42GHz + 4xA55 @1.78GHz
      * GPU: Adreno 640

   *  骁龙 845
      * Xiaomi MI8, Snapdragon 845
      * CPU: 4xA75 @2.8GHz + 4xA75 @1.7GHz
      * GPU: Adreno 630

   *  骁龙 835
      * Xiaomi mi6, Snapdragon 835
      * CPU: 4xA73 @2.45GHz + 4xA53 @1.9GHz
      * GPU: Adreno 540

   *  骁龙 625
      * Xiaomi Redmi6 Pro, Snapdragon 625
      * CPU: 4xA53 @1.8GHz + 4xA53 @1.6GHz
      * GPU: Adreno 506

   *  麒麟 990
      * Huawei Mate 30, Kirin 990
      * CPU: 2xA76 @2.86GHz + 2xA76 @2.09GHz + 4xA55 @1.86GHz
      * GPU: 16 core Mali-G76

   *  麒麟 980
      * Huawei Mate 20, Kirin 980
      * CPU: 2xA76 @2.6GHz + 2xA76 @1.92Ghz + 4xA55 @1.8Ghz
      * GPU: 10 core Mali-G76

   *  RK3399
      * CPU: 2xA72 @1.8GHz + 4xA53 @1.4Ghz
      * GPU: 4 core Mali-T860

* 测试说明
    * Branch: release/v2.10, commit id: b2e9776
    * 使用 Android ndk-r20b，armv8 编译
    * CPU 线程数设为 1，绑定大核
    * 在 GPU 上运行时，开启了 Auto Tune
    * warmup=20, repeats=600，统计平均时间，单位 ms
    * 输入数据全部设为 1.f
## 测试数据

### fp32 浮点模型测试数据

#### CPU 数据
运行时精度为 fp32 的性能数据如下：

|模型|骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|
|MobileNetV1|28.52 |29.22 |60.78 |82.54 |144.20 |38.16 |32.86 |111.76 |
|MobileNetV2|18.84 |23.17 |42.74 |56.23 |107.66 |24.91 |22.51 |79.95 |
|MobileNetV3_large_x1_0|14.55 |18.39 |32.49 |41.95 |96.30 |19.46 |17.78 |71.17 |
|MobileNetV3_small_x1_0|4.75 |6.41 |9.98 |13.98 |37.99 |6.50 |6.00 |23.34 |
|ResNet50|162.40 |192.88 |430.72 |490.54 |842.96 |221.81 |191.14 |638.29 |
|SSD_MobileNetV3_large|33.87 |42.84 |84.70 |103.32 |199.60 |46.02 |40.95 |157.08 |
|HRNet_w18|640.62 |835.62 |1687.78 |2048.81 |4724.20 |910.09 |820.42 |3380.08 |


运行时精度为 fp16 的性能数据如下：

|模型|骁龙 865|骁龙 855|骁龙 845|麒麟 990|
|:----|----:|----:|----:|----:|
|MobileNetV1|14.83 |15.79 |29.62 |20.64 |
|MobileNetV2|9.49 |10.28 |18.93 |12.29 |
|MobileNetV3_large_x1_0|7.84 |8.29 |16.00 |9.75 |
|MobileNetV3_small_x1_0|2.58 |3.03 |5.85 |3.47 |
|ResNet50|84.06 |87.10 |179.46 |109.38 |
|SSD_MobileNetV3_large|18.32 |19.99 |40.13 |24.37 |
|HRNet_w18|388.43 |430.27 |954.59 |544.75 |


#### GPU 数据

|模型|骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|
|MobileNetV1|7.05 |8.85 |10.46 |10.87 |71.42 |8.15 |13.74 |45.91 |
|MobileNetV2|9.48 |9.70 |8.58 |14.14 |52.09 |9.32 |13.08 |37.27 |
|MobileNetV3_large_x1_0|8.90 |9.11 |10.20 |12.04 |46.48 |9.81 |15.19 |32.92 |
|MobileNetV3_small_x1_0|5.79 |5.54 |8.52 |11.43 |20.00 |6.45 |8.71 |18.42 |
|ResNet50|29.70 |35.46 |45.23 |53.66 |392.62 |36.15 |54.23 |238.12 |
|SSD_MobileNetV3_large|27.69 |35.21 |33.37 |42.31 |152.37 |27.25 |35.79 |90.37 |


### int8 量化模型测试数据

#### CPU 数据

|模型|骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|
|MobileNetV1_quant|11.34 |14.81 |52.34 |55.69 |118.76 |14.80 |13.83 |78.30 |
|MobileNetV2_quant|10.55 |14.06 |33.99 |40.87 |85.81 |14.22 |13.06 |57.94 |
|MobileNetV3_large_x1_0_quant|8.11 |10.76 |24.63 |31.30 |70.86 |10.52 |9.73 |48.36 |
|MobileNetV3_small_x1_0_quant|3.04 |4.20 |8.93 |11.27 |25.13 |4.10 |3.75 |17.87 |
|ResNet50_quant|64.60 |80.46 |313.63 |331.30 |691.06 |81.65 |74.68 |489.30 |
|SSD_MobileNetV3_large_quant|20.84 |22.82 |64.16 |74.12 |165.91 |27.11 |25.29 |119.92 |


## 华为昇腾 NPU 的性能数据
请参考 [Paddle Lite 使用华为昇腾 NPU 预测部署](../demo_guides/huawei_ascend_npu)的最新性能数据

## 华为麒麟 NPU 的性能数据
请参考 [Paddle Lite 使用华为麒麟 NPU 预测部署](../demo_guides/huawei_kirin_npu)的最新性能数据

## 瑞芯微 NPU 的性能数据
请参考 [Paddle Lite 使用瑞芯微 NPU 预测部署](../demo_guides/rockchip_npu)的最新性能数据

## 晶晨 NPU 的性能数据
请参考 [Paddle Lite 使用晶晨NPU 预测部署](../demo_guides/amlogic_npu)的最新性能数据

## 联发科 APU 的性能数据
请参考 [Paddle Lite 使用联发科 APU 预测部署](../demo_guides/mediatek_apu)的最新性能数据

## 颖脉 NNA 的性能数据
请参考 [Paddle Lite 使用颖脉 NNA 预测部署](../demo_guides/imagination_nna)的最新性能数据
