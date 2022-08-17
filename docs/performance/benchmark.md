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
    
    * fp32 稀疏化模型
        * [MobileNet](http://10.127.28.26:8879/bridge/sparse.tar)
        * [humanseg](http://10.127.28.26:8879/projects_sparse/humanseg.tar)
        * [picodet](http://10.127.28.26:8879/bridge/picodet_m_320_coco_75.tar)

* 测试机器
   ||骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
   |:----|----:|----:|----:|----:|----:|----:|----:|----:|
   |设备|Xiaomi MI10 |Xiaomi MI9 |Xiaomi MI8 |Xiaomi mi6 |Xiaomi Redmi6 Pro |Huawei Mate 30 |Huawei Mate 20 |瑞芯微RK3399开发板 |
   |CPU|1xA77 @2.84GHz + 3xA77 @2.42GHz + 4xA55 @1.8GHz |1xA76 @2.84GHz + 3xA76 @2.42GHz + 4xA55 @1.78GHz |4xA75 @2.8GHz + 4xA75 @1.7GHz |4xA73 @2.45GHz + 4xA53 @1.9GHz |4xA53 @1.8GHz + 4xA53 @1.6GHz |2xA76 @2.86GHz + 2xA76 @2.09GHz + 4xA55 @1.86GHz |2xA76 @2.6GHz + 2xA76 @1.92Ghz + 4xA55 @1.8Ghz |2xA72 @1.8GHz + 4xA53 @1.4Ghz | 
   |GPU|Adreno 650 |Adreno 640 |Adreno 630 |Adreno 540 |Adreno 506 |16 core Mali-G76 |10 core Mali-G76 |4 core Mali-T860 | 

* 测试说明
    * Branch: release/v2.11, commit id: 4a3bdbe
    * 使用 Android ndk-r22b，armv7 armv8 编译
    * CPU 线程数设为 1，绑定大核
    * 在 GPU 上运行时，开启了 Auto Tune
    * warmup=20, repeats=600，统计平均时间，单位 ms
    * 输入数据全部设为 1.f
## 测试数据

### fp32 浮点模型测试数据

#### ARMV8 CPU 数据
运行时精度为 fp32 的性能数据如下：

* 稠密模型的性能数据如下：

|模型|骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|
|MobileNetV1|28.38 |29.04 |58.70 |82.48 |143.64 |38.22 |32.50 |103.39 |
|MobileNetV2|18.52 |19.37 |37.61 |53.47 |106.17 |24.93 |22.07 |73.37 |
|MobileNetV3_large_x1_0|14.46 |15.59 |29.55 |41.74 |96.11 |19.38 |17.46 |63.34 |
|MobileNetV3_small_x1_0|4.73 |5.39 |9.76 |13.45 |39.08 |6.41 |5.85 |21.67 |
|ResNet50|160.97 |161.13 |339.59 |484.79 |831.62 |222.21 |190.38 |616.68 |
|SSD_MobileNetV3_large|33.62 |36.07 |69.76 |99.88 |193.79 |46.95 |40.87 |153.94|
|HRNet_w18|645.30 |694.41 |1395.66 |2063.99 |4717.07 |905.07 |792.17 |3491.22 |
|humanseg|23.10 |24.69 |50.54 |65.09 |317.87 |31.76 |41.67 |110.08 |
|picodet|41.32 |42.48 |101.18 |121.82 |431.46 |56.40 |73.70 |185.36 |

* 稀疏模型的性能数据如下：

|模型|骁龙 865|骁龙 835|骁龙 625|
|:----|----:|----:|----:|
|MobileNetV1|17.04 |56.02 |173.46 |
|MobileNetV2|12.59 |38.53 |117.95 |
|MobileNetV3|9.61 |29.67 |93.34 |
|humanseg|17.04 |50.34 |287.47 |
|picodet|25.79 |75.79 |452.29 |

运行时精度为 fp16 的性能数据如下：

|模型|骁龙 865|骁龙 855|骁龙 845|麒麟 990|
|:----|----:|----:|----:|----:|
|MobileNetV1|14.83 |15.79 |29.62 |20.69 |
|MobileNetV2|9.31 |10.29 |19.10 |12.34 |
|MobileNetV3_large_x1_0|7.63 |8.15 |15.77 |9.63 |
|MobileNetV3_small_x1_0|2.48 |2.90 |5.64 |3.36 |
|ResNet50|82.00 |83.87 |166.00 |106.65 |
|SSD_MobileNetV3_large|17.78 |19.40 |36.29 |23.62 |
|HRNet_w18|377.88 |418.99 |812.44 |529.34 |

#### ARMV7 CPU 数据
运行时精度为 fp32 的性能数据如下：

* 稠密模型的性能数据如下：

|模型|骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|
|MobileNetV1|31.21 |33.44 |68.04 |93.72 |147.85 |41.18 |35.49 |116.44 |
|MobileNetV2|21.47 |25.79 |46.19 |64.54 |131.05 |28.18 |25.08 |84.09 |
|MobileNetV3_large_x1_0|16.81 |19.47 |34.80 |48.61 |102.01 |21.96 |19.75 |66.28 |
|MobileNetV3_small_x1_0|5.44 |6.48 |11.25 |15.44 |37.74 |7.28 |6.73 |21.01 |
|ResNet50|177.80 |188.07 |377.67 |519.61 |886.98 |237.47 |203.60 |657.27 |
|SSD_MobileNetV3_large|38.02 |45.21 |82.34 |115.40 |210.73 |52.26 |46.02 |156.85 |
|HRNet_w18|733.96 |839.54 |1532.35 |2200.64 |5029.30 |989.71 |869.29 |3256.20 |
|humanseg|24.68 |27.55 |78.15 |73.19 |304.20 |35.11 |45.69 |144.68 |
|picodet|44.90 |46.78 |112.13 |131.64 |626.55 |61.54 |81.52 |234.02 |

* 稀疏模型的性能数据如下：

|模型|骁龙 865|骁龙 835|骁龙 625|
|:----|----:|----:|----:|
|MobileNetV1|19.47 |61.62 |179.51 |
|MobileNetV2|13.98 |41.59 |120.51 |
|MobileNetV3|11.06 |31.43 |94.93 |
|humanseg|18.95 |57.14 |276.18 |
|picodet|29.27 |81.56 |491.77 |


运行时精度为 fp16 的性能数据如下：

|模型|骁龙 865|骁龙 855|骁龙 845|麒麟 990|
|:----|----:|----:|----:|----:|
|MobileNetV1|15.81 |16.39 |58.72 |21.04 |
|MobileNetV2|10.59 |11.55 |25.28 |13.96 |
|MobileNetV3_large_x1_0|8.69 |9.50 |20.01 |11.00 |
|MobileNetV3_small_x1_0|3.08 |3.55 |6.77 |4.00 |
|ResNet50|86.62 |89.15 |289.57 |110.00 |
|SSD_MobileNetV3_large|20.03 |22.12 |48.93 |27.03 |
|HRNet_w18|481.79 |520.02 |1032.48 |650.79 |


#### GPU 数据

|模型|骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|
|MobileNetV1|6.39 |6.41 |8.26 |10.13 |56.04 |8.18 |15.28 |45.94 |
|MobileNetV2|8.96 |8.07 |7.96 |11.49 |43.65 |9.07 |15.83 |37.67 |
|MobileNetV3_large_x1_0|9.89 |9.30 |11.29 |16.64 |40.58 |9.43 |14.89 |32.57 |
|MobileNetV3_small_x1_0|8.10 |6.79 |10.13 |10.37 |19.12 |6.59 |7.69 |18.63 |
|ResNet50|25.22 |29.70 |39.11 |47.65 |337.10 |36.90 |55.54 |237.41 |
|SSD_MobileNetV3_large|22.68 |30.21 |26.58 |26.53 |129.34 |25.14 |32.64 |102.15 |


### int8 量化模型测试数据

#### ARMV8 CPU 数据

|模型|骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|
|MobileNetV1_quant|11.18 |11.86 |44.12 |55.57 |117.41 |14.75 |13.58 |77.60 |
|MobileNetV2_quant|10.37 |11.38 |33.45 |40.78 |83.47 |14.14 |12.94 |56.66 |
|MobileNetV3_large_x1_0_quant|8.03 |8.88 |24.32 |30.57 |66.76 |10.41 |9.57 |46.66 |
|MobileNetV3_small_x1_0_quant|2.89 |3.29 |8.41 |10.86 |22.83 |4.23 |3.90 |15.83 |
|ResNet50_quant|64.50 |66.81 |263.17 |331.76 |690.01 |82.07 |74.52 |473.00 |
|SSD_MobileNetV3_large_quant|20.43 |22.71 |57.45 |73.97 |164.65 |27.26 |25.08 |114.71 |

#### ARMV7 CPU 数据

|模型|骁龙 865|骁龙 855|骁龙 845|骁龙 835|骁龙 625|麒麟 990|麒麟 980|RK3399|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|
|MobileNetV1_quant|15.26 |16.11 |53.71 |63.43 |129.72 |19.86 |17.97 |85.89 |
|MobileNetV2_quant|16.44 |17.96 |43.67 |53.32 |105.39 |22.08 |19.90 |72.77 |
|MobileNetV3_large_x1_0_quant|11.20 |12.50 |30.72 |37.52 |76.98 |15.19 |13.99 |51.93 |
|MobileNetV3_small_x1_0_quant|4.27 |4.82 |10.58 |13.23 |26.86 |5.98 |5.51 |18.55 |
|ResNet50_quant|75.82 |83.01 |301.21 |360.67 |763.98 |96.03 |85.56 |504.01 |
|SSD_MobileNetV3_large_quant|27.64 |31.40 |73.19 |92.01 |194.78 |36.87 |33.25 |131.47 |

## 华为昇腾 NPU 的性能数据
请参考 [Paddle Lite 使用华为昇腾 NPU 预测部署](../demo_guides/huawei_ascend_npu)的最新性能数据

## 华为麒麟 NPU 的性能数据
请参考 [Paddle Lite 使用华为麒麟 NPU 预测部署](../demo_guides/huawei_kirin_npu)的最新性能数据

## 瑞芯微 NPU 的性能数据
请参考 [Paddle Lite 使用瑞芯微 NPU 预测部署](../demo_guides/rockchip_npu)的最新性能数据

## 晶晨 NPU 的性能数据
请参考 [Paddle Lite 使用晶晨 NPU 预测部署](../demo_guides/amlogic_npu)的最新性能数据

## 芯原 TIM-VX 的性能数据
请参考 [Paddle Lite 使用芯原 TIM-VX 预测部署](../demo_guides/verisilicon_timvx)的最新性能数据

## Android NNAPI 的性能数据
请参考 [Paddle Lite 使用 Android NNAPI 预测部署](../demo_guides/android_nnapi)的最新性能数据

## 联发科 APU 的性能数据
请参考 [Paddle Lite 使用联发科 APU 预测部署](../demo_guides/mediatek_apu)的最新性能数据

## 颖脉 NNA 的性能数据
请参考 [Paddle Lite 使用颖脉 NNA 预测部署](../demo_guides/imagination_nna)的最新性能数据

## 英特尔 OpenVINO 的性能数据
请参考 [Paddle Lite 使用英特尔 OpenVINO 预测部署](../demo_guides/intel_openvino)的最新性能数据

## 亿智 NPU 的性能数据
请参考 [Paddle Lite 使用亿智 NPU 预测部署](../demo_guides/eeasytech_npu)的最新性能数据
