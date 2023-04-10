# Arm

Paddle Lite 支持在 Android/iOS/ARMLinux 等移动端设备上运行高性能的 CPU 预测库，目前支持 Ubuntu 环境下 armv8、armv7 的交叉编译。

## 支持现状

### 已支持的芯片

- 高通 888+/888/Gen1/Gen2/875/865/855/845/835/625/8155/8295P 等
- 麒麟 810/820/985/990/990 5G/9000E/9000 等

### 已支持的设备

- HUAWEI Mate 30/40/50 系列，荣耀 V20 系列，nova 6 系列，P40/P50/P60 系列，Mate Xs
- HUAWEI nova 5 系列，nova 6 SE，荣耀 9X 系列，荣耀 Play4T Pro
- 小米 6，小米 8，小米 10，小米 12，小米 13, 小米 MIX2，红米 10X，红米 Note8pro
- 高通 8295 EVK

### 已验证支持的 Paddle 模型

#### 模型
- 图像分类
  - [DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DarkNet/DarkNet53.tar.gz)
  - [DenseNet121](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz)
  - [DPN68](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DPN/DPN68.tar.gz)
  - [EfficientNetB0](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/EfficientNetB0.tgz)
  - [GhostNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/GhostNet/GhostNet_x1_0.tar.gz)
  - [GoogLeNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz)
  - [HRNet-W18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/HRNet/HRNet_W18_C.tar.gz)
  - [Inception-v3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz)
  - [Inception-v4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz)
  - [MobileNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV1.tgz)
  - [MobileNet-v2](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV2.tgz)
  - [MobileNetV3_large](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_large_x1_0.tgz)
  - [MobileNetV3_small](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_small_x1_0.tgz)
  - [PP-LCNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/PPLCNet/PPLCNet_x0_25.tar.gz)
  - [Res2Net50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/Res2Net/Res2Net50_26w_4s.tar.gz)
  - [ResNet-101](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz)
  - [ResNet-18](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet18.tgz)
  - [ResNet-50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz)
  - [ResNeXt50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz)
  - [SE_ResNet50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/SENet/SE_ResNet50_vd.tar.gz)
  - [SqueezeNet-v1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz)
- 目标检测
  - [PPYOLO_tiny](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_tiny_650e_coco.tar.gz)
  - [ssd_mobilenet_v1_relu_voc_fp32_300](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz)
  - [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz)
  - [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v3_large_270e_coco.tgz)
  - [YOLOv3-ResNet50_vd](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz)
- 姿态检测
  - [PP-TinyPose](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/tinypose_128x96.tar.gz)
- 关键点检测
  - [HigherHRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/higherhrnet_hrnet_w32_640.tgz)
  - [HRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/hrnet_w32_384x288.tgz)
- 文本检测 & 文本识别 & 端到端检测识别
  - [ch_ppocr_mobile_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_det_infer.tgz)
  - [ch_ppocr_mobile_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_rec_infer.tgz)
  - [ch_ppocr_server_v2.0_det](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz)
  - [ch_ppocr_server_v2.0_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz)
  - [ch_PP-OCRv2_rec](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz)
  - [CRNN-mv3-CTC](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz)
- 生成网络
  - [ESRGAN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleGAN/esrgan_psnr_x4_div2k.tgz)
- 视频分类
  - [PP-TSN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz)


#### 性能数据

请参考[性能测试文档](benchmark_tools)对模型进行测试。

##### 测试环境

* 模型
    * fp32 浮点模型
        * [AlexNet](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/AlexNet.tar.gz)
        * [OCRv2Det](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ch_PP-OCRv2_det_infer.tar.gz)
        * [OCRv2Rec](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ch_PP-OCRv2_rec_infer.tar.gz)
        * [OCRv3Det](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ch_PP-OCRv3_det_infer.tar.gz)
        * [OCRv3Rec](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ch_PP-OCRv3_rec_infer.tar.gz)
        * [OcrMobileV20Det](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ch_ppocr_mobile_v2.0_det_infer.tar.gz)
        * [OcrMobileV20Rec](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ch_ppocr_mobile_v2.0_rec_infer.tar.gz)
        * [EfficientNet](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/efficientnet.tar.gz)
        * [FaceDetector](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/FaceDetector.tar.gz)
        * [HRNetW18C](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/hrnet_18_voc.tar.gz)
        * [InceptionV1](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/inception_v1.tar.gz)
        * [InceptionV2](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/inception_v2.tar.gz)
        * [Mobilefacenet](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/mobilefacenet.tar.gz)
        * [MobileNetV1](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/MobileNetV1_infer.tar.gz)
        * [MobileNetV2](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/MobileNetV2_infer.tar.gz)
        * [MobileNetV3Large](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/MobileNetV3_large_x1_0_infer.tar.gz)
        * [MobileNetV3Small](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/MobileNetV3_small_x1_0_infer.tar.gz)
        * [PicodetS320Coco](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/picodet_s_320_COCO.tar.gz)
        * [PicodetS320CocoLcnetNonPostprocess](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/picodet_s_320_coco_lcnet_non_postprocess.tar.gz)
        * [PPLCNetV2Base](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/PPLCNetV2_base_infer.tar.gz)
        * [PPLCNetX10](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/PPLCNet_x1_0_infer.tar.gz)
        * [PpyoloTiny650eCoco](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ppyolo_tiny_650e_coco.tar.gz)
        * [HRNetW18-Seg](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/RES-paddle2-HRNetW18-Seg.tar.gz)
        * [HumanSegLite](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/RES-paddle2-PPHumanSegLite.tar.gz)
        * [LIteSegSTDC1](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/RES-paddle2-PPLIteSegSTDC1.tar.gz)
        * [ResNet18](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ResNet18_infer.tar.gz)
        * [ResNet50](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ResNet50_infer.tar.gz)
        * [ShuffleNetV2](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ShuffleNetV2_x1_0_infer.tar.gz)
        * [SqueezeNet](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/SqueezeNet1_0_infer.tar.gz)
        * [SsdMobilenetv1](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/ssd_mobilenetv1.tar.gz)
        * [Tinypose](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/tinypose_128x96.tar.gz)
        * [Transformer](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/transformer.tar.gz)
        * [Vgg16](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/VGG16_infer.tar.gz)
        * [YoloV3](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/yolov3.tar.gz)
        * [Yolov5s](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpufp32/yolov5s.tar.gz)

    * int8 量化模型
        * [DeeplabV3](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/deeplabv3_quant.tar.gz)
        * [EfficientNet](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/EfficientNetB0_quant.tar.gz)
        * [Hrnet18](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/hrnet18_quant.tar.gz)
        * [ERNIE 3.0-Medium](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/medium_quant.tar.gz)
        * [ERNIE 3.0-Micro](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/micro_quant.tar.gz)
        * [ERNIE 3.0-Mini](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/mini_quant.tar.gz)
        * [MobileNetV3](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/MobileNetV3_large_x1_0_quant.tar.gz)
        * [ERNIE 3.0-Nano](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/nano_quant.tar.gz)
        * [Picodet](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/picodet_s_416_coco_npu_quant.tar.gz)
        * [PPHGNet](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/PPHGNet_tiny_quant.tar.gz)
        * [PPLCNetV2](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/PPLCNetV2_base_quant.tar.gz)
        * [Ppliteseg](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/pp_liteseg_quant.tar.gz)
        * [Ppseghumanportrait](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/ppseg_human_portrait_quant.tar.gz)
        * [ResNet50Vd](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/ResNet50_vd_quant.tar.gz)
        * [Unet](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/unet_quant.tar.gz)
        * [Yolov5s](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/yolov5s_quant.tar.gz)
        * [Yolov6s](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/yolov6s_quant.tar.gz)
        * [Yolov7](https://paddlelite-demo.bj.bcebos.com/models/benchmark/armcpuint8/yolov7_quant.tar.gz)


* 测试机器

   |SOC|骁龙 865|骁龙 835|骁龙 625|RK3399|
   |:----|----:|----:|----:|----:|
   |设备|Xiaomi MI10 |Xiaomi mi6 |Xiaomi Redmi6 Pro |瑞芯微RK3399开发板 |
   |CPU|1xA77 @2.84GHz + 3xA77 @2.42GHz + 4xA55 @1.8GHz |4xA73 @2.45GHz + 4xA53 @1.9GHz |4xA53 @1.8GHz + 4xA53 @1.6GHz |2xA72 @1.8GHz + 4xA53 @1.4Ghz | 
   |GPU|Adreno 650 |Adreno 540 |Adreno 506 |4 core Mali-T860 |

* 测试说明
    * Branch: release/v2.13, commit id: 102697b
    * 使用 Android ndk-r22b armv7 armv8 编译
    * CPU 线程数设为 1，绑定大核
    * warmup=10, repeats=30，统计平均时间，单位 ms
    * 输入数据全部设为 1.f
##### 测试数据
###### ARMV8 CPU fp32 浮点模型测试数据

|模型|骁龙 865|骁龙 835|骁龙 625|RK3399|
|:----|----:|----:|----:|----:|
|AlexNet|45.298|122.155|770.008|197.162|
|OCRv2Det|222.664|637.484|1406.197|1093.946|
|OCRv2Rec|12.57|33.971|59.811|44.492|
|OCRv3Det|227.379|646.856|1474.094|1098.47|
|OCRv3Rec|45.114|131.685|223.771|168.793|
|OcrMobileV20Det|96.007|286.668|628.396|481.654|
|OcrMobileV20Rec|5.146|14.839|27.769|19.802|
|EfficientNet|39.976|98.591|181.419|134.529|
|FaceDetector|42.338|130.151|282.142|205.373|
|HRNetW18C|700.674|2101.928|4748.64|3262.984|
|InceptionV1|61.297|165.061|348.055|229.09|
|InceptionV2|83.81|225.911|476.587|328.352|
|Mobilefacenet|13.322|36.752|62.819|49.384|
|MobileNetV1|30.146|85.85|148.344|105.376|
|MobileNetV2|19.981|55.87|112.664|76.301|
|MobileNetV3Large|15.593|42.95|97.458|63.221|
|MobileNetV3Small|5.219|14.461|39.277|21.807|
|PicodetS320Coco|23.292|63.198|117.983|87.032|
|PicodetS320CocoLcnetNonPostprocess|37.311|110.443|195.804|145.417|
|PPLCNetV2Base|33.243|93.989|174.101|121.822|
|PPLCNetX10|10.496|28.814|63.258|40.397|
|PpyoloTiny650eCoco|21.742|60.907|117.07|87.344|
|HRNetW18-Seg|718.029|2148.141|4889.799|3462.36|
|PPHumanSegLite|25.287|69.826|164.99|93.571|
|PPLIteSegSTDC1|207.527|588.422|1461.373|1266.579|
|ResNet18|65.761|179.595|420.178|294.354|
|ResNet50|183.636|508.689|959.749|736.446|
|ShuffleNetV2|9.463|25.096|50.671|33.286|
|SqueezeNet|16.303|45.7|100.08|66.814|
|SsdMobilenetv1|60.46|172.99|279.734|217.854|
|Tinypose|7.164|20.095|39.186|26.982|
|Transformer|80.073|217.179|347.02|265.595|
|Vgg16|336.728|940.533|3169.282|1739.123|
|YoloV3|1204.35|3149.709|9252.66|6004.355|
|Yolov5s|445.271|1187.192|2254.931|1688.49|

###### ARMV8 CPU fp16 浮点模型测试数据

|模型|骁龙 865|
|:----|----:|
|AlexNet|21.066|
|OCRv2Det|114.004|
|OCRv2Rec|6.841|
|OCRv3Det|118.885|
|OCRv3Rec|20.143|
|OcrMobileV20Det|49.925|
|OcrMobileV20Rec|2.816|
|EfficientNet|19.915|
|FaceDetector|21.332|
|HRNetW18C|370.469|
|InceptionV1|27.15|
|InceptionV2|37.731|
|Mobilefacenet|6.45|
|MobileNetV1|15.481|
|MobileNetV2|9.761|
|MobileNetV3Large|7.836|
|MobileNetV3Small|2.661|
|PicodetS320Coco|12.22|
|PicodetS320CocoLcnetNonPostprocess|19.115|
|PPLCNetV2Base|17.326|
|PPLCNetX10|5.297|
|PpyoloTiny650eCoco|14.345|
|HRNetW18-Seg|383.853|
|PPHumanSegLite|13.17|
|PPLIteSegSTDC1|94.311|
|ResNet18|26.404|
|ResNet50|84.92|
|ShuffleNetV2|4.72|
|SqueezeNet|7.779|
|SsdMobilenetv1|31.446|
|Tinypose|4.44|
|Transformer|44.718|
|Vgg16|146.505|
|YoloV3|587.59|
|Yolov5s|223.163|

###### ARMV7 CPU fp32 浮点模型测试数据

|模型|骁龙 865|骁龙 835|骁龙 625|RK3399|
|:----|----:|----:|----:|----:|
|AlexNet|103.301|193.101|532.866|255.368|
|OCRv2Det|234.603|694.595|1591.211|1149.416|
|OCRv2Rec|15.067|37.47|85.617|47.727|
|OCRv3Det|244.443|719.882|1665.616|1170.136|
|OCRv3Rec|52.453|142.195|314.562|184.315|
|OcrMobileV20Det|108.251|313.331|723.745|497.859|
|OcrMobileV20Rec|6.42|15.806|34.612|21.528|
|EfficientNet|54.905|148.798|293.059|205.277|
|FaceDetector|48.584|139.936|338.272|214.22|
|HRNetW18C|928.904|2429.57|5597.829|3626.208|
|InceptionV1|116.245|230.417|459.705|317.239|
|InceptionV2|167.407|326.638|643.686|464.131|
|Mobilefacenet|14.911|40|75.236|56.348|
|MobileNetV1|32.821|93.959|169.211|117.964|
|MobileNetV2|23.214|64.928|138.528|90.181|
|MobileNetV3Large|17.815|48.379|109.924|68.397|
|MobileNetV3Small|6.131|15.864|40.71|22.2|
|PicodetS320Coco|25.602|70.111|162.348|97.302|
|PicodetS320CocoLcnetNonPostprocess|41.794|121.076|261.925|161.35|
|PPLCNetV2Base|36.283|102.993|192.899|130.282|
|PPLCNetX10|12.053|32.736|80.032|43.544|
|PpyoloTiny650eCoco|27.795|67.925|147.947|95.258|
|HRNetW18-Seg|954.581|2461.85|5803.546|3702.918|
|PPHumanSegLite|29.138|74.169|161.762|106.454|
|PPLIteSegSTDC1|349.211|770.709|1823.658|1358.558|
|ResNet18|109.924|239.802|531.152|329.986|
|ResNet50|257.906|615.185|1176.348|824.02|
|ShuffleNetV2|10.807|27.998|54.058|36.379|
|SqueezeNet|28.749|61.456|129.137|87.713|
|SsdMobilenetv1|65.775|186.446|336.495|233.838|
|Tinypose|8.952|22.045|48.234|31.114|
|Transformer|82.289|236.965|411.176|270.432|
|Vgg16|530.045|1220.56|3142.593|1893.435|
|YoloV3|2155.629|4222.444|11587.91|6440.611|
|Yolov5s|515.687|1437.948|3074.772|2025.427|

###### ARMV7 CPU fp16 浮点模型测试数据

|模型|骁龙 865|
|:----|----:|
|AlexNet|23.036|
|OCRv2Det|138.617|
|OCRv2Rec|7.88|
|OCRv3Det|144.002|
|OCRv3Rec|23.069|
|OcrMobileV20Det|60.073|
|OcrMobileV20Rec|3.619|
|EfficientNet|45.1|
|FaceDetector|25.174|
|HRNetW18C|459.19|
|InceptionV1|30.241|
|InceptionV2|41.626|
|Mobilefacenet|7.068|
|MobileNetV1|16.283|
|MobileNetV2|11.047|
|MobileNetV3Large|9.013|
|MobileNetV3Small|3.237|
|PicodetS320Coco|14.238|
|PicodetS320CocoLcnetNonPostprocess|23.54|
|PPLCNetV2Base|19.369|
|PPLCNetX10|6.035|
|PpyoloTiny650eCoco|20.154|
|HRNetW18-Seg|475.219|
|PPHumanSegLite|15.454|
|PPLIteSegSTDC1|112.679|
|ResNet18|29.385|
|ResNet50|89.97|
|ShuffleNetV2|5.229|
|SqueezeNet|9.084|
|SsdMobilenetv1|32.736|
|Tinypose|5.143|
|Transformer|46.836|
|Vgg16|161.807|
|YoloV3|666.677|
|Yolov5s|339.294|





###### ARMV8 CPU int8 量化模型测试数据

|模型|骁龙 865|
|:----|----:|
|DeeplabV3|476.312|
|EfficientNet|39.345|
|Hrnet18|99.822|
|ERNIE 3.0-Medium|94.506|
|ERNIE 3.0-Micro|23.444|
|ERNIE 3.0-Mini|35.075|
|MobileNetV3|8.481|
|ERNIE 3.0-Nano|18.38|
|Picodet|30.397|
|PPHGNet|71.05|
|PPLCNetV2|13.743|
|Ppliteseg|29.396|
|Ppseghumanportrait|17.923|
|ResNet50Vd|70.115|
|Unet|387.707|
|Yolov5s|223.629|
|Yolov6s|330.521|
|Yolov7|1064.985|

###### ARMV7 CPU int8 量化模型测试数据

|模型|骁龙 865|
|:----|----:|
|DeeplabV3|551.74|
|EfficientNet|76.063|
|Hrnet18|126.647|
|ERNIE 3.0-Medium|116.793|
|ERNIE 3.0-Micro|31.18|
|ERNIE 3.0-Mini|46.336|
|MobileNetV3|11.658|
|ERNIE 3.0-Nano|24.933|
|Picodet|42.257|
|PPHGNet|85.009|
|PPLCNetV2|17.912|
|Ppliteseg|41.255|
|Ppseghumanportrait|23.775|
|ResNet50Vd|84.646|
|Unet|473.276|
|Yolov5s|297.14|
|Yolov6s|424.786|
|Yolov7|1317.209|







## 参考示例演示

### 测试设备
- Android arm64-v8a/armeabi-v7a: HUAWEI P40pro
- Linux arm64: RK3399
- Linux armhf: Raspberry Pi 4B

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考 [Docker 统一编译环境搭建](../source_compile/docker_env) 中的 Docker 开发环境进行配置。

### 运行图像分类示例程序

- 下载 Paddle Lite 通用示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - configs
            - imagenet_224.txt # config 文件
            - synset_words.txt # 1000 分类 label 文件
          - datasets
            - test # dataset
              - inputs
                - tabby_cat.jpg # 输入图片
              - outputs
                - tabby_cat.jpg # 输出图片
              - list.txt # 图片清单
          - models
            - resnet50_fp32_224 # Paddle non-combined 格式的 resnet50 float32 模型
              - __model__ # Paddle fluid 模型组网文件，可拖入 https://lutzroeder.github.io/netron/ 进行可视化显示网络结构
              - bn2a_branch1_mean # Paddle fluid 模型参数文件
              - bn2a_branch1_scale
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.linux.amd64 # 已编译好的，适用于 amd64
            - demo # 已编译好的，适用于 amd64 的示例程序
          - build.linux.arm64 # 已编译好的，适用于 arm64
            - demo # 已编译好的，适用于 arm64 的示例程序
            ...
          ...
          - demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序 ssh 运行脚本
          - run_with_adb.sh # 示例程序 adb 运行脚本
      - libs
        - PaddleLite
          - android
            - arm64-v8a
            - armeabi-v7a
          - linux
            - arm64
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - armhf
              - include
              - lib
        - OpenCV # OpenCV 预编译库
      - object_detection_demo # 目标检测示例程序
  ```

- 进入 `PaddleLite-generic-demo/image_classification_demo/shell/`；

- 执行以下命令观察 mobilenet_v1_int8_224_per_layer 模型的性能和结果；

  ```shell
  运行 mobilenet_v1_int8_224_per_layer 模型

  For android arm64-v8a
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android arm64-v8a cpu <adb设备号>

    Top1 Egyptian cat - 0.503239
    Top2 tabby, tabby cat - 0.419854
    Top3 tiger cat - 0.065506
    Top4 lynx, catamount - 0.007992
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000494
    [0] Preprocess time: 6.712000 ms Prediction time: 16.859000 ms Postprocess time: 6.026000 ms
    Preprocess time: avg 6.712000 ms, max 6.712000 ms, min 6.712000 ms
    Prediction time: avg 16.859000 ms, max 16.859000 ms, min 16.859000 ms
    Postprocess time: avg 6.026000 ms, max 6.026000 ms, min 6.026000 ms

  For android armeabi-v7a
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android armeabi-v7a cpu <adb设备号>

    Top1 Egyptian cat - 0.502124
    Top2 tabby, tabby cat - 0.413927
    Top3 tiger cat - 0.071703
    Top4 lynx, catamount - 0.008436
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000563
    [0] Preprocess time: 6.717000 ms Prediction time: 44.779000 ms Postprocess time: 6.444000 ms
    Preprocess time: avg 6.717000 ms, max 6.717000 ms, min 6.717000 ms
    Prediction time: avg 44.779000 ms, max 44.779000 ms, min 44.779000 ms
    Postprocess time: avg 6.444000 ms, max 6.444000 ms, min 6.444000 ms

  For linux arm64
  本地执行
  $ ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 cpu
  通过 SSH 远程执行
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 cpu <IP地址> 22 <用户名> <密码>

    Top1 Egyptian cat - 0.503239
    Top2 tabby, tabby cat - 0.419854
    Top3 tiger cat - 0.065506
    Top4 lynx, catamount - 0.007992
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000494
    Preprocess time: 12.637000 ms, avg 12.637000 ms, max 12.637000 ms, min 12.637000 ms
    Prediction time: 78.751000 ms, avg 78.751000 ms, max 78.751000 ms, min 78.751000 ms
    Postprocess time: 9.969000 ms, avg 9.969000 ms, max 9.969000 ms, min 9.969000 ms

  For linux armhf
  本地执行
  $ ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux armhf cpu
  通过 SSH 远程执行
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux armhf cpu <IP地址> 22 <用户名> <密码>

    Top1 Egyptian cat - 0.502124
    Top2 tabby, tabby cat - 0.413927
    Top3 tiger cat - 0.071703
    Top4 lynx, catamount - 0.008436
    Top5 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor - 0.000563
    Preprocess time: 12.541000 ms, avg 12.541000 ms, max 12.541000 ms, min 12.541000 ms
    Prediction time: 96.863000 ms, avg 96.863000 ms, max 96.863000 ms, min 96.863000 ms
    Postprocess time: 13.324000 ms, avg 13.324000 ms, max 13.324000 ms, min 13.324000 ms
  ```

- 如果需要更改测试模型为 resnet50 ，执行命令修改为如下：

  ```shell
  For android arm64-v8a
  $ ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a cpu <adb设备号>

  For android armeabi-v7a
  $ ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android armeabi-v7a cpu <adb设备号>

  For linux arm64
  本地执行
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 cpu
  通过 SSH 远程执行
  $ ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 cpu <IP地址> 22 <用户名> <密码>

  For linux armhf
  本地执行
  $ ./run.sh resnet50_fp32_224 imagenet_224.txt test linux armhf cpu
  通过 SSH 远程执行
  $ ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux armhf cpu <IP地址> 22 <用户名> <密码>
  ```

- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/inputs` 目录下，同时将图片文件名添加到 `PaddleLite-generic-demo/image_classification_demo/assets/datasets/test/list.txt` 中；

- 如果需要重新编译示例程序，直接运行

  ```shell
  For android arm64-v8a
  $ ./build.sh android arm64-v8a

  For android armeabi-v7a
  $ ./build.sh android armeabi-v7a

  For linux arm64
  $ ./build.sh linux arm64
  
  For linux armhf
  $ ./build.sh linux armhf
  ```

### 更新支持 Arm 的 Paddle Lite 库

- 下载 Paddle Lite 源码

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ```

- 编译并生成 armv8 和 armv7 的部署库

  - For android arm64-v8a（注：--with_arm82_fp16=ON 编译选项可在部分机型启用 FP16 能力，但要求 NDK 版本 > 19 ）
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv8 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv8 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/arm64-v8a/lib/
      ```

  - For android armeabi-v7a（注：--with_arm82_fp16=ON 编译选项可在部分机型启用 FP16 能力，但要求 NDK 版本 > 19 ）
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --with_extra=ON --with_cv=ON --with_exception=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```

- 编译并生成 arm64 和 armhf 的部署库

  - For linux arm64
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_cv=ON --with_exception=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --with_cv=ON --with_exception=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

  - For linux armhf
    - tiny_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_cv=ON --with_exception=ON
      ```

    - full_publish 编译
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_cv=ON --with_exception=ON full_publish
      ```

    - 替换头文件和库
      ```shell
      替换 include 目录
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      
      替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      
      替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      ```

- 替换头文件后需要重新编译示例程序

## 高级特性

- 性能分析和精度分析

  android 平台下分析：

  - 开启性能分析，会打印出每个 op 耗时信息和汇总信息

  ```bash
  $ ./lite/tools/build.sh \
    --arm_os=android \
    --arm_abi=armv8 \
    --build_extra=on \
    --build_cv=on \
    --arm_lang=clang \
    --with_profile=ON \
  test
  ```

  - 开启精度分析，会打印出每个 op 输出数据的均值和标准差信息

  ```bash
  # 开启性能分析，会打印出每个 op 耗时信息和汇总信息
  $ ./lite/tools/build.sh \
    --arm_os=android \
    --arm_abi=armv8 \
    --build_extra=on \
    --build_cv=on \
    --arm_lang=clang \
    --with_profile=ON \
    --with_precision_profile=ON \
    test
  ```

  详细输出信息的说明可查阅 [Profiler 工具](../user_guides/profiler)。

- FP16 模型推理

  - 单测编译的时候，需要添加 `--build_arm82_fp16=ON` 选项，即：

  ```bash
  $ export NDK_ROOT=/disk/android-ndk-r20b #ndk_version > 19
  $ ./lite/tools/build.sh \
    --arm_os=android \
    --arm_abi=armv8 \
    --build_extra=on \
    --build_cv=on \
    --arm_lang=clang \
    --build_arm82_fp16=ON \
    test
  ```

  - 模型在 OPT 转换的时候，需要添加 `--enable_fp16=1` 选项，完成 FP16 模型转换，即：

  ```bash
  $ ./build.opt/lite/api/opt \
    --optimize_out_type=naive_buffer \
    --enable_fp16=1 \
    --optimize_out caffe_mv1_fp16 \
    --model_dir ./caffe_mv1
  ```

  - 执行

    - 推送 OPT 转换后的模型至设备, 运行时请将 `use_optimize_nb` 设置为1

    ```bash
    将转换好的模型文件推送到 `/data/local/tmp/arm_cpu` 目录下
    $ adb push caffe_mv1_fp16.nb /data/local/tmp/arm_cpu/
    $ adb shell chmod +x /data/local/tmp/arm_cpu/test_mobilenetv1

    $ adb shell "\
       /data/local/tmp/arm_cpu/test_mobilenetv1 \
       --use_optimize_nb=1 \
       --model_dir=/data/local/tmp/arm_cpu/caffe_mv1_fp16 \
       --input_shape=1,3,224,224 \
       --warmup=10 \
       --repeats=100"
    ```

    - 推送原始模型至设备, 运行时请将 `use_optimize_nb` 设置为0， `use_fp16` 设置为1；（`use_fp16` 默认为0）

    ```bash
    将 fluid 原始模型文件推送到 `/data/local/tmp/arm_cpu` 目录下
    $ adb push caffe_mv1 /data/local/tmp/arm_cpu/
    $ adb shell chmod +x /data/local/tmp/arm_cpu/test_mobilenetv1

    $ adb shell "export GLOG_v=1; \
        /data/local/tmp/arm_cpu/test_mobilenetv1 \
        --use_optimize_nb=0 \
        --use_fp16=1 \
        --model_dir=/data/local/tmp/arm_cpu/caffe_mv1 \
        --input_shape=1,3,224,224 \
        --warmup=10 \
       --repeats=100"
    ```

  注：如果想输入真实数据，请将预处理好的输入数据用文本格式保存。在执行的时候加上 `--in_txt=./*.txt` 选项即可
