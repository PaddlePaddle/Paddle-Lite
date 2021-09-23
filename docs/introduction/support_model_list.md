# 支持模型

目前，Paddle-Lite已严格验证 52 个模型的精度和性能。对视觉类模型做到了充分的支持，覆盖分类、检测和定位，也包含了特色的 OCR 模型的支持。对 NLP 模型也做到了广泛支持，包含翻译、语义表达等。

除了已严格验证的模型，Paddle-Lite对其他CV和NLP模型也可以做到大概率支持。

| 类别 | 类别细分 | 模型 | 支持平台 |
|-|-|:-|:-|
| CV | 分类 | [MobileNetV1](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz) | ARM, X86, GPU(OPENCL), NPU, RKNPU, APU |
| CV | 分类 | [MobileNetV2](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v2_fp32_224_fluid.tar.gz) | ARM, X86, GPU(OPENCL), NPU |
| CV | 分类 | [ResNet18](https://paddlelite-demo.bj.bcebos.com/models/resnet18_fp32_224_fluid.tar.gz) | ARM, GPU(OPENCL), NPU, RKNPU |
| CV | 分类 | [ResNet50](https://paddlelite-demo.bj.bcebos.com/models/resnet50_fp32_224_fluid.tar.gz) | ARM X86, GPU(OPENCL), NPU, XPU, RKNPU |
| CV | 分类 | [MnasNet](https://paddlelite-demo.bj.bcebos.com/models/mnasnet_fp32_224_fluid.tar.gz) | ARM, GPU(OPENCL), NPU |
| CV | 分类 | [EfficientNet*](https://github.com/PaddlePaddle/PaddleClas) | ARM, GPU(OPENCL)|
| CV | 分类 | [SqueezeNet](https://paddlelite-demo.bj.bcebos.com/models/squeezenet_fp32_224_fluid.tar.gz) | ARM, GPU(OPENCL), NPU |
| CV | 分类 | [ShufflenetV2*](https://github.com/PaddlePaddle/PaddleClas) | ARM, GPU(OPENCL) |
| CV | 分类 | [ShuffleNet](https://paddlepaddle-inference-banchmark.bj.bcebos.com/shufflenet_inference.tar.gz) | ARM, GPU(OPENCL) |
| CV | 分类 | [InceptionV4](https://paddle-inference-dist.bj.bcebos.com/inception_v4_simple.tar.gz) | ARM, X86, GPU(OPENCL), NPU |
| CV | 分类 | [VGG16](https://paddlepaddle-inference-banchmark.bj.bcebos.com/VGG16_inference.tar) | ARM, GPU(OPENCL) |
| CV | 分类 | [VGG19](https://paddlepaddle-inference-banchmark.bj.bcebos.com/VGG19_inference.tar) | ARM, GPU(OPENCL), XPU|
| CV | 分类 | [GoogleNet](https://paddlepaddle-inference-banchmark.bj.bcebos.com/GoogleNet_inference.tar) | ARM, X86, XPU |
| CV | 检测 | [MobileNet-SSD](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_fluid.tar.gz) | ARM, GPU(OPENCL), NPU* |
| CV | 检测 | [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid.tar.gz) | ARM, GPU(OPENCL), NPU* |
| CV | 检测 | [Faster RCNN](https://paddlepaddle-inference-banchmark.bj.bcebos.com/faster_rcnn.tar) | ARM, GPU(OPENCL) |
| CV | 检测 | [Mask RCNN*](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md) | ARM |
| CV | 分割 | [Deeplabv3](https://paddlelite-demo.bj.bcebos.com/models/deeplab_mobilenet_fp32_fluid.tar.gz) | ARM, OPENCL(GPU) |
| CV | 分割 | [UNet](https://paddlelite-demo.bj.bcebos.com/models/Unet.zip) | ARM, GPU(OPENCL) |
| CV | 人脸 | [FaceDetection](https://paddlelite-demo.bj.bcebos.com/models/facedetection_fp32_240_430_fluid.tar.gz) | ARM |
| CV | 人脸 | [FaceBoxes*](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/featured_model/FACE_DETECTION.md#FaceBoxes) | ARM |
| CV | 人脸 | [BlazeFace*](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/featured_model/FACE_DETECTION.md#BlazeFace) | ARM |
| CV | 人脸 | [MTCNN](https://paddlelite-demo.bj.bcebos.com/models/mtcnn.zip) | ARM, GPU(OPENCL) |
| CV | OCR | [OCR-Detection-DB](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) | ARM |
| CV | OCR | [OCR-Recognition-CRNN](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) | ARM, GPU(OPENCL) |
| CV | OCR | [OCR-angle classifier](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | ARM, GPU(OPENCL) |
| CV | OCR | [OCR-Attention](https://paddle-inference-dist.bj.bcebos.com/ocr_attention.tar.gz) | ARM |
| CV | GAN | [CycleGAN*](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan/cycle_gan) | NPU |
| NLP | 机器翻译 | [Transformer*](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/machine_translation/transformer) | ARM, NPU* |
| NLP | 机器翻译 | [BERT](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/bert.tar.gz) | XPU |
| NLP | 语义表示 | [ERNIE](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ernie.tar.gz) | XPU |

**注意：**
1. 模型列表中 * 代表该模型链接来自[PaddlePaddle/models](https://github.com/PaddlePaddle/models), 否则为推理模型的下载链接
2. 支持平台列表中 NPU* 代表ARM+NPU异构计算, 否则为NPU计算
3. 寻找更多的可支持的模型还可以转至[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)，[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)，[paddle-models](https://github.com/PaddlePaddle/models)， [PaddleHub](https://github.com/PaddlePaddle/PaddleHub)




# 支持模型

目前，Paddle-Lite 已严格验证 52 个模型的精度和性能。对视觉类模型做到了充分的支持，覆盖分类、检测和定位，也包含了特色的 OCR 模型的支持。对 NLP 模型也做到了广泛支持，包含翻译、语义表达等等。

除了已严格验证的模型，Paddle-Lite 对其他 CV 和 NLP 模型也可以做到大概率支持。
| 类别 | 类别细分 | 模型 | 支持平台 |
|-|-|:-|:-|
| CV | 分类 | MobileNetV1 | ARM，X86，HuaweiKirinNPU，RockchipNPU，MediatekAPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | MobileNetV2 | ARM，X86，HuaweiKirinNPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | DPN68 | ARM，X86 |
| CV | 分类 | AlexNet | ARM，X86 |
| CV | 分类 | DarkNet53 | ARM，X86 |
| CV | 分类 | DenseNet121 | ARM，X86 |
| CV | 分类 | EfficientNetB0 | ARM，X86 |
| CV | 分类 | GhostNet_x1_3 | ARM |
| CV | 分类 | HRNet_W18_C | ARM，X86 |
| CV | 分类 | RegNetX_4GF | ARM，X86 |
| CV | 分类 | Xception41 | ARM，X86 |
| CV | 分类 | ResNet18 | ARM，HuaweiKirinNPU，RockchipNPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | ResNet50 | ARM，X86，HuaweiKirinNPU，RockchipNPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | MnasNet| ARM，HuaweiKirinNPU，HuaweiAscendNPU |
| CV | 分类 | EfficientNetB0 | ARM，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | SqueezeNet | ARM，HuaweiKirinNPU，BaiduXPU |
| CV | 分类 | ShufflenetV2 | ARM，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | ShuffleNet | ARM |
| CV | 分类 | InceptionV4 | ARM，X86，HuaweiKirinNPU，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | VGG16 | ARM，BaiduXPU，HuaweiAscendNPU |
| CV | 分类 | VGG19 | ARM，BaiduXPU，HuaweiAscendNPU|
| CV | 分类 | GoogleNet | ARM，X86，BaiduXPU |
| CV | 检测 | MobileNet-SSD | ARM，HuaweiKirinNPU*，HuaweiAscendNPU* |
| CV | 检测 | YOLOv3-MobileNetV3 | ARM，HuaweiKirinNPU*，HuaweiAscendNPU* |
| CV | 检测 | Faster RCNN | ARM |
| CV | 检测 | Mask RCNN* | ARM |
| CV | 检测 | yolov3_darknet53_270e_coco | ARM |
| CV | 检测 | ppyolo_2x | ARM |
| CV | 检测 | solov2_r50_fpn_1x | ARM |
| CV | 检测 | yolov3_mobilenet_v3 | ARM |
| CV | 检测 | yolov3_r50vd_dcn | ARM |
| CV | OCR | ch_ppocr_mobile_v2.0_cls_infer | ARM，X86 |
| CV | OCR | ch_ppocr_mobile_v2.0_det_infer | ARM |
| CV | OCR | ch_ppocr_mobile_v2.0_rec_infer | ARM |
| CV | OCR | ch_ppocr_server_v2.0_rec_infer | ARM |
| CV | OCR | CRNN | ARM |
| CV | OCR | DB | ARM |
| CV | OCR | OCR-Attention | ARM |
| CV | REG | inference_dnn | ARM |
| CV | 分割 | Deeplabv3 | ARM |
| CV | 分割 | UNet | ARM |
| CV | 分割 | bisenet | ARM |
| CV | 分割 | fastscnn | ARM |
| CV | 分割 | bisenet_v2 | ARM |
| CV | 人脸 | FaceDetection | ARM |
| CV | 人脸 | FaceBoxes*| ARM |
| CV | 人脸 | BlazeFace* | ARM |
| CV | 人脸 | MTCNN | ARM |
| CV | GAN | CycleGAN* | HuaweiKirinNPU |
| NLP | 机器翻译 | Transformer* | ARM，HuaweiKirinNPU* |
| NLP | 机器翻译 | BERT | BaiduXPU |
| NLP | 语义表示 | ERNIE | BaiduXPU |

**注意：**
1. 模型列表中 * 代表该模型链接来自[ PaddlePaddle/models ](https://github.com/PaddlePaddle/models)，否则为推理模型的下载链接
2. 支持平台列表中 HuaweiKirinNPU* 代表 ARM + HuaweiKirinNPU 异构计算，否则为 HuaweiKirinNPU 计算
3. 支持平台列表中 HuaweiAscendNPU* 代表 X86 或 ARM+HuaweiAscendNPU 异构计算，否则为 HuaweiAscendNPU 计算
