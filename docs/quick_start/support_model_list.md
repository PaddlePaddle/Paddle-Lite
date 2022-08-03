# 支持模型

目前，Paddle Lite 已严格验证 52 个模型的精度和性能。对视觉类模型做到了充分的支持，覆盖分类、检测和定位，也包含了特色的 OCR 模型的支持。对 NLP 模型也做到了广泛支持，包含翻译、语义表达等等。

除了已严格验证的模型，Paddle Lite 对其他 CV 和 NLP 模型也可以做到大概率支持。

| 类别 | 类别细分 | 模型 | 支持平台 |
|-|-|:-|:-|
| CV | 分类 | [MobileNetV1](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz) | ARM, X86, GPU(OPENCL,METAL), HuaweiKirinNPU, RockchipNPU, MediatekAPU, KunlunxinXPU, HuaweiAscendNPU, VerisiliconTIMVX, AndroidNNAPI |
| CV | 分类 | [MobileNetV2](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v2_fp32_224_fluid.tar.gz) | ARM, X86, GPU(OPENCL,METAL), HuaweiKirinNPU, KunlunxinXPU, HuaweiAscendNPU |
| CV | 分类 | [MobileNetV3_large](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_large_x1_0.tar.gz) | ARM, X86, GPU(OPENCL,METAL), HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | [MobileNetV3_small](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_small_x1_0.tar.gz) | ARM, X86, GPU(OPENCL,METAL), HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | DPN68 | ARM, X86, HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | AlexNet | ARM, X86, HuaweiAscendNPU |
| CV | 分类 | DarkNet53 | ARM, X86, HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | DenseNet121 | ARM, X86, HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | [EfficientNetB0](https://paddlelite-demo.bj.bcebos.com/models/EfficientNetB0.tar.gz) | ARM, X86, GPU(OPENCL), KunlunxinXPU, HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | GhostNet_x1_3 | ARM,HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | HRNet_W18_C | ARM, X86,HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | RegNetX_4GF | ARM, X86 |
| CV | 分类 | Xception41 | ARM, X86 |
| CV | 分类 | [ResNet18](https://paddlelite-demo.bj.bcebos.com/models/resnet18_fp32_224_fluid.tar.gz) | ARM, X86, GPU(OPENCL,METAL), HuaweiKirinNPU, RockchipNPU, KunlunxinXPU, HuaweiAscendNPU, VerisiliconTIMVX, AndroidNNAPI |
| CV | 分类 | [ResNet50](https://paddlelite-demo.bj.bcebos.com/models/resnet50_fp32_224_fluid.tar.gz) | ARM, X86, GPU(OPENCL,METAL), HuaweiKirinNPU, RockchipNPU, KunlunxinXPU, HuaweiAscendNPU, VerisiliconTIMVX, AndroidNNAPI, IntelOpenVINO|
| CV | 分类 | [ResNet101](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz) | ARM, X86, HuaweiKirinNPU, RockchipNPU, KunlunxinXPU, HuaweiAscendNPU |
| CV | 分类 | [ResNeXt50](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz) | ARM, X86, HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | [MnasNet](https://paddlelite-demo.bj.bcebos.com/models/mnasnet_fp32_224_fluid.tar.gz)| ARM, HuaweiKirinNPU, HuaweiAscendNPU |
| CV | 分类 | [SqueezeNet](https://paddlelite-demo.bj.bcebos.com/models/squeezenet_fp32_224_fluid.tar.gz) | ARM, HuaweiKirinNPU, KunlunxinXPU, HuaweiAscendNPU |
| CV | 分类 | ShuffleNet | ARM,HuaweiAscendNPU |
| CV | 分类 | [ShufflenetV2](https://paddlelite-demo.bj.bcebos.com/models/shufflenetv2.tar.gz) | ARM, KunlunxinXPU, HuaweiAscendNPU |
| CV | 分类 | [InceptionV3](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz) | ARM, X86, HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 分类 | [InceptionV4](https://paddlelite-demo.bj.bcebos.com/models/inceptionv4.tar.gz) | ARM, X86, HuaweiKirinNPU, KunlunxinXPU, HuaweiAscendNPU |
| CV | 分类 | VGG16 | ARM, X86, GPU(OPENCL), KunlunxinXPU, HuaweiAscendNPU |
| CV | 分类 | VGG19 | ARM, X86, GPU(OPENCL,METAL), KunlunxinXPU, HuaweiAscendNPU|
| CV | 分类 | GoogleNet | ARM, X86, KunlunxinXPU, HuaweiAscendNPU, HuaweiKirinNPU |
| CV | 检测 | [SSD-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_fluid.tar.gz) | ARM, HuaweiKirinNPU*, HuaweiAscendNPU*, VerisiliconTIMVX, AndroidNNAPI |
| CV | 检测 | [SSD-MobileNetV3-large](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/ssdlite_mobilenet_v3_large.tar.gz) | ARM, X86, GPU(OPENCL,METAL),HuaweiAscendNPU* |
| CV | 检测 | [SSD-VGG16](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_vgg16_300_240e_voc.tgz) | ARM, X86, HuaweiAscendNPU* |
| CV | 检测 | [YOLOv3-DarkNet53](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz) | ARM, X86, HuaweiAscendNPU* |
| CV | 检测 | [YOLOv3-MobileNetV1](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz) | ARM, X86, HuaweiAscendNPU*, HuaweiKirinNPU |
| CV | 检测 | [YOLOv3-MobileNetV3](https://paddlelite-demo.bj.bcebos.com/models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid.tar.gz) | ARM, X86, HuaweiAscendNPU*, HuaweiKirinNPU |
| CV | 检测 | [yolov3_r50vd_dcn](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz) | ARM, HuaweiKirinNPU*, HuaweiAscendNPU*, HuaweiKirinNPU |
| CV | 检测 | [YOLOv4](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov4_cspdarknet.tgz) | ARM, X86, HuaweiAscendNPU* |
| CV | 检测 | Faster RCNN | ARM |
| CV | 检测 | [Mask RCNN*](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md) | ARM |
| CV | 检测 | ppyolo_2x | ARM,HuaweiAscendNPU* |
| CV | 检测 | solov2_r50_fpn_1x | ARM |
| CV | OCR | ch_ppocr_mobile_v2.0_cls_infer | ARM, X86, GPU(OPENCL),HuaweiAscendNPU |
| CV | OCR | [ch_ppocr_mobile_v2.0_det_infer](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_det_infer.tgz) | ARM, X86, GPU(OPENCL), HuaweiAscendNPU, HuaweiKirinNPU |
| CV | OCR | [ch_ppocr_mobile_v2.0_rec_infer](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/ch_ppocr_mobile_v2.0_rec_infer.tgz) | ARM, X86, GPU(OPENCL), HuaweiAscendNPU*, HuaweiKirinNPU* |
| CV | OCR | ch_ppocr_server_v2.0_rec_infer | ARM,HuaweiAscendNPU*, HuaweiKirinNPU* |
| CV | OCR | CRNN | ARM,HuaweiAscendNPU |
| CV | OCR | DB | ARM, GPU(OPENCL),HuaweiAscendNPU |
| CV | OCR | [OCR-Attention](https://paddle-inference-dist.bj.bcebos.com/ocr_attention.tar.gz) | ARM |
| CV | REG | inference_dnn | ARM, GPU(OPENCL) |
| CV | 分割 | [Deeplabv3](https://paddlelite-demo.bj.bcebos.com/models/deeplab_mobilenet_fp32_fluid.tar.gz) | ARM, GPU(OPENCL), HuaweiAscendNPU |
| CV | 分割 | [UNet](https://paddlelite-demo.bj.bcebos.com/models/Unet.zip) | ARM, GPU(OPENCL), HuaweiAscendNPU |
| CV | 分割 | bisenet | ARM, GPU(OPENCL),HuaweiAscendNPU |
| CV | 分割 | fastscnn | ARM, GPU(OPENCL) |
| CV | 分割 | bisenet_v2 | ARM, GPU(OPENCL),HuaweiAscendNPU |
| CV | 关键点 | [HigherHRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/higherhrnet_hrnet_w32_640.tgz) | ARM,X86,HuaweiAscendNPU,HuaweiKirinNPU |
| CV | 关键点 | [HRNet](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/hrnet_w32_384x288.tgz) | ARM,X86,HuaweiAscendNPU,HuaweiKirinNPU |
| CV | 人脸 | [FaceDetection](https://paddlelite-demo.bj.bcebos.com/models/facedetection_fp32_240_430_fluid.tar.gz) | ARM |
| CV | 人脸 | [FaceBoxes*](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/featured_model/FACE_DETECTION.md#FaceBoxes) | ARM, GPU(OPENCL), HuaweiAscendNPU |
| CV | 人脸 | [BlazeFace*](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/featured_model/FACE_DETECTION.md#BlazeFace) | ARM,HuaweiAscendNPU |
| CV | 人脸 | [MTCNN](https://paddlelite-demo.bj.bcebos.com/models/mtcnn.zip)  | ARM, GPU(OPENCL) |
| NLP | 机器翻译 | [Transformer*](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/machine_translation/transformer) | ARM,HuaweiKirinNPU*,HuaweiAscendNPU* |
| NLP | 机器翻译 | [BERT](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/bert.tar.gz) | KunlunxinXPU,HuaweiAscendNPU |
| NLP | 语义表示 | [ERNIE](https://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ernie.tar.gz) | KunlunxinXPU,HuaweiAscendNPU |
| NLP | 语义理解 | [ERNIE-TINY](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_tiny.tgz) | ARM,KunlunxinXPU,HuaweiAscendNPU |
| GAN | 风格转换 | [CycleGAN*](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan/cycle_gan) | HuaweiKirinNPU |
| GAN | 超分辨率 | [ESRGAN](https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleGAN/esrgan_psnr_x4_div2k.tgz) |ARM,X86,HuaweiAscendNPU,HuaweiKirinNPU|

**注意：**

1. 模型列表中 * 代表该模型链接来自[ PaddlePaddle/models ](https://github.com/PaddlePaddle/models)，否则为推理模型的下载链接
2. 支持平台列表中 HuaweiKirinNPU* 代表 ARM + HuaweiKirinNPU 异构计算，否则为 HuaweiKirinNPU 计算
3. 支持平台列表中 HuaweiAscendNPU* 代表 X86 或 ARM+HuaweiAscendNPU 异构计算，否则为 HuaweiAscendNPU 计算
4. 寻找更多的可支持的模型还可以转至[ PaddlePaddle/models ](https://github.com/PaddlePaddle/models), [ PaddleHub ](https://github.com/PaddlePaddle/PaddleHub), [ PaddleOCR ](https://github.com/PaddlePaddle/PaddleOCR), [ PaddleDetection ](https://github.com/PaddlePaddle/PaddleDetection)
