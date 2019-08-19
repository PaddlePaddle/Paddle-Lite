# Paddle Web

Paddle Web is an open source deep learning framework designed to work on web browser. It is compatible with PaddlePaddle model.

## Key Features

### Modular

Paddle Web is built on Atom system which is a versatile framework to support GPGPU operation on WebGL. It is quite modular and could be used to make computation tasks faster by utilizing WebGL.

### High Performance

Paddle Web could run TinyYolo model in less than 30ms on chrome. This is fast enough to run deep learning models in many realtime scenarios.

### High Compatibility

Hardware compatibility: Paddle Lite supports a diversity of hardwares — ARM CPU, Mali GPU, Adreno GPU, Huawei NPU and FPGA. In the near future, we will also support AI microchips from Cambricon and Bitmain.

Model compatibility: The Op of Paddle Lite is fully compatible to that of PaddlePaddle. The accuracy and performance of 18 models (mostly CV models and OCR models) and 85 operators have been validated. In the future, we will also support other models.

Framework compatibility: In addition to models trained on PaddlePaddle, those trained on Caffe and TensorFlow can also be converted to be used on Paddle Lite, via [X2Paddle](https://github.com/PaddlePaddle/X2Paddle). In the future to come, we will also support models of ONNX format.

## How To Build

```bash
npm i
npm run server
```

## Feedback and Community Support

- Questions, reports, and suggestions are welcome through Github Issues!
- Forum: Opinions and questions are welcome at our [PaddlePaddle Forum](https://ai.baidu.com/forum/topic/list/168)！
- QQ group chat: 696965088
