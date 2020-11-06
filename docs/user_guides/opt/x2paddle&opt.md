## 合并x2paddle和opt的一键脚本

**背景**：如果想用Paddle-Lite运行第三方来源（tensorflow、caffe、onnx）模型，一般需要经过两次转化。即使用x2paddle工具将第三方模型转化为PaddlePaddle格式，再使用opt将PaddlePaddle模型转化为Padde-Lite可支持格式。
为了简化这一过程，我们提供一键脚本，将x2paddle转化和opt转化合并：

**一键转化脚本**：[auto_transform.sh](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.3/lite/tools/auto_transform.sh)


**环境要求**：使用`auto_transform.sh`脚本转化第三方模型时，需要先安装x2paddle环境，请参考[x2paddle环境安装方法](https://github.com/PaddlePaddle/X2Paddle#环境依赖) 安装x2paddle和x2paddle依赖项(tensorflow、caffe等)。

**使用方法**：

（1）打印帮助帮助信息：`sh ./auto_transform.sh`，Ubuntu下请执行 `bash ./auto_transform.sh`

（2）转化模型方法

```bash
USAGE:
    auto_transform.sh combines the function of x2paddle and opt, it can 
    tranform model from tensorflow/caffe/onnx form into paddle-lite naive-buffer form.
----------------------------------------
example:
    sh ./auto_transform.sh --framework=tensorflow --model=tf_model.pb --optimize_out=opt_model_result
----------------------------------------
Arguments about x2paddle:
    --framework=(tensorflow|caffe|onnx);
    --model='model file for tensorflow or onnx';
    --prototxt='proto file for caffe' --weight='weight file for caffe'
 For TensorFlow:
   --framework=tensorflow --model=tf_model.pb

 For Caffe:
   --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel

 For ONNX
   --framework=onnx --model=onnx_model.onnx

Arguments about opt:
    --valid_targets=(arm|opencl|x86|npu); valid targets on Paddle-Lite.
    --fluid_save_dir='path to outputed model after x2paddle'
    --optimize_out='path to outputed Paddle-Lite model'
----------------------------------------
```
