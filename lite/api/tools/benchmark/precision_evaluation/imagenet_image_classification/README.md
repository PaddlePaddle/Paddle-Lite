# 使用 ILSVRC 2012 验证集进行分类模型的精度评测

## 1. 背景
`benchmark_bin` 工具可以读取真实的验证集数据，进行计算后，给出性能数据和精度数据。

2012 ILSVRC 验证集共 5K 张图片，约 6GB 存储空间，占用空间比较大，为适应端侧应用场景从中筛选出 1K 张图片进行验证。
筛选方法：从验证集中，每个类别挑选一张图片，最后形成一个含有 1K 张图片的验证集子集。

[ILSVRC 2012 image classification task](http://www.image-net.org/challenges/LSVRC/2012/)，可在此链接下载。

## 2. 适用场景
目前基于验证集的精度评测功能仅支持安卓系统，对其它系统的支持在开发中。

## 3. 在 Android 上运行精度评测
### 3.1 编译

```shell
./lite/tools/build_android.sh --toolchain=clang --with_benchmark=ON full_publish
```

编译完成后，会生成 `build.lite.*./lite/api/benchmark_bin` 二进制文件、预编译好的 OpenCV4.1.0 和验证集。

```
   build.lite.*./third_party
   ├── opencv4.1.0                              预编译好的 OpenCV
   ├── validation_dataset
   │   ├── ILSVRC2012_1000_cls                  ILSVRC 2012 图片数据
   │   ├── ILSVRC2012_1000_cls_label_list.txt   ILSVRC 2012 标签数据
```

### 3.2 运行
需要将如下文件通过 `adb` 上传至手机：
- Paddle 模型（combined 或 uncombined 格式均可）或已经 `opt` 工具离线优化后的 `.nb` 文件
- 二进制文件`benchmark_bin`
- ILSVRC 2012 图片数据
- ILSVRC 2012 标签数据
- 数据集的配置文件 `config.txt`

在 Host 端机器上操作例子如下：
```shell
# 获取模型文件
wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
tar zxvf MobileNetV1.tar.gz

# 上传文件
adb shell mkdir /data/local/tmp/benchmark
adb push MobileNetV1 /data/local/tmp/benchmark
adb push build.lite.android.armv8.clang/lite/api/benchmark_bin /data/local/tmp/benchmark
adb push build.lite.android.armv8.clang/third_party/validation_dataset /data/local/tmp/benchmark
adb push lite/api/tools/benchmark/precision_evaluation/imagenet_image_classification/config.txt /data/local/tmp/benchmark

# 执行性能测试
adb shell "cd /data/local/tmp/benchmark;
  ./benchmark_bin \
    --model_file=MobileNetV1/inference.pdmodel \
    --param_file=MobileNetV1/inference.pdiparams \
    --validation_set=ILSVRC2012 \
    --config_path=config.txt \
    --input_shape=1,3,224,224 \
    --warmup=10 \
    --repeats=20 \
    --backend=opencl"
```

部分输出日志如下：

```shell
======= Accurancy Info =======
config: /data/local/tmp/config.txt
label_path:/data/local/tmp/imagenet1k_label_list.txt
ground_truth_images_path:/data/local/tmp/ILSVRC2012_1000_cls/val_list_1k.txt
resize_short_size:256
crop_size:224
mean:0.485,0.456,0.406
scale:1/0.229,1/0.224,1/0.225
topk:2
store_result_as_image:1

Top-1 Accurancy: 0.739
Top-2 Accurancy: 0.121

======= Perf Info =======
Time(unit: ms):
init  = 11.925
first = 119.312
min   = 6.472
max   = 9.091
avg   = 7.785
```
