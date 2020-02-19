# C++ Demo

## 编译

首先按照[PaddleLite 源码编译](https://github.com/PaddlePaddle/Paddle-Lite/wiki/source_compile)准备交叉编译环境，之后拉取最新[PaddleLite release发布版代码](https://github.com/PaddlePaddle/Paddle-Lite)。下面以Android-ARMv8架构为例，介绍编译过程，并最终在手机上跑通MobilNetv1模型。

进入 Paddle-Lite 目录，运行以下命令编译代码（**需加编译选项`--build_extra=ON`确保完整编译**）：

```
./lite/tools/build.sh        \
    --arm_os=android         \
    --arm_abi=armv8          \
    --arm_lang=gcc           \
    --android_stl=c++_static \
    --build_extra=ON         \
    full_publish
```

编译完成后 `./build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/` 文件夹下包含：

- cxx
	- include (头文件文件夹)
	- lib          (库文件文件夹)
		- libpaddle_api_full_bundled.a
		- libpaddle_api_light_bundled.a
		- libpaddle_light_api_shared.so
		- libpaddle_full_api_shared.so
- demo
	- cxx  （C++ demo）
		- mobile_light  (light api demo)
		- mobile_full    (full api demo)
    - mobile_detection    (detection model api demo)
    - mobile_classify    (classify model api demo)
		- Makefile.def
		- include
- third_party  （第三方库文件夹）
	- gflags

## 准备执行环境

执行环境有两种：使用安卓手机；若没安卓手机，也可在安卓模拟器中执行。

### 环境一：使用安卓手机

将手机连上电脑，在手机上打开选项 -> 开启-开发者模式 -> 开启-USB调试模式。确保 `adb devices` 能够看到相应的设备。

### 环境二：使用安卓模拟器

运行下面命令，分别创建安卓armv8、armv7架构的模拟器。若需在真机测试，将模拟器换成相应架构的真机环境即可。

```
*android-armv8*
adb kill-server
adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
echo n | avdmanager create avd -f -n paddle-armv8 -k "system-images;android-24;google_apis;arm64-v8a"
echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv8 -noaudio -no-window -gpu off -port 5554 &
sleep 1m
```

```
*android-armv7*
adb kill-server
adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
echo n | avdmanager create avd -f -n paddle-armv7 -k "system-images;android-24;google_apis;armeabi-v7a"
echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv7 -noaudio -no-window -gpu off -port 5554 &
sleep 1m
```

## 下载模型并运行示例

```
cd inference_lite_lib.android.armv8/demo/cxx/mobile_full
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz

make

adb push mobilenet_v1 /data/local/tmp/
adb push mobilenetv1_full_api /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobilenetv1_full_api
adb shell "/data/local/tmp/mobilenetv1_full_api --model_dir=/data/local/tmp/mobilenet_v1 --optimized_model_dir=/data/local/tmp/mobilenet_v1.opt"
```

注：我们也提供了轻量级 API 的 demo、图像分类demo和目标检测demo，支持图像输入；

### Light API Demo

```
cd ../mobile_light
make
adb push mobilenetv1_light_api /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobilenetv1_light_api
adb shell "/data/local/tmp/mobilenetv1_light_api --model_dir=/data/local/tmp/mobilenet_v1.opt  "
```


### 图像分类 Demo

```
cd ../mobile_classify
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz
make
adb push mobile_classify /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push labels.txt /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobile_classify
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && /data/local/tmp/mobile_classify /data/local/tmp/mobilenet_v1.opt /data/local/tmp/test.jpg /data/local/tmp/labels.txt"
```

### 目标检测 Demo

```
cd ../mobile_detection
wget https://paddle-inference-dist.bj.bcebos.com/mobilenetv1-ssd.tar.gz
tar zxvf mobilenetv1-ssd.tar.gz
make
adb push mobile_detection /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobile_detection
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && /data/local/tmp/mobile_detection /data/local/tmp/mobilenetv1-ssd /data/local/tmp/test.jpg"
adb pull /data/local/tmp/test_detection_result.jpg ./
```

## Demo 程序运行结果

### light API Demo 运行结果

运行成功后 ，将在控制台输出预测结果的前10个类别的预测概率：

```
Output dim: 1000
Output[0]: 0.000191
Output[100]: 0.000160
Output[200]: 0.000264
Output[300]: 0.000211
Output[400]: 0.001032
Output[500]: 0.000110
Output[600]: 0.004829
Output[700]: 0.001845
Output[800]: 0.000202
Output[900]: 0.000586
```

### 图像分类 Demo 运行结果

运行成功后 ，将在控制台输出预测结果的前5个类别的类型索引、名字和预测概率：

```
parameter:  model_dir, image_path and label_file are necessary
parameter:  topk, input_width,  input_height, are optional
i: 0, index: 285, name:  Egyptian cat, score: 0.482870
i: 1, index: 281, name:  tabby, tabby cat, score: 0.471593
i: 2, index: 282, name:  tiger cat, score: 0.039779
i: 3, index: 287, name:  lynx, catamount, score: 0.002430
i: 4, index: 722, name:  ping-pong ball, score: 0.000508
```

### 目标检测 Demo 运行结果

运行成功后 ，将在控制台输出检测目标的类型、预测概率和坐标：

```
running result:
detection image size: 935, 1241, detect object: person, score: 0.996098, location: x=187, y=43, width=540, height=592
detection image size: 935, 1241, detect object: person, score: 0.935293, location: x=123, y=639, width=579, height=597
```

## 如何在代码中使用 API

在C++中使用PaddleLite API非常简单，不需要添加太多额外代码，具体步骤如下：

- 加入头文件引用

```
  #include <iostream>
  #include <vector>
  #include "paddle_api.h"
  #include "paddle_use_kernels.h"
  #include "paddle_use_ops.h"
  #include "paddle_use_passes.h"
```

- 通过MobileConfig设置：模型文件位置（model_dir）、线程数（thread）和能耗模式( power mode )。输入数据（input），从 MobileConfig 创建 PaddlePredictor 并执行预测。  （注：Lite还支持从memory直接加载模型，可以通过MobileConfig::set_model_buffer方法实现）

代码示例：

```
// 1. Create MobileConfig
MobileConfig config;

// 2. Load model
config.set_model_dir("path to your model directory"); // model dir
/*load model: Lite supports loading model from file or from memory (naive buffer from optimized model)
//Method One: Load model from memory:
void set_model_buffer(const char* model_buffer,
                    size_t model_buffer_size,
                    const char* param_buffer,
                    size_t param_buffer_size)
//Method Two: Load model from file:
void set_model_dir(const std::string& model_dir)  */

// 3. Set MobileConfig (or you can skip this step to use default value):
config.set_power_mode(LITE_POWER_HIGH); // power mode
/*power modes: Lite supports the following power modes
    LITE_POWER_HIGH
    LITE_POWER_LOW
    LITE_POWER_FULL
    LITE_POWER_NO_BIND
    LITE_POWER_RAND_HIGH
    LITE_POWER_RAND_LOW */
config.set_threads("num of threads"); // threads

// 4. Create PaddlePredictor by MobileConfig
std::shared_ptr<PaddlePredictor> predictor =
    CreatePaddlePredictor<MobileConfig>(config);

// 5. Prepare input data
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 224, 224});
auto *data = input_tensor -> mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}

// 6. Run predictor
predictor->Run();

// 7. Get output
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
```

## CxxConfig案例: OCR_model的运行

1. OCR 模型文件：
   - 我们提供Pb格式的[ocr_attention_mode](https://paddle-inference-dist.cdn.bcebos.com/ocr_attention.tar.gz)l下载
   - 也可以从[Paddle/model项目](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition)中训练出模型
2. 示例代码：


```
#include "paddle_api.h"         // NOLINT
#include "paddle_use_passes.h"  // NOLINT
#include <gflags/gflags.h>
#include <stdio.h>
#include <vector>
using namespace paddle::lite_api; // NOLINT

DEFINE_string(model_dir, "", "Model dir path.");
DEFINE_bool(prefer_int8_kernel, false, "Prefer to run model with int8 kernels");

int64_t ShapeProduction(const shape_t &shape) {
  int64_t res = 1;
  for (auto i : shape)
    res *= i;
  return res;
}

void RunModel() {
  // 1. Set CxxConfig
  CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  std::vector<Place> valid_places({Place{TARGET(kARM), PRECISION(kFloat)}});
  if (FLAGS_prefer_int8_kernel) {
    valid_places.insert(valid_places.begin(),
                        Place{TARGET(kARM), PRECISION(kInt8)});
  }
  config.set_valid_places(valid_places);

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);

  // 3. Prepare input data
  // input 0
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(shape_t({1, 1, 48, 512}));
  auto *data = input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    data[i] = 1;
  }
  // input1
  std::unique_ptr<Tensor> init_ids(std::move(predictor->GetInput(1)));
  init_ids->Resize(shape_t({1, 1}));
  auto *data_ids = init_ids->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(init_ids->shape()); ++i) {
    data_ids[i] = 0;
  }

  lod_t lod_i;
  lod_i.push_back({0, 1});
  lod_i.push_back({0, 1});
  init_ids->SetLoD(lod_i);
  // input2
  std::unique_ptr<Tensor> init_scores(std::move(predictor->GetInput(2)));
  init_scores->Resize(shape_t({1, 1}));
  auto *data_scores = init_scores->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(init_scores->shape()); ++i) {
    data_scores[i] = 0;
  }
  lod_t lod_s;
  lod_s.push_back({0, 1});
  lod_s.push_back({0, 1});
  init_scores->SetLoD(lod_s);

  // 4. Run predictor
  predictor->Run();

  // 5. Get output
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  for (int i = 0; i < ShapeProduction(output_tensor->shape()); i++) {
    printf("Output[%d]: %f\n", i, output_tensor->data<float>()[i]);
  }
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  RunModel();
  return 0;
}
```

3. 运行方法：
 参考以上代码编译出可执行文件`OCR_DEMO`，模型文件夹为`ocr_attention`。手机以USB调试、文件传输模式连接电脑。
```
简单编译出`OCR_DEMO`的方法：用以上示例代码替换编译结果中`build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/mobile_full/mobilenetv1_full_api.cc`文件的内容，终端进入该路径（`demo/cxx/mobile_full/`），终端中执行`make && mv mobilenetv1_full_api OCR_DEMO`即编译出了OCR模型的可执行文件`OCR_DEMO`
```
   在终端中输入以下命令执行OCR model测试：

```
#OCR_DEMO为编译出的可执行文件名称；ocr_attention为ocr_attention模型的文件夹名称；libpaddle_full_api_shared.so是编译出的动态库文件，位于`build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/lib`
adb push OCR_DEMO /data/local/tmp
adb push ocr_attention /data/local/tmp
adb push libpaddle_full_api_shared.so /data/local/tmp/
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && cd /data/local/tmp && ./OCR_DEMO --model_dir=./OCR_DEMO'
```

4. 运行结果

<img src='https://user-images.githubusercontent.com/45189361/64398400-46531580-d097-11e9-9f1c-5aba1dfbc24f.png' align='left' width="150" height="200"/>
