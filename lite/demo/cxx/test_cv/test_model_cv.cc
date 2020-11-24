// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"               // NOLINT
#include "paddle_image_preprocess.h"  // NOLINT
#include "time.h"                     // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api;  // NOLINT

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}
// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(
    const float* din, float* dout, int size, float* mean, float* scale) {
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);

  float* dout_c0 = dout;
  float* dout_c1 = dout + size;
  float* dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c0++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c0++) = (*(din++) - mean[2]) * scale[2];
  }
}
void pre_process(const cv::Mat& img, int width, int height, Tensor dstTensor) {
#ifdef LITE_WITH_CV
  typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
  typedef paddle::lite::utils::cv::FlipParam FlipParam;
  typedef paddle::lite::utils::cv::TransParam TransParam;
  typedef paddle::lite::utils::cv::ImagePreprocess ImagePreprocess;
  typedef paddle::lite_api::DataLayoutType LayoutType;
  // init TransParam
  TransParam tp;
  tp.iw = img.cols;
  tp.ih = img.rows;
  tp.ow = width;
  tp.oh = height;
  ImageFormat srcFormat = ImageFormat::BGR;
  ImageFormat dstFormat = ImageFormat::RGB;
  // init ImagePreprocess
  ImagePreprocess img_process(srcFormat, dstFormat, tp);
  // init temp var
  const uint8_t* img_ptr = reinterpret_cast<const uint8_t*>(img.data);
  uint8_t* rgb_ptr = new uint8_t[img.cols * img.rows * 3];
  uint8_t* resize_ptr = new uint8_t[width * height * 3];
  // do convert bgr--rgb
  img_process.image_convert(img_ptr, rgb_ptr);
  // do resize
  img_process.image_resize(rgb_ptr, resize_ptr);
  // data--tensor and normalize
  float means[3] = {103.94f, 116.78f, 123.68f};
  float scales[3] = {0.017f, 0.017f, 0.017f};
  img_process.image_to_tensor(
      resize_ptr, &dstTensor, LayoutType::kNCHW, means, scales);
  float* data = dstTensor.mutable_data<float>();
#else
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  float means[3] = {0.485f, 0.456f, 0.406f};
  float scales[3] = {0.229f, 0.224f, 0.225f};
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  float* data = dstTensor.mutable_data<float>();
  neon_mean_scale(dimg, data, width * height, means, scales);
#endif
}

void RunModel(std::string model_file,
              std::string img_path,
              std::vector<int> input_shape,
              PowerMode power_mode,
              int thread_num,
              int test_iter,
              int warmup = 0) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);
  config.set_power_mode(power_mode);
  config.set_threads(thread_num);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  // 3. Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(
      {input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
  auto* data = input_tensor->mutable_data<float>();
  // read img and pre-process
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);

  pre_process(img, input_shape[3], input_shape[2], *input_tensor);

  // 4. Run predictor
  for (int i = 0; i < warmup; ++i) {
    predictor->Run();
  }
  double lps = 0.f;
  double min_time = 1000000.f;
  double max_time = 0.f;
  for (int i = 0; i < test_iter; ++i) {
    clock_t begin = clock();
    predictor->Run();
    clock_t end = clock();
    double t = (end - begin) * 1000;
    t = t / CLOCKS_PER_SEC;
    lps += t;
    if (t < min_time) {
      min_time = t;
    }
    if (t > max_time) {
      max_time = t;
    }
    std::cout << "iter: " << i << ", time: " << t << " ms" << std::endl;
  }
  std::cout << "================== Speed Report ==================="
            << std::endl;
  std::cout << "Model: " << model_file
            << ", power_mode: " << static_cast<int>(power_mode)
            << ", threads num " << thread_num << ", warmup: " << warmup
            << ", repeats: " << test_iter << ", avg time: " << lps / test_iter
            << " ms"
            << ", min time: " << min_time << " ms"
            << ", max time: " << max_time << " ms." << std::endl;

  // 5. Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int output_num = 1;
  for (int i = 0; i < shape_out.size(); ++i) {
    output_num *= shape_out[i];
  }
  std::cout << "output_num: " << output_num << std::endl;
  for (int i = 0; i < output_num; i += 100) {
    std::cout << "i: " << i << ", out: " << outptr[i] << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " model_file image_path input_shape\n";
    exit(1);
  }
  std::string model_file = argv[1];
  std::string img_path = argv[2];
  std::vector<int> input_shape;
  input_shape.push_back(atoi(argv[3]));
  input_shape.push_back(atoi(argv[4]));
  input_shape.push_back(atoi(argv[5]));
  input_shape.push_back(atoi(argv[6]));
  int power_mode = 3;
  int threads = 1;
  int test_iter = 100;
  int warmup = 10;
  if (argc > 7) {
    power_mode = atoi(argv[7]);
  }
  if (argc > 8) {
    threads = atoi(argv[8]);
  }
  if (argc > 9) {
    test_iter = atoi(argv[9]);
  }
  if (argc > 10) {
    warmup = atoi(argv[10]);
  }
  RunModel(model_file,
           img_path,
           input_shape,
           (PowerMode)power_mode,
           threads,
           test_iter,
           warmup);
  return 0;
}
