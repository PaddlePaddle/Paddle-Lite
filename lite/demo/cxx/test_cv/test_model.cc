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
#include "paddle_api.h"  // NOLINT
// #include "paddle_image_preprocess.h"
// #include "time.h"

using namespace paddle::lite_api;  // NOLINT

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

void pre_process(const cv::Mat& img,
                 int width,
                 int height,
                 Tensor dstTensor,
                 float* means,
                 float* scales) {
#ifdef LITE_WITH_CV
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
  img_process.imageConvert(img_ptr, rgb_ptr);
  // do resize
  img_process.imageResize(rgb_ptr, resize_ptr);
  // data--tensor and normalize
  img_process.image2Tensor(
      resize_ptr, dstTensor, LayoutType::kNCHW, means, scales);
#else
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  float* data = dstTensor.mutable_data<float>();
  neon_mean_scale(dimg, data, width * height, means, scales);
#endif
}

void RunModel(std::string model_dir,
              std::string img_path,
              std::vector<int> input_shape,
              PowerMode power_mode,
              int thread_num,
              int test_iter,
              int warmup = 0) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_dir(model_dir);
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
  //   pre_process(img, width, height, data);
  float means[3] = {103.94f, 116.78f, 123.68f};
  float scales[3] = {0.017f, 0.017f, 0.017f};
  pre_process(img, width, height, *input_tensor, means, scales);

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
    LOG(INFO) << "iter: " << i << ", time: " << t << " ms";
  }
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir
            << ", power_mode: " << static_cast<int>(power_mode)
            << ", threads num " << thread_num << ", warmup: " << warmup
            << ", repeats: " << test_iter << ", avg time: " << lps / test_iter
            << " ms"
            << ", min time: " << min_time << " ms"
            << ", max time: " << max_time << " ms.";

  // 5. Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int output_num = 1;
  for (int i = 0; i < shape_out.size(); ++i) {
    output_num *= shape_out[i];
  }
  LOG(INFO) << "output_num: " << output_num;
  LOG(INFO) << "out " << outptr[0];
  LOG(INFO) << "out " << outptr[1];
}

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " model_dir image_path input_shape\n";
    exit(1);
  }
  std::string model_dir = argv[1];
  std::string img_path = argv[2];
  std::vector<int> input_shape;
  input_shape[0] = atoi(argv[3]);
  input_shape[1] = atoi(argv[4]);
  input_shape[2] = atoi(argv[5]);
  input_shape[3] = atoi(argv[6]);
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

  RunModel(model_dir,
           img_path,
           input_shape,
           (PowerMode)power_mode,
           threads,
           test_iter,
           warmup);
  return 0;
}
