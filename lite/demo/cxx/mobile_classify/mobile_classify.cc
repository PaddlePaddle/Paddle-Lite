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
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api;  // NOLINT

void load_labels(std::string path, std::vector<std::string>* labels) {
  FILE* fp = fopen(path.c_str(), "r");
  if (fp == nullptr) {
    printf("load label file failed \n");
    return;
  }
  while (!feof(fp)) {
    char str[1024];
    fgets(str, 1024, fp);
    std::string str_s(str);

    if (str_s.length() > 0) {
      for (int i = 0; i < str_s.length(); i++) {
        if (str_s[i] == ' ') {
          std::string strr = str_s.substr(i, str_s.length() - i - 1);
          labels->push_back(strr);
          i = str_s.length();
        }
      }
    }
  }
  fclose(fp);
}

void print_topk(const float* scores,
                const int size,
                const int topk,
                const std::vector<std::string>& labels) {
  std::vector<std::pair<float, int>> vec;
  vec.resize(size);
  for (int i = 0; i < size; i++) {
    vec[i] = std::make_pair(scores[i], i);
  }

  std::partial_sort(vec.begin(),
                    vec.begin() + topk,
                    vec.end(),
                    std::greater<std::pair<float, int>>());

  // print topk and score
  for (int i = 0; i < topk; i++) {
    float score = vec[i].first;
    int index = vec[i].second;
    printf("i: %d, index: %d, name: %s, score: %f \n",
           i,
           index,
           labels[index].c_str(),
           score);
  }
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

void pre_process(const cv::Mat& img,
                 int width,
                 int height,
                 Tensor dstTensor,
                 float* means,
                 float* scales) {
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  float* data = dstTensor.mutable_data<float>();
  neon_mean_scale(dimg, data, width * height, means, scales);
}

void RunModel(std::string model_file,
              std::string img_path,
              const std::vector<std::string>& labels,
              const int topk,
              int width,
              int height) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, height, width});
  auto* data = input_tensor->mutable_data<float>();
  // read img and pre-process
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  //   pre_process(img, width, height, data);
  float means[3] = {0.485f, 0.456f, 0.406f};
  float scales[3] = {0.229f, 0.224f, 0.225f};
  pre_process(img, width, height, *input_tensor, means, scales);

  // 4. Run predictor
  predictor->Run();

  // 5. Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t cnt = 1;
  for (auto& i : shape_out) {
    cnt *= i;
  }
  print_topk(outptr, cnt, topk, labels);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " model_file image_path label_file\n";
    exit(1);
  }
  printf("parameter:  model_file, image_path and label_file are necessary \n");
  printf("parameter:  topk, input_width,  input_height, are optional \n");
  std::string model_file = argv[1];
  std::string img_path = argv[2];
  std::string label_file = argv[3];
  std::vector<std::string> labels;
  load_labels(label_file, &labels);
  int topk = 5;
  int height = 224;
  int width = 224;
  if (argc > 4) {
    topk = atoi(argv[4]);
  }
  if (argc > 6) {
    width = atoi(argv[5]);
    height = atoi(argv[6]);
  }

  RunModel(model_file, img_path, labels, topk, width, height);
  return 0;
}
