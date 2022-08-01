// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "lite/api/tools/benchmark/precision_evaluation/utils.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite_api {

const std::string GetAbsPath(const std::string file_name) {
  char abs_path_buff[PATH_MAX];
  if (realpath(file_name.c_str(), abs_path_buff)) {
    return std::string(abs_path_buff);
  } else {
    std::cerr << "Get abs path error!" << std::endl;
    std::abort();
  }
}

const std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cerr << "Failed to open file " << path << std::endl;
    std::abort();
  }
  return m_vec;
}

const std::map<std::string, std::string> LoadConfigTxt(
    std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, std::string> dict;
  for (auto i = 0; i < config.size(); i++) {
    std::vector<std::string> res = lite::Split(config[i], ":");
    std::cout << "key: " << res[0] << "\t value: " << res[1] << std::endl;
    dict[res[0]] = res[1];
  }
  return dict;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  std::cout << "======= Configuration for Precision Benchmark ======"
            << std::endl;
  for (auto iter = config.begin(); iter != config.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << std::endl;
}

cv::Mat ResizeImage(const cv::Mat &img, const int resize_short_size) {
  int w = img.cols;
  int h = img.rows;

  cv::Mat resize_img;

  float ratio = 1.f;
  if (h < w) {
    ratio = resize_short_size / static_cast<float>(h);
  } else {
    ratio = resize_short_size / static_cast<float>(w);
  }
  int resize_h = round(h * ratio);
  int resize_w = round(w * ratio);

  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  return resize_img;
}

cv::Mat CenterCropImg(const cv::Mat &img, const int crop_size) {
  int resize_w = img.cols;
  int resize_h = img.rows;
  int w_start = static_cast<int>((resize_w - crop_size) / 2);
  int h_start = static_cast<int>((resize_h - crop_size) / 2);
  cv::Rect rect(w_start, h_start, crop_size, crop_size);
  cv::Mat crop_img = img(rect);
  return crop_img;
}

// Fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void NeonMeanScale(const float *din,
                   float *dout,
                   int size,
                   const std::vector<float> mean,
                   const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3!" << std::endl;
    std::abort();
  }

  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

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
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}

}  // namespace lite_api
}  // namespace paddle
