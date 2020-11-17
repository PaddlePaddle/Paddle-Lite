// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "test_helper.h"  // NOLINT

#include <sys/time.h>
#include <time.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

int64_t ShapeProduction(const std::vector<int64_t>& shape) {
  int64_t num = 1;
  for (auto i : shape) {
    num *= i;
  }
  return num;
}

std::vector<int64_t> GetIntNumsFromStr(const std::string& str) {
  std::vector<int64_t> nums;
  std::string tmp_str = str;
  while (!tmp_str.empty()) {
    int num = atoi(tmp_str.data());
    nums.push_back(num);
    size_t next_offset = tmp_str.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return nums;
}

std::vector<double> GetDoubleNumsFromStr(const std::string& str) {
  std::vector<double> nums;
  std::string tmp_str = str;
  while (!tmp_str.empty()) {
    double num = atof(tmp_str.data());
    nums.push_back(num);
    size_t next_offset = tmp_str.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return nums;
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
    *(dout_c0++) = (*(din++) - mean[0]) / scale[0];
    *(dout_c0++) = (*(din++) - mean[1]) / scale[1];
    *(dout_c0++) = (*(din++) - mean[2]) / scale[2];
  }
}

// Process img and set it as input
void process_img(const cv::Mat& img,
                 int width,
                 int height,
                 float* dest_data,
                 float* means,
                 float* scales) {
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  neon_mean_scale(dimg, dest_data, width * height, means, scales);
}
