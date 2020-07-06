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

#include "pre_process.h"  //NOLINT
#include <time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#define MAX_DICT_LENGTH 6624

double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

// fill tensor with mean and scale and trans
// layout: nhwc -> nchw, neon speed up
void NeonMeanScale(const float* din,
                   float* dout,
                   int size,
                   const std::vector<float>& mean,
                   const std::vector<float>& scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3\n";
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

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
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}

// resize image to a size multiple of 32 which is required by the network
cv::Mat DetResizeImg(const cv::Mat& img,
                     int max_size_len,
                     float* ratio_h,
                     float* ratio_w) {
  const int len = 32;
  int w = img.cols;
  int h = img.rows;
  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
    } else {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
    }
  }

  int resize_h = static_cast<int>(h * ratio);
  int resize_w = static_cast<int>(w * ratio);
  if (resize_h % len == 0) {
    resize_h = resize_h;
  } else if (resize_h / len < 1) {
    resize_h = len;
  } else {
    resize_h = (resize_h / len - 1) * len;
  }
  if (resize_w % len == 0) {
    resize_w = resize_w;
  } else if (resize_w / len < 1) {
    resize_w = len;
  } else {
    resize_w = (resize_w / len - 1) * len;
  }

  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  *ratio_h = static_cast<float>(resize_h) / static_cast<float>(h);
  *ratio_w = static_cast<float>(resize_w) / static_cast<float>(w);
  return resize_img;
}

cv::Mat CrnnResizeImg(const cv::Mat& img, float wh_ratio) {
  const int len = 32;
  const std::vector<int> rec_image_shape = {3, 32, 320};
  int img_height = rec_image_shape[1];
  int img_width = static_cast<int>(len * wh_ratio);
  float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);

  int resize_width = 0;
  if (ceilf(img_height * ratio) > img_width) {
    resize_width = img_width;
  } else {
    resize_width = static_cast<int>(ceilf(img_height * ratio));
  }

  cv::Mat resize_img;
  cv::resize(img,
             resize_img,
             cv::Size(resize_width, img_height),
             0.f,
             0.f,
             cv::INTER_LINEAR);
  return resize_img;
}

std::vector<std::string> ReadDict(const std::string& path) {
  std::vector<std::string> charactors(MAX_DICT_LENGTH);
  std::ifstream ifs;
  ifs.open(path);
  if (!ifs.is_open()) {
    std::cerr << "open file " << path << " failed" << std::endl;
  } else {
    std::string con = "";
    int count = 0;
    while (ifs) {
      getline(ifs, charactors[count]);
      count++;
    }
  }
  return charactors;
}

cv::Mat GetRotateCropImage(const cv::Mat& src_image,
                           const std::vector<std::vector<int>>& box) {
  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  int left = *std::min_element(x_collect, x_collect + 4);
  int right = *std::max_element(x_collect, x_collect + 4);
  int top = *std::min_element(y_collect, y_collect + 4);
  int bottom = *std::max_element(y_collect, y_collect + 4);
  auto crop_rect = cv::Rect(left, top, right - left, bottom - top);

  cv::Mat img_crop;
  src_image(crop_rect).copyTo(img_crop);

  std::vector<std::vector<int>> points = box;
  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
    points[i][1] -= top;
  }
  int img_crop_width =
      static_cast<int>(sqrt(pow(points[0][0] - points[1][0], 2) +
                            pow(points[0][1] - points[1][1], 2)));
  int img_crop_height =
      static_cast<int>(sqrt(pow(points[0][0] - points[3][0], 2) +
                            pow(points[0][1] - points[3][1], 2)));
  cv::Point2f pts_std[4];
  cv::Point2f point_sf[4];
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);
  point_sf[0] = cv::Point2f(points[0][0], points[0][1]);
  point_sf[1] = cv::Point2f(points[1][0], points[1][1]);
  point_sf[2] = cv::Point2f(points[2][0], points[2][1]);
  point_sf[3] = cv::Point2f(points[3][0], points[3][1]);

  cv::Mat dst_img;
  cv::Mat trans_mat = cv::getPerspectiveTransform(point_sf, pts_std);
  cv::warpPerspective(img_crop,
                      dst_img,
                      trans_mat,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);

  const float ratio = 1.5;
  if (static_cast<float>(dst_img.rows) >=
      static_cast<float>(dst_img.cols) * ratio) {
    cv::Mat res_img = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, res_img);
    cv::flip(res_img, res_img, 0);
    return res_img;
  } else {
    return dst_img;
  }
}
