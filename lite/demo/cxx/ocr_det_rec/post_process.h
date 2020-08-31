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
#pragma once

#include <math.h>
#include <iostream>
#include <vector>
#include "clipper.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using vector_3d_int = std::vector<std::vector<std::vector<int>>>;
using vector_2d_fp = std::vector<std::vector<float>>;

template <typename T>
T Clamp(T x, T min, T max) {
  if (x > max)
    return max;
  else if (x < min)
    return min;
  else
    return x;
}

vector_2d_fp Mat2Vec(const cv::Mat& mat);

float GetContourArea(const vector_2d_fp& box, float unclip_ratio);

cv::RotatedRect Unclip(const vector_2d_fp& box);

void QuickSort(vector_2d_fp* input, int left, int right);

void QuickSort(std::vector<std::vector<int>>* box,
               int left,
               int right,
               int axis);

std::vector<std::vector<int>> OrderPointsClockwise(
    std::vector<std::vector<int>> pts);

vector_2d_fp GetMiniBoxes(const cv::RotatedRect& box);

float BoxScoreFast(const vector_2d_fp& box_array, const cv::Mat& pred);

vector_3d_int BoxesFromBitmap(const cv::Mat& pred, const cv::Mat& bitmap);

vector_3d_int FilterTagDetRes(vector_3d_int boxes,
                              float ratio_h,
                              float ratio_w,
                              int img_width,
                              int img_height);
