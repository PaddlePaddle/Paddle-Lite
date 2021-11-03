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
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/utils/io.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

// Adapt v2.0 detection models
template <class T = float>
void SetDetectionInputV2(
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
    std::vector<int> input_shape,
    std::vector<T> raw_data,
    int input_size) {
  auto im_shape_tensor = predictor->GetInput(0);
  auto* im_shape_data = im_shape_tensor->mutable_data<float>();
  im_shape_data[0] = input_shape[2];
  im_shape_data[1] = input_shape[3];

  auto input_tensor = predictor->GetInput(1);
  input_tensor->Resize(
      std::vector<int64_t>(input_shape.begin(), input_shape.end()));
  auto* data = input_tensor->mutable_data<float>();
  if (raw_data.empty()) {
    for (int i = 0; i < input_size; i++) {
      data[i] = 0.f;
    }
  } else {
    memcpy(data, raw_data.data(), sizeof(float) * input_size);
  }

  auto scale_factor_tensor = predictor->GetInput(2);
  auto* scale_factor_data = scale_factor_tensor->mutable_data<float>();
  scale_factor_data[0] = 1;
  scale_factor_data[1] = 1;
}

// Adapt v1.0 detection models
template <class T = float>
void SetDetectionInputV1(
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
    std::vector<int> input_shape,
    std::vector<T> raw_data,
    int input_size) {
  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(
      std::vector<int64_t>(input_shape.begin(), input_shape.end()));
  auto* data = input_tensor->mutable_data<float>();
  if (raw_data.empty()) {
    for (int i = 0; i < input_size; i++) {
      data[i] = 0.f;
    }
  } else {
    memcpy(data, raw_data.data(), sizeof(float) * input_size);
  }
}

float CalBertOutAccuracy(const std::vector<std::vector<float>>& out,
                         const std::string& out_file) {
  auto lines = ReadLines(out_file);
  std::vector<std::vector<float>> ref_out;
  for (auto line : lines) {
    ref_out.emplace_back(Split<float>(line, " "));
  }

  int right_num = 0;
  for (size_t i = 0; i < out.size(); i++) {
    std::vector<size_t> out_index{0, 1, 2};
    std::vector<size_t> ref_out_index{0, 1, 2};

    std::sort(out_index.begin(),
              out_index.end(),
              [&out, i](size_t a, size_t b) { return out[i][a] > out[i][b]; });
    std::sort(ref_out_index.begin(),
              ref_out_index.end(),
              [&ref_out, i](size_t a, size_t b) {
                return ref_out[i][a] > ref_out[i][b];
              });
    right_num += (out_index == ref_out_index);
  }

  return static_cast<float>(right_num) / static_cast<float>(out.size());
}

float CalErnieOutAccuracy(const std::vector<std::vector<float>>& out,
                          const std::string& out_file) {
  auto lines = ReadLines(out_file);
  std::vector<std::vector<float>> ref_out;
  for (auto line : lines) {
    ref_out.emplace_back(Split<float>(line, " "));
  }

  int right_num = 0;
  for (size_t i = 0; i < out.size(); i++) {
    right_num += (std::fabs(out[i][0] - ref_out[i][0]) < 0.01f);
  }

  return static_cast<float>(right_num) / static_cast<float>(out.size());
}

}  // namespace lite
}  // namespace paddle
