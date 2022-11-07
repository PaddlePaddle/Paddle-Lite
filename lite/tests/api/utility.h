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

#pragma once
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

template <class T = float>
T CalOutAccuracy(const std::vector<std::vector<T>>& out,
                 const std::vector<std::vector<T>>& ref_out,
                 const float abs_error = 1e-5) {
  size_t right_count = 0;
  size_t all_count = 0;
  for (size_t i = 0; i < out.size(); i++) {
    CHECK_EQ(out[i].size(), ref_out[i].size()) << "Size error, i: " << i;
    for (size_t j = 0; j < out[i].size(); j++) {
      if (std::abs(out[i][j] - ref_out[i][j]) < abs_error) {
        right_count++;
      }
      all_count++;
    }
  }
  return static_cast<float>(right_count) / static_cast<float>(all_count);
}

template <class T = float>
T CalOutAccuracy(const std::vector<std::vector<T>>& error_result,
                 const float abs_error = 1e-5) {
  size_t right_count = 0;
  size_t all_count = 0;
  for (size_t i = 0; i < error_result.size(); i++) {
    for (size_t j = 0; j < error_result[i].size(); j++) {
      if (error_result[i][j] < abs_error) {
        right_count++;
      }
      all_count++;
    }
  }
  return static_cast<float>(right_count) / static_cast<float>(all_count);
}

template <typename T>
void fill_tensor(std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
                 const int idx,
                 const T* data,
                 const std::vector<int64_t>& shape) {
  auto tensor = predictor->GetInput(idx);
  tensor->Resize(shape);
  auto tensor_data = tensor->mutable_data<T>();
  int64_t size = 1;
  for (auto i : shape) size *= i;
  memcpy(tensor_data, data, sizeof(T) * size);
}

template <class T = float>
void SetDetectionInput(
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
    std::vector<int> input_shape,
    std::vector<T> raw_data,
    int input_size) {
  auto input_names = predictor->GetInputNames();
  int batch_size = input_shape[0];
  int rh = input_shape[2];
  int rw = input_shape[3];
  for (const auto& tensor_name : input_names) {
    auto in_tensor = predictor->GetInputByName(tensor_name);
    if (tensor_name == "image") {
      in_tensor->Resize(
          std::vector<int64_t>(input_shape.begin(), input_shape.end()));
      auto* input_data = in_tensor->mutable_data<T>();
      if (raw_data.empty()) {
        for (int i = 0; i < input_size; i++) {
          input_data[i] = 0.f;
        }
      } else {
        memcpy(input_data, raw_data.data(), sizeof(T) * input_size);
      }
    } else if (tensor_name == "im_shape" || tensor_name == "im_size") {
      in_tensor->Resize({batch_size, 2});
      auto* im_shape_data = in_tensor->mutable_data<T>();
      for (int i = 0; i < batch_size * 2; i += 2) {
        im_shape_data[i] = rh;
        im_shape_data[i + 1] = rw;
      }
    } else if (tensor_name == "scale_factor") {
      in_tensor->Resize({batch_size, 2});
      auto* scale_factor_data = in_tensor->mutable_data<T>();
      for (int i = 0; i < batch_size * 2; i++) {
        scale_factor_data[i] = 1;
      }
    } else {
      LOG(FATAL) << "Unsupported the input: " << tensor_name;
    }
  }
}

std::vector<std::string> SplitString(const std::string& s,
                                     const std::string& seperator) {
  std::vector<std::string> result;
  typedef std::string::size_type string_size;
  string_size i = 0;

  while (i != s.size()) {
    int flag = 0;
    while (i != s.size() && flag == 0) {
      flag = 1;
      for (string_size x = 0; x < seperator.size(); ++x) {
        if (s[i] == seperator[x]) {
          ++i;
          flag = 0;
          break;
        }
      }
    }

    flag = 0;
    string_size j = i;
    while (j != s.size() && flag == 0) {
      for (string_size x = 0; x < seperator.size(); ++x) {
        if (s[j] == seperator[x]) {
          flag = 1;
          break;
        }
        if (flag == 0) ++j;
      }
    }
    if (i != j) {
      result.push_back(s.substr(i, j - i));
      i = j;
    }
  }
  return result;
}

void LoadSpecificData(
    const std::string data_path,
    std::vector<std::vector<std::vector<uint8_t>>>& data_set,         // NOLINT
    std::vector<std::vector<std::vector<int64_t>>>& data_set_shapes,  // NOLINT
    std::shared_ptr<paddle::lite_api::PaddlePredictor>& predictor,    // NOLINT
    std::string data_type = "input") {
  auto input_lines = ReadLines(data_path);
  for (auto line : input_lines) {
    std::vector<std::vector<int64_t>> input_tensor_shapes;
    std::vector<std::vector<uint8_t>> input_tensor_datas;
    std::vector<std::string> line_split_string = SplitString(line, ";");
    for (int32_t i = 0; i < line_split_string.size(); i++) {
      input_tensor_shapes.push_back(
          Split<int64_t>(Split(line_split_string[i], ":")[0], " "));
      auto input_tensor = data_type == "input" ? predictor->GetInput(i)
                                               : predictor->GetOutput(i);
      auto input_tensor_precision = input_tensor->precision();
      if (input_tensor_precision == PRECISION(kInt32)) {
        std::vector<int> origin_data =
            Split<int>(Split(line_split_string[i], ":")[1], " ");
        std::vector<uint8_t> bytes_data(origin_data.size() * sizeof(int));
        memcpy(reinterpret_cast<void*>(&bytes_data[0]),
               reinterpret_cast<void*>(&origin_data[0]),
               origin_data.size() * sizeof(int));
        input_tensor_datas.push_back(bytes_data);
      } else if (input_tensor_precision == PRECISION(kFloat)) {
        std::vector<float> origin_data =
            Split<float>(Split(line_split_string[i], ":")[1], " ");
        std::vector<uint8_t> bytes_data(origin_data.size() * sizeof(float));
        memcpy(reinterpret_cast<void*>(&bytes_data[0]),
               reinterpret_cast<void*>(&origin_data[0]),
               origin_data.size() * sizeof(float));
        input_tensor_datas.push_back(bytes_data);
      } else if (input_tensor_precision == PRECISION(kInt64)) {
        std::vector<int64_t> origin_data =
            Split<int64_t>(Split(line_split_string[i], ":")[1], " ");
        std::vector<uint8_t> bytes_data(origin_data.size() * sizeof(int64_t));
        memcpy(reinterpret_cast<void*>(&bytes_data[0]),
               reinterpret_cast<void*>(&origin_data[0]),
               origin_data.size() * sizeof(int64_t));
        input_tensor_datas.push_back(bytes_data);
      } else {
        LOG(FATAL) << "LoadSpecificData only support the precision is "
                      "float/int32/int64";
      }
    }
    data_set.push_back(input_tensor_datas);
    data_set_shapes.push_back(input_tensor_shapes);
  }
}

void FillModelInput(
    std::vector<std::vector<uint8_t>> input_data,                     // NOLINT
    const std::vector<std::vector<int64_t>> input_shape,              // NOLINT
    std::shared_ptr<paddle::lite_api::PaddlePredictor>& predictor) {  // NOLINT
  auto input_names = predictor->GetInputNames();
  for (int idx = 0; idx < input_names.size(); idx++) {
    auto tensor = predictor->GetInput(idx);
    auto input_size = 1;
    for (auto shape : input_shape[idx]) {
      input_size *= shape;
    }
    tensor->Resize(input_shape[idx]);
    auto precision = tensor->precision();
    if (precision == PRECISION(kFloat)) {
      auto tensor_data = tensor->mutable_data<float>();
      memcpy(tensor_data, input_data[idx].data(), input_data[idx].size());
    } else if (precision == PRECISION(kInt32)) {
      auto tensor_data = tensor->mutable_data<int32_t>();
      memcpy(tensor_data, input_data[idx].data(), input_data[idx].size());
    } else if (precision == PRECISION(kInt64)) {
      auto tensor_data = tensor->mutable_data<int64_t>();
      memcpy(tensor_data, input_data[idx].data(), input_data[idx].size());
    } else {
      LOG(FATAL)
          << " FillModelInput only support the precision is float/int32/int64";
    }
  }
}

void GetModelOutputAndAbsError(
    std::shared_ptr<paddle::lite_api::PaddlePredictor>& predictor,  // NOLINT
    const std::vector<std::vector<uint8_t>> golden_ouputs_data,     // NOLINT
    const std::vector<std::vector<int64_t>> golden_outputs_shape,   // NOLINT
    std::vector<float>& abs_error) {                                // NOLINT
  auto output_names = predictor->GetOutputNames();
  if (output_names.size() != golden_ouputs_data.size() ||
      output_names.size() != golden_outputs_shape.size()) {
    LOG(FATAL) << "The output num: " << output_names.size()
               << " is not equal to the golden output num:"
               << golden_ouputs_data.size() << "!";
  }
  for (int idx = 0; idx < output_names.size(); idx++) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
        std::move(predictor->GetOutput(idx)));
    int64_t output_size = 1;
    for (auto dim : output_tensor->shape()) {
      output_size *= dim;
    }
    auto precision = output_tensor->precision();
    if (precision == PRECISION(kFloat)) {
      const float* output_data = output_tensor->mutable_data<float>();
      const float* golden_ouput_data =
          reinterpret_cast<const float*>(&golden_ouputs_data[idx][0]);
      for (int i = 0; i < output_size; i++) {
        abs_error.push_back(std::abs(output_data[i] - golden_ouput_data[i]));
      }
    } else if (precision == PRECISION(kInt32)) {
      const int* output_data = output_tensor->mutable_data<int>();
      const int* golden_ouput_data =
          reinterpret_cast<const int*>(&golden_ouputs_data[idx][0]);
      for (int i = 0; i < output_size; i++) {
        abs_error.push_back(std::abs(output_data[i] - golden_ouput_data[i]));
      }
    } else if (precision == PRECISION(kInt64)) {
      const int64_t* output_data = output_tensor->mutable_data<int64_t>();
      const int64_t* golden_ouput_data =
          reinterpret_cast<const int64_t*>(&golden_ouputs_data[idx][0]);
      for (int i = 0; i < output_size; i++) {
        abs_error.push_back(std::abs(output_data[i] - golden_ouput_data[i]));
      }
    } else {
      LOG(FATAL)
          << "FillModelInput only support the precision is float/int32/int64";
    }
  }
}

}  // namespace lite
}  // namespace paddle
