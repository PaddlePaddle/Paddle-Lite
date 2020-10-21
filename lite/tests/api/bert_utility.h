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
#include "lite/utils/cp_logging.h"
#include "lite/utils/io.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

template <class T = int64_t>
void ReadRawData(const std::string& input_data_dir,
                 std::vector<std::vector<T>>* input0,
                 std::vector<std::vector<T>>* input1,
                 std::vector<std::vector<T>>* input2,
                 std::vector<std::vector<T>>* input3,
                 std::vector<std::vector<int64_t>>* input_shapes) {
  auto lines = ReadLines(input_data_dir);
  for (auto line : lines) {
    std::vector<std::string> shape_and_data = Split(line, ";");
    std::vector<int64_t> input_shape =
        Split<int64_t>(Split(shape_and_data[0], ":")[0], " ");
    input_shapes->emplace_back(input_shape);

    std::vector<T> input0_data =
        Split<T>(Split(shape_and_data[0], ":")[1], " ");
    input0->emplace_back(input0_data);
    std::vector<T> input1_data =
        Split<T>(Split(shape_and_data[1], ":")[1], " ");
    input1->emplace_back(input1_data);
    std::vector<T> input2_data =
        Split<T>(Split(shape_and_data[2], ":")[1], " ");
    input2->emplace_back(input2_data);
    std::vector<T> input3_data =
        Split<T>(Split(shape_and_data[3], ":")[1], " ");
    input3->emplace_back(input3_data);
  }
}

void ReadMmdnnRawData(const std::string& input_data_dir,
                      std::vector<std::vector<int64_t>>* data,
                      std::vector<std::vector<uint64_t>>* lod) {
  auto lines = ReadLines(input_data_dir);
  for (auto line : lines) {
    std::vector<std::string> data_str = Split(line, ";");
    if (data->empty()) {
      for (size_t i = 1; i < data_str.size(); i++) {
        data->push_back(std::vector<int64_t>());
        lod->push_back({0});
      }
    }

    for (size_t i = 1; i < data_str.size(); i++) {
      std::vector<std::string> data_es = Split(data_str[i], " ");
      if (data_es.empty()) {
        data_es.push_back("128000");
      }
      for (auto e : data_es) {
        data->at(i - 1).push_back(std::stoll(e));
      }
      lod->at(i - 1).push_back(lod->at(i - 1).back() + data_es.size());
    }
  }
}

template <class T = int64_t>
void FillTensor(const std::shared_ptr<lite_api::PaddlePredictor>& predictor,
                int tensor_id,
                const std::vector<int64_t>& tensor_shape,
                const std::vector<T>& tensor_value,
                const std::vector<std::vector<uint64_t>> tensor_lod = {}) {
  auto tensor_x = predictor->GetInput(tensor_id);
  tensor_x->Resize(tensor_shape);
  int64_t tensor_size = 1;
  for (size_t i = 0; i < tensor_shape.size(); i++) {
    tensor_size *= tensor_shape[i];
  }
  CHECK_EQ(static_cast<size_t>(tensor_size), tensor_value.size());
  memcpy(tensor_x->mutable_data<T>(),
         tensor_value.data(),
         sizeof(T) * tensor_size);
  if (!tensor_lod.empty()) {
    tensor_x->SetLoD(tensor_lod);
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

float CalMmdnnOutAccuracy(const std::vector<float>& out,
                          const std::string& out_file) {
  std::string ref_out_str = ReadFile(out_file);
  std::vector<float> ref_out = Split<float>(ref_out_str, "\n");

  int right_num = 0;
  for (size_t i = 0; i < out.size(); i++) {
    if (std::fabs(out[i] - ref_out[i]) < 1e-3) {
      right_num++;
    }
  }

  return static_cast<float>(right_num) / static_cast<float>(out.size());
}

}  // namespace lite
}  // namespace paddle
