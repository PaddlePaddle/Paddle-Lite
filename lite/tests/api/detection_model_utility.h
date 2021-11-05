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
#include <gflags/gflags.h>
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

void LoadData(
    const std::string& data_path,
    std::vector<std::vector<std::vector<uint8_t>>>* data_set,
    std::vector<std::vector<std::vector<int64_t>>>* data_set_shapes,
    const std::shared_ptr<paddle::lite_api::PaddlePredictor>& predictor,
    const std::string& data_type = "input") {
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
        LOG(FATAL) << "Load data failed! Data precision is Unsupported!";
      }
    }
    data_set->push_back(input_tensor_datas);
    data_set_shapes->push_back(input_tensor_shapes);
  }
}

template <typename T>
void fill_tensor(
    const std::shared_ptr<paddle::lite_api::PaddlePredictor>& predictor,
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

void FillModelInput(
    std::vector<std::vector<uint8_t>> input_data,
    const std::vector<std::vector<int64_t>> input_shape,
    const std::shared_ptr<paddle::lite_api::PaddlePredictor>& predictor) {
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
          << "FillModelInput func not supoort this precision, please add!";
    }
  }
}

void TestDetectionModel(std::string model_dir,
                        std::string data_dir,
                        std::string data_type,
                        int32_t model_version,
                        std::vector<std::string> nnadapter_device_names,
                        std::string nnadapter_context_properties,
                        std::vector<paddle::lite_api::Place> valid_places) {
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  // Use the full api with CxxConfig to generate the optimized model
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(model_dir);
  cxx_config.set_valid_places(valid_places);
  cxx_config.set_nnadapter_device_names(nnadapter_device_names);
  cxx_config.set_nnadapter_context_properties(nnadapter_context_properties);
  predictor = lite_api::CreatePaddlePredictor(cxx_config);
  predictor->SaveOptimizedModel(model_dir,
                                paddle::lite_api::LiteModelType::kNaiveBuffer);
  // Use the light api with MobileConfig to load and run the optimized model
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_dir + ".nb");
  mobile_config.set_threads(FLAGS_threads);
  mobile_config.set_power_mode(
      static_cast<lite_api::PowerMode>(FLAGS_power_mode));
  mobile_config.set_nnadapter_device_names(nnadapter_device_names);
  mobile_config.set_nnadapter_context_properties(nnadapter_context_properties);
  predictor = paddle::lite_api::CreatePaddlePredictor(mobile_config);

  // Load input_data
  std::vector<std::vector<std::vector<uint8_t>>> input_data_set;
  std::vector<std::vector<std::vector<int64_t>>> input_data_set_shapes;
  std::string input_path;
  if (data_type == "ssd_voc") {
    if (model_version == 1) {
      input_path = data_dir + "/ssd_data_v1.txt";
    } else if (model_version == 2) {
      input_path = data_dir + "/ssd_data_v2.txt";
    } else {
      LOG(FATAL) << "Unsupported model version: " << model_version;
    }
  } else if (data_type == "yolo_coco") {
    if (model_version == 1) {
      input_path = data_dir + "/yolo_data_v1.txt";
    } else if (model_version == 2) {
      input_path = data_dir + "/yolo_data_v2.txt";
    } else {
      LOG(FATAL) << "Unsupported model version: " << model_version;
    }
  } else {
    LOG(FATAL) << "Unsupported data type: " << data_type;
  }

  LoadData(
      input_path, &input_data_set, &input_data_set_shapes, predictor, "input");
  int32_t iteration = input_data_set.size();
  // Warm up
  for (int i = 0; i < 1; i++) {
    FillModelInput(input_data_set[i], input_data_set_shapes[i], predictor);
    predictor->Run();
  }

  double cost_time = 0;
  for (size_t i = 0; i < iteration; ++i) {
    FillModelInput(input_data_set[i], input_data_set_shapes[i], predictor);
    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_GT(output_shape[0], 0);
    ASSERT_EQ(output_shape[1], 6);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir << ", threads num " << FLAGS_threads
            << ", iteration: " << iteration << ", spend "
            << cost_time / iteration / 1000.0 << " ms in average.";
}

}  // namespace lite
}  // namespace paddle
