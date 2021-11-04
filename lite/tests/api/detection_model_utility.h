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

// Adapt v2.0 detection models
template <class T = float>
void SetDetectionInputV2(
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
    std::vector<int> input_shape,
    std::vector<T> raw_data,
    int input_size) {
  auto im_shape_tensor = predictor->GetInput(0);
  im_shape_tensor->Resize(std::vector<int64_t>({1, 2}));
  auto* im_shape_data = im_shape_tensor->mutable_data<T>();
  im_shape_data[0] = input_shape[2];
  im_shape_data[1] = input_shape[3];

  auto input_tensor = predictor->GetInput(1);
  input_tensor->Resize(
      std::vector<int64_t>(input_shape.begin(), input_shape.end()));
  auto* data = input_tensor->mutable_data<T>();
  if (raw_data.empty()) {
    for (int i = 0; i < input_size; i++) {
      data[i] = 0.f;
    }
  } else {
    memcpy(data, raw_data.data(), sizeof(T) * input_size);
  }

  auto scale_factor_tensor = predictor->GetInput(2);
  scale_factor_tensor->Resize(std::vector<int64_t>({1, 2}));
  auto* scale_factor_data = scale_factor_tensor->mutable_data<T>();
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
  auto* data = input_tensor->mutable_data<T>();
  if (raw_data.empty()) {
    for (int i = 0; i < input_size; i++) {
      data[i] = 0.f;
    }
  } else {
    memcpy(data, raw_data.data(), sizeof(T) * input_size);
  }
}

void TestSSDDetectionModel(std::string model_dir,
                           std::string data_dir,
                           int32_t batch,
                           int32_t channel,
                           int32_t height,
                           int32_t width,
                           int32_t iteration,
                           int32_t version,
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

  std::string raw_data_dir = data_dir + std::string("/raw_data");
  std::vector<int> input_shape{batch, channel, height, width};
  auto raw_data = ReadRawData(raw_data_dir, input_shape, iteration);

  int input_size = 1;
  for (auto i : input_shape) {
    input_size *= i;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    if (version == 1) {
      SetDetectionInputV1(predictor, input_shape, {}, input_size);
    } else if (version == 2) {
      SetDetectionInputV2(predictor, input_shape, {}, input_size);
    } else {
      LOG(INFO) << "Unsupported the input version!";
      return;
    }

    predictor->Run();
  }

  std::vector<std::vector<float>> out_rets;
  out_rets.resize(iteration);
  double cost_time = 0;
  for (size_t i = 0; i < raw_data.size(); ++i) {
    if (version == 1) {
      SetDetectionInputV1(predictor, input_shape, raw_data[i], input_size);
    } else if (version == 2) {
      SetDetectionInputV2(predictor, input_shape, raw_data[i], input_size);
    } else {
      LOG(INFO) << "Unsupported the input version!";
      return;
    }
    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_GT(output_shape[0], 0);
    ASSERT_EQ(output_shape[1], 6);

    int output_size = output_shape[0] * output_shape[1];
    out_rets[i].resize(output_size);
    memcpy(&(out_rets[i].at(0)), output_data, sizeof(float) * output_size);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", batch: " << batch
            << ", iteration: " << iteration << ", spend "
            << cost_time / iteration / 1000.0 << " ms in average.";
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
