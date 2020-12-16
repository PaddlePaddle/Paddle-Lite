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
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

void ReadTxtFile(const std::string& file_path, float* dest, int num) {
  CHECK(!file_path.empty());
  CHECK(dest != nullptr);
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    LOG(FATAL) << "open file error:" << file_path;
  }
  for (int i = 0; i < num; i++) {
    ifs >> dest[i];
  }
}

int64_t ShapeProduction(const std::vector<int64_t>& shape) {
  int64_t num = 1;
  for (auto i : shape) {
    num *= i;
  }
  return num;
}

void TestClassificationModel(
    const std::string& model_dir,
    const std::string& model_file,
    const std::string& params_file,
    const std::string& data_dir,
    int max_index_gt,
    float max_value_gt,
    float eps = 1e-2,
    int batch_size = 1,
    int channel = 3,
    int height = 224,
    int width = 224,
    const std::vector<lite_api::Place>& valid_places = {lite_api::Place{
        TARGET(kARM), PRECISION(kFloat)}},
    int threads = 1,
    int power_mode = 3 /* no bind */) {
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(model_dir);
  cxx_config.set_model_file(model_file);
  cxx_config.set_param_file(params_file);
  cxx_config.set_valid_places(valid_places);
  auto predictor = lite_api::CreatePaddlePredictor(cxx_config);
  predictor->SaveOptimizedModel(model_dir,
                                lite_api::LiteModelType::kNaiveBuffer);

  lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_dir + ".nb");
  mobile_config.set_threads(threads);
  mobile_config.set_power_mode(static_cast<lite_api::PowerMode>(power_mode));
  predictor = lite_api::CreatePaddlePredictor(mobile_config);

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize({batch_size, channel, height, width});
  auto* data = input_tensor->mutable_data<float>();
  int input_size = batch_size * channel * height * width;
  ReadTxtFile(data_dir, data, input_size);

  predictor->Run();

  auto out_tensor = predictor->GetOutput(0);
  int64_t out_size = ShapeProduction(out_tensor->shape());
  auto* out_data = out_tensor->data<float>();
  auto max_iter = std::max_element(out_data, out_data + out_size);
  float max_value = *max_iter;
  int max_index = max_iter - out_data;

  ASSERT_EQ(out_size, 1000);
  ASSERT_EQ(max_index, max_index_gt);
  float error =
      (std::abs(max_value) - std::abs(max_value_gt)) / std::abs(max_value_gt);
  ASSERT_LT(error, eps);
  LOG(INFO) << "output max index:" << max_index;
  LOG(INFO) << "output max value:" << max_value;
}

}  // namespace lite
}  // namespace paddle
