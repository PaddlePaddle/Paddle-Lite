// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/api/light_api.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>

DEFINE_string(optimized_model, "", "");

namespace paddle {
namespace lite {

TEST(LightAPI, load) {
  if (FLAGS_optimized_model.empty()) {
    FLAGS_optimized_model = "lite_naive_model";
  }
  LightPredictor predictor(FLAGS_optimized_model, "", "");
  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<int64_t>({100, 100})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor.PrepareFeedFetch();
  const std::vector<std::string> inputs = predictor.GetInputNames();

  LOG(INFO) << "input size: " << inputs.size();
  for (size_t i = 0; i < inputs.size(); i++) {
    LOG(INFO) << "inputnames: " << inputs[i];
  }
  const std::vector<std::string> outputs = predictor.GetOutputNames();
  for (size_t i = 0; i < outputs.size(); i++) {
    LOG(INFO) << "outputnames: " << outputs[i];
  }

  auto& precisions = predictor.GetInputPrecisions();
  ASSERT_EQ(precisions.size(), 1);
  ASSERT_EQ(precisions[0], PrecisionType::kFloat);

  predictor.Run();

  const auto* output = predictor.GetOutput(0);
  const float* raw_output = output->data<float>();

  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "out " << raw_output[i];
  }
}

TEST(LightAPI, loadNaiveBuffer) {
  if (FLAGS_optimized_model.empty()) {
    FLAGS_optimized_model = "lite_naive_model";
  }

  auto model_path = std::string(FLAGS_optimized_model) + "/__model__.nb";
  auto params_path = std::string(FLAGS_optimized_model) + "/param.nb";
  std::string model_buffer = lite::ReadFile(model_path);
  size_t size_model = model_buffer.length();
  std::string params_buffer = lite::ReadFile(params_path);
  size_t size_params = params_buffer.length();
  LOG(INFO) << "sizeModel: " << size_model;
  LOG(INFO) << "sizeParams: " << size_params;

  lite_api::MobileConfig config;
  config.set_model_buffer(
      model_buffer.c_str(), size_model, params_buffer.c_str(), size_params);
  LightPredictor predictor(config.model_dir(),
                           config.model_buffer(),
                           config.param_buffer(),
                           config.is_model_from_memory(),
                           lite_api::LiteModelType::kNaiveBuffer);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<int64_t>({100, 100})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor.Run();

  const auto* output = predictor.GetOutput(0);
  const float* raw_output = output->data<float>();

  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "out " << raw_output[i];
  }
}

}  // namespace lite
}  // namespace paddle
