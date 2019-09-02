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

#include "lite/api/paddle_api.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/io.h"
DEFINE_string(model_dir, "", "");

namespace paddle {
namespace lite_api {

TEST(CxxApi, run) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_preferred_place(Place{TARGET(kX86), PRECISION(kFloat)});
  config.set_valid_places({
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor->Run();

  auto output = predictor->GetOutput(0);
  auto* out = output->data<float>();
  LOG(INFO) << out[0];
  LOG(INFO) << out[1];

  EXPECT_NEAR(out[0], 50.2132, 1e-3);
  EXPECT_NEAR(out[1], -28.8729, 1e-3);

  predictor->SaveOptimizedModel(FLAGS_model_dir + ".opt2");
  predictor->SaveOptimizedModel(FLAGS_model_dir + ".opt2.naive",
                                LiteModelType::kNaiveBuffer);
}

// Demo1 for Mobile Devices :Load model from file and run
#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
TEST(LightApi, run) {
  lite_api::MobileConfig config;
  config.set_model_dir(FLAGS_model_dir + ".opt2.naive");

  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor->Run();

  auto output = predictor->GetOutput(0);
  auto* out = output->data<float>();
  LOG(INFO) << out[0];
  LOG(INFO) << out[1];

  EXPECT_NEAR(out[0], 50.2132, 1e-3);
  EXPECT_NEAR(out[1], -28.8729, 1e-3);
}

// Demo2 for Loading model from memory
TEST(MobileConfig, LoadfromMemory) {
  // Get naive buffer
  auto model_path = std::string(FLAGS_model_dir) + ".opt2.naive/__model__.nb";
  auto params_path = std::string(FLAGS_model_dir) + ".opt2.naive/param.nb";
  std::string model_buffer = lite::ReadFile(model_path);
  size_t size_model = model_buffer.length();
  std::string params_buffer = lite::ReadFile(params_path);
  size_t size_params = params_buffer.length();
  // set model buffer and run model
  lite_api::MobileConfig config;
  config.set_model_buffer(
      model_buffer.c_str(), size_model, params_buffer.c_str(), size_params);

  auto predictor = lite_api::CreatePaddlePredictor(config);
  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor->Run();

  const auto output = predictor->GetOutput(0);
  const float* raw_output = output->data<float>();

  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "out " << raw_output[i];
  }
}

#endif

}  // namespace lite_api
}  // namespace paddle
