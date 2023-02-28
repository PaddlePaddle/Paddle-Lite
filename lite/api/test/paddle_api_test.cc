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
#include "lite/utils/io.h"
#include "lite/utils/log/cp_logging.h"

DEFINE_string(model_dir, "", "");

namespace paddle {
namespace lite_api {

TEST(CxxApi, run) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  auto predictor = lite_api::CreatePaddlePredictor(config);

  LOG(INFO) << "Version: " << predictor->GetVersion();

  auto inputs = predictor->GetInputNames();
  LOG(INFO) << "input size: " << inputs.size();
  for (size_t i = 0; i < inputs.size(); i++) {
    LOG(INFO) << "inputnames: " << inputs[i];
  }
  auto outputs = predictor->GetOutputNames();
  for (size_t i = 0; i < outputs.size(); i++) {
    LOG(INFO) << "outputnames: " << outputs[i];
  }
  auto input_tensor = predictor->GetInputByName(inputs[0]);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor->Run();

  predictor->TryShrinkMemory();
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor->Run();
  auto output = predictor->GetTensor(outputs[0]);
  auto* out = output->data<float>();
  LOG(INFO) << out[0];
  LOG(INFO) << out[1];

  EXPECT_NEAR(out[0], 50.2132, 1e-3);
  EXPECT_NEAR(out[1], -28.8729, 1e-3);

  predictor->SaveOptimizedModel(FLAGS_model_dir + ".opt2");
  predictor->SaveOptimizedModel(
      FLAGS_model_dir + ".opt2.naive", LiteModelType::kNaiveBuffer, true);
}

TEST(CxxApi, share_external_data) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto inputs = predictor->GetInputNames();
  auto outputs = predictor->GetOutputNames();

  std::vector<float> external_data(100 * 100);
  for (int i = 0; i < 100 * 100; i++) {
    external_data[i] = i;
  }

  auto input_tensor = predictor->GetInputByName(inputs[0]);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  size_t memory_size = 100 * 100 * sizeof(float);
  input_tensor->ShareExternalMemory(static_cast<void*>(external_data.data()),
                                    memory_size,
                                    config.valid_places()[0].target);

  predictor->Run();

  auto output = predictor->GetTensor(outputs[0]);
  auto* out = output->data<float>();
  LOG(INFO) << out[0];
  LOG(INFO) << out[1];

  EXPECT_NEAR(out[0], 50.2132, 1e-3);
  EXPECT_NEAR(out[1], -28.8729, 1e-3);
}

// Demo1 for Mobile Devices :Load model from file and run
#ifdef LITE_WITH_ARM
TEST(LightApi, run) {
  lite_api::MobileConfig config;
  LOG(INFO) << "This devices support fp16 instruction: "
            << config.check_fp16_valid();
  config.set_model_from_file(FLAGS_model_dir + ".opt2.naive.nb");
  // disable L3 cache on workspace_ allocating
  config.SetArmL3CacheSize(L3CacheSetMethod::kDeviceL2Cache);
  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto inputs = predictor->GetInputNames();
  LOG(INFO) << "input size: " << inputs.size();
  for (int i = 0; i < inputs.size(); i++) {
    LOG(INFO) << "inputnames: " << inputs.at(i);
  }
  auto outputs = predictor->GetOutputNames();
  for (int i = 0; i < outputs.size(); i++) {
    LOG(INFO) << "outputnames: " << outputs.at(i);
  }

  LOG(INFO) << "Version: " << predictor->GetVersion();

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor->Run();

  predictor->TryShrinkMemory();
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  data = input_tensor->mutable_data<float>();
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
  auto model_file = std::string(FLAGS_model_dir) + ".opt2.naive.nb";
  std::string model_buffer = lite::ReadFile(model_file);
  // set model buffer and run model
  lite_api::MobileConfig config;
  LOG(INFO) << "This devices support fp16 instruction: "
            << config.check_fp16_valid();
  config.set_model_from_buffer(model_buffer);
  // allocate 1M initial space for workspace_
  config.SetArmL3CacheSize(L3CacheSetMethod::kAbsolute, 1024 * 1024);

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
