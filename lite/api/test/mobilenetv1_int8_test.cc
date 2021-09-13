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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/test_helper.h"
#include "lite/core/op_registry.h"

DEFINE_string(input_img_txt_path,
              "",
              "if set input_img_txt_path, read the img filename as input.");

namespace paddle {
namespace lite {

void TestModel(const std::vector<Place>& valid_places) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(FLAGS_model_dir, "", "", valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  if (FLAGS_input_img_txt_path.empty()) {
    for (int i = 0; i < item_size; i++) {
      data[i] = 1;
    }
  } else {
    std::fstream fs(FLAGS_input_img_txt_path, std::ios::in);
    if (!fs.is_open()) {
      LOG(FATAL) << "open input_img_txt error.";
    }
    for (int i = 0; i < item_size; i++) {
      fs >> data[i];
    }
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  std::vector<std::vector<float>> results;
  // i = 1
  // ground truth result from fluid
  results.emplace_back(std::vector<float>(
      {0.0002451055, 0.0002585023, 0.0002659616, 0.0002823}));
  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 1000);

  int step = 50;
  for (int i = 0; i < results.size(); ++i) {
    for (int j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(out->data<float>()[j * step + (out->dims()[1] * i)],
                  results[i][j],
                  1e-6);
    }
  }

  auto* out_data = out->data<float>();
  LOG(INFO) << "output data:";
  for (int i = 0; i < out->numel(); i += step) {
    LOG(INFO) << out_data[i];
  }
  float max_val = out_data[0];
  int max_val_arg = 0;
  for (int i = 1; i < out->numel(); i++) {
    if (max_val < out_data[i]) {
      max_val = out_data[i];
      max_val_arg = i;
    }
  }
  LOG(INFO) << "max val:" << max_val << ", max_val_arg:" << max_val_arg;
}

TEST(MobileNetV1, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kInt8)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places);
}

}  // namespace lite
}  // namespace paddle
