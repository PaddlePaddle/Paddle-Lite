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
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

void TestModel(const std::vector<Place>& valid_places, bool use_npu = false) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(FLAGS_model_dir, "", "", valid_places);

  auto* init_scores = predictor.GetInput(2);
  init_scores->Resize(DDim(std::vector<DDim::value_type>({1, 1})));
  auto* data_scores = init_scores->mutable_data<float>();
  auto scores_size = init_scores->dims().production();
  for (int i = 0; i < scores_size; i++) {
    data_scores[i] = 0;
  }
  auto lod_scores = init_scores->mutable_lod();
  std::vector<std::vector<uint64_t>> lod_s{{0, 1}, {0, 1}};
  *lod_scores = lod_s;

  auto* init_ids = predictor.GetInput(1);
  init_ids->Resize(DDim(std::vector<DDim::value_type>({1, 1})));
  auto* data_ids = init_ids->mutable_data<int64_t>();
  auto ids_size = init_ids->dims().production();
  for (int i = 0; i < ids_size; i++) {
    data_ids[i] = 0;
  }
  auto lod_ids = init_ids->mutable_lod();
  std::vector<std::vector<uint64_t>> lod_i{{0, 1}, {0, 1}};
  *lod_ids = lod_i;

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 1, 48, 512})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
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

  //  std::vector<std::vector<float>> results;
  //  // i = 1
  //  results.emplace_back(std::vector<float>(
  //      {0.00019130898, 9.467885e-05,  0.00015971427, 0.0003650665,
  //       0.00026431272, 0.00060884043, 0.0002107942,  0.0015819625,
  //       0.0010323516,  0.00010079765, 0.00011006987, 0.0017364529,
  //       0.0048292773,  0.0013995157,  0.0018453331,  0.0002428986,
  //       0.00020211363, 0.00013668182, 0.0005855956,  0.00025901722}));
  //  auto* out = predictor.GetOutput(0);
  //  ASSERT_EQ(out->dims().size(), 2);
  //  ASSERT_EQ(out->dims()[0], 1);
  //  ASSERT_EQ(out->dims()[1], 1000);
  //
  //  int step = 50;
  //  for (int i = 0; i < results.size(); ++i) {
  //    for (int j = 0; j < results[i].size(); ++j) {
  //      EXPECT_NEAR(out->data<float>()[j * step + (out->dims()[1] * i)],
  //                  results[i][j],
  //                  1e-6);
  //    }
  //  }
}

TEST(OcrAttention, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kInt64)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places);
}

}  // namespace lite
}  // namespace paddle
