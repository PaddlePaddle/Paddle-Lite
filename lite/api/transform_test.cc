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

void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place,
               bool use_npu = false) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(FLAGS_model_dir, preferred_place, valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 256, 1})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 0;
  }
  auto lod_input = input_tensor->mutable_lod();
  std::vector<std::vector<uint64_t>> lod_s{{0, 1}, {0, 1}};
  *lod_input = lod_s;

  auto* src_pos = predictor.GetInput(1);
  src_pos->Resize(DDim(std::vector<DDim::value_type>({1, 256, 1})));
  auto* src_data = src_pos->mutable_data<float>();
  item_size = src_pos->dims().production();
  for (int i = 0; i < item_size; i++) {
    src_data[i] = 0;
  }
  auto lod_src_pos = src_pos->mutable_lod();
  *lod_src_pos = lod_s;

  auto* attn_bias = predictor.GetInput(2);
  attn_bias->Resize(DDim(std::vector<DDim::value_type>({1, 8, 256, 256})));
  auto* attn_bias_data = attn_bias->mutable_data<float>();
  item_size = attn_bias->dims().production();
  for (int i = 0; i < item_size; i++) {
    attn_bias_data[i] = 0;
  }
  auto lod_attn_bias = attn_bias->mutable_lod();
  *lod_attn_bias = lod_s;

  auto* trg_word = predictor.GetInput(3);
  trg_word->Resize(DDim(std::vector<DDim::value_type>({1, 1, 1})));
  auto* trg_word_data = trg_word->mutable_data<float>();
  item_size = trg_word->dims().production();
  for (int i = 0; i < item_size; i++) {
    trg_word_data[i] = 0;
  }
  auto lod_trg = trg_word->mutable_lod();
  *lod_trg = lod_s;

  auto* init_scores = predictor.GetInput(4);
  init_scores->Resize(DDim(std::vector<DDim::value_type>({1, 1})));
  auto* data_scores = init_scores->mutable_data<float>();
  auto scores_size = input_tensor->dims().production();
  for (int i = 0; i < scores_size; i++) {
    data_scores[i] = 0;
  }
  auto lod_scores = init_scores->mutable_lod();
  *lod_scores = lod_s;

  auto* init_ids = predictor.GetInput(5);
  init_ids->Resize(DDim(std::vector<DDim::value_type>({1})));
  auto* data_ids = init_ids->mutable_data<float>();
  auto ids_size = init_ids->dims().production();
  for (int i = 0; i < ids_size; i++) {
    data_ids[i] = 0;
  }
  auto lod_ids = init_ids->mutable_lod();
  *lod_ids = lod_s;

  auto* trg_bias = predictor.GetInput(6);
  trg_bias->Resize(DDim(std::vector<DDim::value_type>({1, 8, 1, 256})));
  auto* trg_bias_data = trg_bias->mutable_data<float>();
  item_size = trg_bias->dims().production();
  for (int i = 0; i < item_size; i++) {
    trg_bias_data[i] = 0;
  }
  auto lod_trg_bias = trg_bias->mutable_lod();
  *lod_trg_bias = lod_s;

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

  auto* outs = predictor.GetOutputs();
  for (auto out : *outs) {
    LOG(INFO) << out;
  }
}

TEST(OcrAttention, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kARM), PRECISION(kFloat)}));
}

}  // namespace lite
}  // namespace paddle
