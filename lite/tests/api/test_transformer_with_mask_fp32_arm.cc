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
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

template <typename T>
void SetTensorData(const std::vector<T> &data,
                   const std::vector<int64_t> &shape,
                   paddle::lite_api::Tensor *tensor,
                   const std::vector<std::vector<uint64_t>> &lod = {}) {
  tensor->Resize(shape);
  tensor->SetLoD(lod);
  std::copy(data.begin(), data.end(), tensor->mutable_data<T>());
}

void PrepareInputData(
    const std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor,
    std::vector<int64_t> src_word_data,
    int max_seq_len = 16,  // padding
    int max_out_len = 8,
    int bos_idx = 0,
    int eos_idx = 1,
    int n_head = 8) {
  // src_word
  auto src_word = predictor->GetInput(0);
  int seq_len = src_word_data.size();
  for (int i = seq_len; i < max_seq_len; i++) {
    src_word_data.push_back(eos_idx);
  }
  std::vector<int64_t> src_word_shape{
      1, static_cast<int64_t>(src_word_data.size())};
  SetTensorData<int64_t>(src_word_data, src_word_shape, src_word.get());
  // src_pos
  auto src_pos = predictor->GetInput(1);
  std::vector<int64_t> src_pos_data(src_word_data.size());
  std::iota(src_pos_data.begin(), src_pos_data.end(), 0);
  std::vector<int64_t> src_pos_shape{1,
                                     static_cast<int64_t>(src_pos_data.size())};
  SetTensorData<int64_t>(src_pos_data, src_pos_shape, src_pos.get());
  // src_slf_attn_bias
  auto src_slf_attn_bias = predictor->GetInput(2);
  std::vector<float> src_slf_attn_bias_data(1 * n_head * src_word_data.size() *
                                            src_word_data.size());
  int offset = 0;
  for (int j = 0; j < 1 * n_head * src_word_data.size(); j++) {
    for (int i = 0; i < seq_len; i++) {
      src_slf_attn_bias_data[offset++] = 0.0f;
    }
    for (int i = seq_len; i < src_word_data.size(); i++) {
      src_slf_attn_bias_data[offset++] = -1e9f;
    }
  }
  std::vector<int64_t> src_slf_attn_bias_shape{
      1,
      n_head,
      static_cast<int64_t>(src_word_data.size()),
      static_cast<int64_t>(src_word_data.size())};
  SetTensorData<float>(
      src_slf_attn_bias_data, src_slf_attn_bias_shape, src_slf_attn_bias.get());
  // trg_word
  auto trg_word = predictor->GetInput(3);
  std::vector<int64_t> trg_word_data(2, 0);
  std::vector<int64_t> trg_word_shape{2, 1};
  std::vector<uint64_t> lod_level_0{0, 2};
  std::vector<uint64_t> lod_level_1{0, 1, 2};
  std::vector<std::vector<uint64_t>> trg_word_lod(2);
  trg_word_lod[0] = lod_level_0;
  trg_word_lod[1] = lod_level_1;
  SetTensorData<int64_t>(
      trg_word_data, trg_word_shape, trg_word.get(), trg_word_lod);
  // init_score
  auto init_score = predictor->GetInput(4);
  std::vector<float> init_score_data(2);
  init_score_data[0] = 0;
  init_score_data[1] = -1e9f;
  std::vector<int64_t> init_score_shape{2, 1};
  std::vector<std::vector<uint64_t>> init_score_lod(trg_word_lod);
  SetTensorData<float>(
      init_score_data, init_score_shape, init_score.get(), init_score_lod);
  // init_idx
  auto init_idx = predictor->GetInput(5);
  std::vector<int32_t> init_idx_data(2, 0);
  std::vector<int64_t> init_idx_shape{2};
  SetTensorData<int32_t>(init_idx_data, init_idx_shape, init_idx.get());
  // trg_slf_attn_bias
  auto trg_slf_attn_bias = predictor->GetInput(6);
  std::vector<float> trg_slf_attn_bias_data(max_out_len * n_head * 1 *
                                            max_out_len);
  offset = 0;
  for (int k = 0; k < max_out_len; k++) {
    for (int j = 0; j < n_head; j++) {
      for (int i = 0; i < max_out_len; i++) {
        trg_slf_attn_bias_data[offset++] = (i <= k) ? 0.0f : -1e9f;
      }
    }
  }
  std::vector<int64_t> trg_slf_attn_bias_shape{
      max_out_len, n_head, 1, max_out_len};
  SetTensorData<float>(
      trg_slf_attn_bias_data, trg_slf_attn_bias_shape, trg_slf_attn_bias.get());
  // trg_src_attn_bias
  auto trg_src_attn_bias = predictor->GetInput(7);
  std::vector<float> trg_src_attn_bias_data(1 * n_head * 1 *
                                            src_word_data.size());
  offset = 0;
  for (int j = 0; j < 1 * n_head * 1; j++) {
    for (int i = 0; i < seq_len; i++) {
      trg_src_attn_bias_data[offset++] = 0.0f;
    }
    for (int i = seq_len; i < src_word_data.size(); i++) {
      trg_src_attn_bias_data[offset++] = -1e9f;
    }
  }
  std::vector<int64_t> trg_src_attn_bias_shape{
      1, n_head, 1, static_cast<int64_t>(src_word_data.size())};
  SetTensorData<float>(
      trg_src_attn_bias_data, trg_src_attn_bias_shape, trg_src_attn_bias.get());
  // kv_padding_selection
  auto kv_padding_selection = predictor->GetInput(8);
  std::vector<float> kv_padding_selection_data(max_out_len * n_head *
                                               max_out_len * 1);
  offset = 0;
  for (int k = 0; k < max_out_len; k++) {
    for (int j = 0; j < n_head; j++) {
      for (int i = 0; i < max_out_len; i++) {
        kv_padding_selection_data[offset++] = (i == k) ? 1.0f : 0.0f;
      }
    }
  }
  std::vector<int64_t> kv_padding_selection_shape{
      max_out_len, n_head, max_out_len, 1};
  SetTensorData<float>(kv_padding_selection_data,
                       kv_padding_selection_shape,
                       kv_padding_selection.get());
}

void CheckOutputData(
    const std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor,
    const std::vector<int64_t> &ref_seq_ids_data,
    const std::vector<float> &ref_seq_scores_data) {
  // seq_ids
  auto seq_ids = predictor->GetOutput(0);
  auto seq_ids_shape = seq_ids->shape();
  auto seq_ids_size = std::accumulate(seq_ids_shape.begin(),
                                      seq_ids_shape.end(),
                                      1,
                                      std::multiplies<int64_t>());
  ASSERT_EQ(seq_ids_size, ref_seq_ids_data.size());
  auto *seq_ids_data = seq_ids->data<int64_t>();
  for (size_t i = 0; i < seq_ids_size; i++) {
    EXPECT_EQ(seq_ids_data[i], ref_seq_ids_data[i]);
  }
  // seq_scores
  auto seq_scores = predictor->GetOutput(1);
  auto seq_scores_shape = seq_scores->shape();
  auto seq_scores_size = std::accumulate(seq_scores_shape.begin(),
                                         seq_scores_shape.end(),
                                         1,
                                         std::multiplies<int64_t>());
  ASSERT_EQ(seq_scores_size, ref_seq_scores_data.size());
  auto *seq_scores_data = seq_scores->data<float>();
  for (size_t i = 0; i < seq_scores_size; i++) {
    EXPECT_NEAR(seq_scores_data[i], ref_seq_scores_data[i], 1e-5);
  }
}

TEST(TransformerWithMask, test_transformer_with_mask_fp32_arm) {
  // Save the optimized model by using full api with CxxConfig
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(FLAGS_model_dir);
  cxx_config.set_valid_places(
      {lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
       lite_api::Place{TARGET(kARM), PRECISION(kInt64)}});
  auto predictor = lite_api::CreatePaddlePredictor(cxx_config);
  predictor->SaveOptimizedModel(FLAGS_model_dir + ".nb",
                                paddle::lite_api::LiteModelType::kNaiveBuffer);
  // Load the optimized model and run inference by using light api with
  // MobileConfig
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(FLAGS_model_dir + ".nb");
  mobile_config.set_threads(1);
  mobile_config.set_power_mode(paddle::lite_api::PowerMode::LITE_POWER_HIGH);
  std::vector<std::pair<std::vector<int64_t>,
                        std::pair<std::vector<int64_t>, std::vector<float>>>>
      test_cases = {
          {{16, 16, 16, 1},
           {{0, 16, 16, 16, 16, 16, 16, 1, 0, 16, 16, 16, 16, 16, 9, 1},
            {0.0f,
             -0.939061f,
             -1.91494f,
             -2.94378f,
             -4.26457f,
             -5.82675f,
             -7.45856f,
             -7.58065f,
             0.0f,
             -0.939061f,
             -1.91494f,
             -2.94378f,
             -4.26457f,
             -5.82675f,
             -8.70994f,
             -8.8053f}}},
          {{16, 16, 16, 10, 1},
           {{0, 6, 53, 11, 1, 0, 6, 53, 56, 4, 1},
            {0.0f,
             -2.36122f,
             -4.1678f,
             -6.19764f,
             -7.69256f,
             0.0f,
             -2.36122f,
             -4.1678f,
             -6.20145f,
             -7.66355f,
             -8.63024f}}},
          {{126, 4, 33, 1},
           {{0, 68, 5, 17, 1, 0, 68, 5, 13, 14, 1},
            {0.0f,
             -0.829941f,
             -1.20217f,
             -2.23938f,
             -2.98262f,
             0.0f,
             -0.829941f,
             -1.20217f,
             -2.25051f,
             -3.07555f,
             -3.57711f}}},
          {{126, 4, 33, 99, 1},
           {{0, 14, 242, 17, 1, 0, 93, 38, 27, 68, 1},
            {0.f,
             -1.8504f,
             -2.66679f,
             -3.09469f,
             -3.63227f,
             0.0f,
             -1.33829f,
             -1.41656f,
             -3.1333f,
             -3.27901f,
             -3.88582f}}}};
  for (auto &test_case : test_cases) {
    PrepareInputData(predictor, test_case.first);
    predictor->Run();
    CheckOutputData(predictor, test_case.second.first, test_case.second.second);
  }
}

}  // namespace lite
}  // namespace paddle
