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
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"

DEFINE_string(input_data_file_path, "", "The path of input data file");

namespace paddle {
namespace lite {

template <typename T>
void GetValueFromStream(std::stringstream* ss, T* t) {
  (*ss) >> (*t);
}

template <>
void GetValueFromStream<std::string>(std::stringstream* ss, std::string* t) {
  *t = ss->str();
}

template <typename T>
void Split(const std::string& line, char sep, std::vector<T>* v) {
  std::stringstream ss;
  T t;
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      GetValueFromStream<T>(&ss, &t);
      v->push_back(std::move(t));
      ss.str({});
      ss.clear();
    }
  }

  if (!ss.str().empty()) {
    GetValueFromStream<T>(&ss, &t);
    v->push_back(std::move(t));
    ss.str({});
    ss.clear();
  }
}

template <typename T>
bool ParseTensor(const std::vector<T>& data,
                 const std::vector<int64_t>& shape,
                 lite_api::Tensor* tensor,
                 const std::vector<std::vector<uint64_t>>& lod = {}) {
  tensor->Resize(shape);
  tensor->SetLoD(lod);
  std::copy(data.begin(), data.end(), tensor->mutable_data<T>());
  return true;
}

bool ParseLine(const std::string& line,
               const std::shared_ptr<lite_api::PaddlePredictor>& predictor) {
  // src_word
  int batch_size = 1;
  auto src_word = predictor->GetInput(0);
  std::vector<int64_t> src_word_data;
  Split(line, ' ', &src_word_data);
  std::vector<int64_t> src_word_shape{
      batch_size, static_cast<int64_t>(src_word_data.size()), 1};
  ParseTensor<int64_t>(src_word_data, src_word_shape, src_word.get());
  // src_pos
  auto src_pos = predictor->GetInput(1);
  std::vector<int64_t> src_pos_data(src_word_data.size());
  std::iota(src_pos_data.begin(), src_pos_data.end(), 0);
  std::vector<int64_t> src_pos_shape{
      batch_size, static_cast<int64_t>(src_pos_data.size()), 1};
  ParseTensor<int64_t>(src_pos_data, src_pos_shape, src_pos.get());
  // trg_word
  auto trg_word = predictor->GetInput(2);
  std::vector<int64_t> trg_word_data(batch_size, 0);
  std::vector<int64_t> trg_word_shape{batch_size, 1, 1};
  std::vector<uint64_t> lod(batch_size + 1);
  std::iota(lod.begin(), lod.end(), 0);
  std::vector<std::vector<uint64_t>> trg_word_lod(2, lod);
  ParseTensor<int64_t>(
      trg_word_data, trg_word_shape, trg_word.get(), trg_word_lod);
  // init_score
  auto init_score = predictor->GetInput(3);
  std::vector<float> init_score_data(batch_size, 0);
  std::vector<int64_t> init_score_shape{batch_size, 1};
  std::vector<std::vector<uint64_t>> init_score_lod(trg_word_lod);
  ParseTensor<float>(
      init_score_data, init_score_shape, init_score.get(), init_score_lod);
  // init_idx
  auto init_idx = predictor->GetInput(4);
  std::vector<int32_t> init_idx_data(batch_size);
  std::iota(init_idx_data.begin(), init_idx_data.end(), 0);
  std::vector<int64_t> init_idx_shape{batch_size};
  ParseTensor<int32_t>(init_idx_data, init_idx_shape, init_idx.get());
  return true;
}

std::vector<std::string> ParseInput(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream fin(path.c_str());
  if (fin.is_open()) {
    std::string line;
    while (std::getline(fin, line)) {
      lines.push_back(line);
    }
  }
  return lines;
}

void ParseResult(const std::shared_ptr<lite_api::PaddlePredictor>& predictor,
                 std::vector<int64_t>* ids,
                 std::vector<float>* scores,
                 int64_t bos = 0,
                 int64_t eos = 1) {
  auto seq_ids = predictor->GetOutput(0);
  auto seq_scores = predictor->GetOutput(1);
#if 0
  auto lod = seq_ids->lod();
  for (size_t i = 0; i < lod[0].size() - 1; i++) {
    size_t start = lod[0][i];
    size_t end = lod[0][i + 1];
    for (size_t j = 0; j < end - start; j++) {
      size_t sub_start = lod[1][start + j];
      size_t sub_end = lod[1][start + j + 1];
      auto data = seq_ids->data<int64_t>();
      for (size_t k = sub_start + 1; k < sub_end && data[k] != eos; k++) {
        ids->push_back(data[k]);
      }
      auto score = seq_scores->data<float>()[sub_end - 1];
      scores->push_back(exp(-score));
    }
  }
#else
  auto seq_ids_shape = seq_ids->shape();
  auto seq_ids_size = std::accumulate(seq_ids_shape.begin(),
                                      seq_ids_shape.end(),
                                      1,
                                      std::multiplies<int64_t>());
  auto seq_ids_data = seq_ids->data<int64_t>();
  ids->resize(seq_ids_size);
  for (int i = 0; i < seq_ids_size; i++) {
    (*ids)[i] = seq_ids_data[i];
  }
  auto seq_scores_shape = seq_scores->shape();
  auto seq_scores_size = std::accumulate(seq_scores_shape.begin(),
                                         seq_scores_shape.end(),
                                         1,
                                         std::multiplies<int64_t>());
  auto seq_scores_data = seq_scores->data<float>();
  scores->resize(seq_scores_size);
  for (int i = 0; i < seq_scores_size; i++) {
    (*scores)[i] = seq_scores_data[i];
  }
#endif
}

TEST(Transformer, test_transformer_lite_npu) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_power_mode(lite_api::LITE_POWER_HIGH);
  config.set_valid_places({lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kARM), PRECISION(kInt64)}});
  auto predictor = lite_api::CreatePaddlePredictor(config);
  // Warm up
  ParseLine(
      "28 13 21 99 273 15 95 20 20 20 20 13 55 21 2638 33 19 734 232 13 46 430 "
      "131 756 2198 510 44 2223 2431 1 ",
      predictor);
  predictor->Run();
  // Run all test cases
  int max_idx = 0;
  std::string max_input;
  double max_cost = 0.0f;
  std::vector<std::string> inputs = ParseInput(FLAGS_input_data_file_path);
  for (int i = 0; i < inputs.size(); i++) {
    LOG(INFO) << "[" << i << "] " << inputs[i];
    // Fill input data and repeat running
    double total_cost = 0.0f;
    for (int j = 0; j < FLAGS_repeats; j++) {
      ParseLine(inputs[i], predictor);
      auto start = GetCurrentUS();
      predictor->Run();
      auto cur_cost = (GetCurrentUS() - start) / 1000.0;
      LOG(INFO) << cur_cost << " ms";
      total_cost += cur_cost;
    }
    double avg_cost = total_cost / FLAGS_repeats;
    LOG(INFO) << avg_cost << " ms in average.";
    // Print output data
    std::vector<int64_t> ids;
    std::vector<float> scores;
    ParseResult(predictor, &ids, &scores);
    LOG(INFO) << "ids:";
    for (int j = 0; j < ids.size(); j++) {
      LOG(INFO) << ids[j];
    }
    LOG(INFO) << "scores:";
    for (int j = 0; j < scores.size(); j++) {
      LOG(INFO) << scores[i];
    }
    if (avg_cost > max_cost) {
      max_idx = i;
      max_cost = avg_cost;
      max_input = inputs[i];
    }
    LOG(INFO) << "i=" << max_idx << " input=" << max_input
              << " cost=" << max_cost;
  }
}

}  // namespace lite
}  // namespace paddle
