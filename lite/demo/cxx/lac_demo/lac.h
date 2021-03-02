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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle_api.h"  // NOLINT

enum CODE_TYPE {
  CODE_GB18030 = 0,
  CODE_UTF8 = 1,
};

struct OutputItem {
  std::string word;
  std::string tag;
};

class LAC {
 private:
  CODE_TYPE _codetype;

  std::vector<std::string> _seq_words;
  std::vector<std::vector<std::string>> _seq_words_batch;
  std::vector<std::vector<size_t>> _lod;
  std::vector<std::string> _labels;
  std::vector<OutputItem> _results;
  std::vector<std::vector<OutputItem>> _results_batch;

  std::shared_ptr<std::unordered_map<int64_t, std::string>> _id2label_dict;
  std::shared_ptr<std::unordered_map<std::string, std::string>> _q2b_dict;
  std::shared_ptr<std::unordered_map<std::string, int64_t>> _word2id_dict;
  std::unordered_map<std::string, std::string> _config_dict;
  int64_t _oov_id;

  std::shared_ptr<paddle::lite_api::PaddlePredictor> _predictor;   //
  std::unique_ptr<paddle::lite_api::Tensor> _input_tensor;         //
  std::unique_ptr<const paddle::lite_api::Tensor> _output_tensor;  //

  int feed_data(const std::vector<std::string> &querys);

  std::vector<OutputItem> parse_targets(const std::vector<std::string> &tag_ids,
                                        const std::vector<std::string> &words);

 public:
  explicit LAC(const std::string &model_path,
               const std::string &conf_path,
               int threads = 1,
               CODE_TYPE type = CODE_UTF8);

  std::vector<OutputItem> lexer(const std::string &query);
  std::vector<std::vector<OutputItem>> lexer(
      const std::vector<std::string> &query);
};
