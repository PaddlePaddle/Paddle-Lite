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

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "lac.h"  // NOLINT

enum RVAL {
  _SUCCESS = 0,
  _FAILD = -1,
};

std::vector<std::string> split_tokens(const std::string &line,
                                      const std::string &pattern);

std::unordered_map<std::string, int64_t> load_word2id_dict(
    const std::string &filepath);

std::unordered_map<std::string, std::string> load_q2b_dict(
    const std::string &filepath);

std::unordered_map<int64_t, std::string> load_id2label_dict(
    const std::string &filepath);

int get_next_gb18030(const char *str);
int get_next_utf8(const char *str);
int get_next_word(const char *str, int codetype);

std::vector<std::string> split_words(const char *input, int len, int codetype);
std::vector<std::string> split_words(const std::string &input, int codetype);
