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

#include "lac_util.h"  // NOLINT

std::vector<std::string> split_tokens(const std::string &line,
                                      const std::string &pattern) {
  std::vector<std::string> tokens;
  if ("" == line || "" == pattern) {
    return tokens;
  }

  int pos = 0;
  int size = line.size();
  for (int i = 0; i < size; i++) {
    pos = line.find(pattern, i);
    if (-1 != pos) {
      tokens.push_back(line.substr(i, pos - i));
      i = pos + pattern.size() - 1;
    } else {
      tokens.push_back(line.substr(i));
      break;
    }
  }  // end of for
  return tokens;
}

std::unordered_map<std::string, int64_t> load_word2id_dict(
    const std::string &filepath) {
  std::unordered_map<std::string, int64_t> kv_dict;
  std::ifstream infile(filepath);
  if (infile.fail()) {
    return kv_dict;
  }
  std::string line = "";
  std::vector<std::string> tokens;
  while (std::getline(infile, line) && infile.good()) {
    tokens = split_tokens(line, "\t");
    if ("" == line || 2 != tokens.size()) {
      continue;
    }
    int64_t val = std::stoll(tokens[0]);
    std::string key = tokens[1];
    kv_dict[key] = val;
  }
  infile.close();
  return kv_dict;
}

std::unordered_map<std::string, std::string> load_q2b_dict(
    const std::string &filepath) {
  std::unordered_map<std::string, std::string> kv_dict;
  std::ifstream infile(filepath);
  if (infile.fail()) {
    return kv_dict;
  }
  std::string line = "";
  std::vector<std::string> tokens;
  while (std::getline(infile, line) && infile.good()) {
    tokens = split_tokens(line, "\t");
    if ("" == line || 2 != tokens.size()) {
      continue;
    }
    kv_dict[tokens[0]] = tokens[1];
  }
  infile.close();
  return kv_dict;
}

std::unordered_map<int64_t, std::string> load_id2label_dict(
    const std::string &filepath) {
  std::unordered_map<int64_t, std::string> kv_dict;
  std::ifstream infile(filepath);
  if (infile.fail()) {
    return kv_dict;
  }
  std::string line = "";
  std::vector<std::string> tokens;
  while (std::getline(infile, line) && infile.good()) {
    tokens = split_tokens(line, "\t");
    if ("" == line || 2 != tokens.size()) {
      continue;
    }
    int64_t key = std::stoll(tokens[0]);
    std::string val = tokens[1];
    kv_dict[key] = val;
  }
  infile.close();
  return kv_dict;
}

int get_next_gb18030(const char *str) {
  unsigned char *str_in = (unsigned char *)str;
  if (str_in[0] < 0x80) {
    return 1;
  }
  if (str_in[0] >= 0x81 && str_in[0] <= 0xfe && str_in[1] >= 0x40 &&
      str_in[1] <= 0xFE && str_in[1] != 0x7F) {
    return 2;
  }
  if (str_in[0] >= 0x81 && str_in[0] <= 0xfe && str_in[1] >= 0x30 &&
      str_in[1] <= 0x39 && str_in[2] >= 0x81 && str_in[2] <= 0xfe &&
      str_in[3] >= 0x30 && str_in[3] <= 0x39) {
    return 4;
  }
  return 0;
}

int get_next_utf8(const char *str) {
  unsigned char *str_in = (unsigned char *)str;
  if (str_in[0] < 0x80) {
    return 1;
  }
  if (str_in[0] >= 0xC2 && str_in[0] < 0xE0 && str_in[1] >> 6 == 2) {
    return 2;
  }
  if (str_in[0] >> 4 == 14 && str_in[1] >> 6 == 2 && str_in[2] >> 6 == 2 &&
      (str_in[0] > 0xE0 || str_in[1] >= 0xA0)) {
    return 3;
  }
  if (str_in[0] >> 3 == 30 && str_in[1] >> 6 == 2 && str_in[2] >> 6 == 2 &&
      str_in[3] >> 6 == 2 && str_in[0] <= 0xF4 &&
      (str_in[0] > 0xF0 || str_in[1] >= 0x90)) {
    return 4;
  }
  return 0;
}

int get_next_word(const char *str, int codetype) {
  int len = 0;
  switch (codetype) {
    case CODE_GB18030:
      len = get_next_gb18030(str);
      break;
    case CODE_UTF8:
      len = get_next_utf8(str);
      break;
    default:
      len = 0;
      break;
  }
  len = len == 0 ? 1 : len;
  return len;
}

std::vector<std::string> split_words(const char *input, int len, int codetype) {
  std::vector<std::string> words;
  char *p = const_cast<char *>(input);
  int temp_len = 0;
  std::string key;
  for (int i = 0; i < len; i += temp_len) {
    temp_len = get_next_word(p, codetype);
    key.assign(p, temp_len);
    words.push_back(key);
    p += temp_len;
  }
  return words;
}

std::vector<std::string> split_words(const std::string &input, int codetype) {
  const char *p = input.c_str();
  int len = input.length();
  return split_words(p, len, codetype);
}
