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

#include <string>
#include <vector>

namespace nnadapter {

std::string string_format(const std::string fmt_str, ...);

template <typename T = std::string>
static T string_parse(const std::string& v) {
  return v;
}

template <>
int32_t string_parse<int32_t>(const std::string& v) {
  return std::stoi(v);
}

template <>
int64_t string_parse<int64_t>(const std::string& v) {
  return std::stoll(v);
}

template <>
float string_parse<float>(const std::string& v) {
  return std::stof(v);
}

template <>
double string_parse<double>(const std::string& v) {
  return std::stod(v);
}

template <>
bool string_parse<bool>(const std::string& v) {
  std::string upper;
  for (size_t i = 0; i < v.length(); i++) {
    char ch = v[i];
    if (ch >= 'a' && ch <= 'z') {
      ch = ch - 'a' + 'A';
    }
    upper.push_back(ch);
  }
  return upper == "TRUE" || upper == "1";
}

template <class T = std::string>
static std::vector<T> string_split(const std::string& original,
                                   const std::string& separator) {
  std::vector<T> results;
  std::string::size_type pos1, pos2;
  pos2 = original.find(separator);
  pos1 = 0;
  while (std::string::npos != pos2) {
    results.push_back(string_parse<T>(original.substr(pos1, pos2 - pos1)));
    pos1 = pos2 + separator.size();
    pos2 = original.find(separator, pos1);
  }
  if (pos1 != original.length()) {
    results.push_back(string_parse<T>(original.substr(pos1)));
  }
  return results;
}

}  // namespace nnadapter
