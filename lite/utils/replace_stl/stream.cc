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

#include "lite/utils/replace_stl/stream.h"
#include <assert.h>
#include <stdio.h>
#include "lite/utils/string.h"

#if defined(LITE_ON_TINY_PUBLISH) && !defined(LITE_WITH_XCODE)

namespace paddle {
namespace lite {
namespace replace_stl {

#ifndef LITE_WITH_LOG
#define ADD_DATA_AS_STRING(data_, obj_)
#else
#define ADD_DATA_AS_STRING(data_, obj_)    \
  std::string text = std::to_string(obj_); \
  pad(text);                               \
  data_ = data_ + text;

void ostream::pad(const std::string& text) {
  if (display_width_ > 0) {
    if (display_width_ < text.size()) {
      fprintf(stderr, "Replace STL IO display length less than text\n");
      assert(false);
    } else {
      for (int i = 0; i < display_width_ - text.size(); ++i) {
        data_.push_back(' ');
      }
      display_width_ = -1;
    }
  }
}
#endif

template <>
ostream& ostream::operator<<(const char* obj) {
  data_ = data_ + std::string(obj);
  return *this;
}

template <>
ostream& ostream::operator<<(char* const& obj) {
  data_ = data_ + std::string(obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const char& obj) {
  data_ = data_ + obj;
  return *this;
}

template <>
ostream& ostream::operator<<(const std::string& obj) {
  data_ = data_ + obj;
  return *this;
}

template <>
ostream& ostream::operator<<(const int16_t& obj) {
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const int& obj) {
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const bool& obj) {
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const long& obj) {  // NOLINT
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const long long& obj) {  // NOLINT
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const unsigned& obj) {
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const uint16_t& obj) {
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const unsigned long& obj) {  // NOLINT
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const unsigned long long& obj) {  // NOLINT
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const float& obj) {
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const double& obj) {
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const long double& obj) {
  ADD_DATA_AS_STRING(data_, obj);
  return *this;
}

template <>
ostream& ostream::operator<<(const LiteIoWidth& obj) {
  int width = obj.width;
  assert(width > 0);
  display_width_ = width;
  return *this;
}

}  // namespace replace_stl
}  // namespace lite
}  // namespace paddle

#endif  // LITE_ON_TINY_PUBLISH
