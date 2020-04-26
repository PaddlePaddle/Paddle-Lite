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

#ifdef LITE_ON_TINY_PUBLISH
#include <stdlib.h>
#include <string>
#else
#include <iomanip>
#include <iostream>
#include <sstream>
#endif

#ifdef LITE_ON_TINY_PUBLISH
namespace paddle {
namespace lite {

namespace replace_stl {

struct LiteIoWidth {
  explicit LiteIoWidth(int value) : width(value) {}
  int width;
};

static LiteIoWidth setw(int width) { return LiteIoWidth(width); }

class ostream {
 public:
  ostream() {}
  explicit ostream(const std::string& x) : data_(x) {}
  ~ostream() {}

  const char* c_str() { return data_.c_str(); }

  const std::string& str() { return data_; }
  const std::string& str(const std::string& x) {
    data_ = x;
    return data_;
  }

  template <typename T>
  ostream& operator<<(const T& obj) {
    fprintf(stderr,
            "'ostream& operator<<(const %s& obj)' is unimplemented in "
            "replace_stl\n",
            typeid(T).name());
  }

  template <typename T>
  ostream& operator<<(const T* obj) {
    fprintf(stderr,
            "'ostream& operator<<(const %s* obj)' is unimplemented in "
            "replace_stl\n",
            typeid(T).name());
  }

 private:
  void pad(const std::string& text);
  std::string data_;
  int display_width_{-1};  // -1 refers to no setting
};

class stringstream : public ostream {
 public:
  stringstream() = default;

  ~stringstream() {}

  explicit stringstream(const std::string& x) : ostream(x) {}  // NOLINT

  void precision(int x) {}
};

}  // namespace replace_stl

}  // namespace lite
}  // namespace paddle

// replace namespace
namespace STL = paddle::lite::replace_stl;
#else
namespace STL = std;
#endif
