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

class ostream {
 public:
  ostream() {}
  explicit ostream(const std::string& x) : _data(x) {}
  ~ostream() {}

  const char* c_str() { return _data.c_str(); }

  const std::string& str() { return _data; }
  const std::string& str(const std::string& x) {
    _data = x;
    return _data;
  }

  template <typename T>
  ostream& operator<<(const T& obj);

  template <typename T>
  ostream& operator<<(const T* obj);

 private:
  std::string _data;
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
