/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "attribute.h"

namespace paddle_mobile {
namespace framework {



/*
 * Variant<int, float, std::string, std::vector<int>, std::vector<float>,
          std::vector<std::string>, bool, std::vector<bool>, BlockDesc *,
          int64_t>
 * */

struct PrintVistor: Vistor<Print &>{
  PrintVistor(Print &printer):printer_(printer){
  }
  template <typename T>
  Print &operator()(const T &value){
    printer_ << value;
    return printer_;
  }
 private:
  Print &printer_;
};

Print &operator<<(Print &printer, const Attribute &attr) {
  Attribute::ApplyVistor(PrintVistor(printer), attr);
//  std::vector<std::string> v = {"1", "2"};
//  printer << (v);
  return printer;
}

}
}  // namespace paddle_mobile
