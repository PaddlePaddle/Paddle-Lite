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

#include "pass/model_obfuscate.h"

namespace paddle_mobile {
namespace pass {

ModelObfuscatePass::ModelObfuscatePass(std::string key) {
  for (auto c : key) {
    acc *= base;
    acc += (int)c;
    acc %= stride;
  }
  acc += stride;
}

void ModelObfuscatePass::convert_data(char *data, int len) {
  for (int i = 0; i < len; i += acc) {
    data[i] = 255 - data[i];
  }
}

}  // namespace pass
}  // namespace paddle_mobile
