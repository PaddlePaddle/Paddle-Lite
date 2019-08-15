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

#pragma once

#include <string>
#include "pass/pass_base.h"

namespace paddle_mobile {
namespace pass {

class ModelObfuscatePass : public PassBase {
 public:
  ModelObfuscatePass(std::string key);
  void convert_data(char *data, int len);
  int version = 1;

 private:
  int acc = 0;
  int base = 17;
  int stride = 100;
};

}  // namespace pass
}  // namespace paddle_mobile
