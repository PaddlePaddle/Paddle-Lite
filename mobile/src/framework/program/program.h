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
#include "common/types.h"
#include "framework/program/program_desc.h"
#include "framework/scope.h"

namespace paddle_mobile {
namespace framework {

template <typename Device, typename T = float>
class Program {
 public:
  std::shared_ptr<ProgramDesc> originProgram;
  std::shared_ptr<ProgramDesc> optimizeProgram;
  std::shared_ptr<Scope> scope;
  std::string model_path;
  std::string para_path;
  bool combined = false;
  bool quantification = false;
  size_t combined_params_len;
  uint8_t *combined_params_buf;
  int quantification_fold = 1;
};

}  // namespace framework
}  // namespace paddle_mobile
