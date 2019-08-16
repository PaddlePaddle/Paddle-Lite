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
#include "framework/program/program.h"

namespace paddle_mobile {

template <typename Dtype = CPU, Precision P = Precision::FP32>
class Loader {
 public:
  const framework::Program<Dtype, P> Load(const std::string &dirname,
                                          bool optimize = false,
                                          bool quantification = false,
                                          bool can_add_split = false);

  const framework::Program<Dtype, P> Load(const std::string &model_path,
                                          const std::string &para_path,
                                          bool optimize = false,
                                          bool quantification = false);

  const framework::Program<Dtype, P> LoadCombinedMemory(
      size_t model_len, const uint8_t *model_buf, size_t combined_params_len,
      const uint8_t *combined_params_buf, bool optimize = false,
      bool quantification = false);

 private:
  const framework::Program<Dtype, P> LoadProgram(const std::string &model_path,
                                                 bool optimize = false,
                                                 bool quantification = false,
                                                 bool can_add_split = false);
};

}  // namespace paddle_mobile
