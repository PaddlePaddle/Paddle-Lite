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
#include "framework/lod_tensor.h"
#include "framework/paddle_mobile_object.h"
#include "framework/program/program.h"

namespace paddle_mobile {

template <typename Dtype, Precision P = Precision::FP32>
class Loader : PaddleMobileObject {
 public:
  const framework::Program<Dtype, P> Load(const std::string &dirname);

 private:
  void LoadVar(framework::LoDTensor *tensor, const std::string &file_path);
};

}  // namespace paddle_mobile
