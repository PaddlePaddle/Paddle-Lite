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
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class YoloBoxParserCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::YoloBoxParam;

  void Run() override;
  virtual ~YoloBoxParserCompute() = default;

 private:
  // std::vector<int> anchors0_;
  // std::vector<int> anchors1_;
  // std::vector<int> anchors2_;
  // std::vector<int> downsample_ratio_;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
