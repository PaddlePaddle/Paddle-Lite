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
#include <algorithm>
#include <string>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/operators/pad2d_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType Ptype, PrecisionType OutType>
class Pad2dCompute : public KernelLite<TARGET(kARM), Ptype> {
 public:
  using param_t = operators::Pad2dParam;

  void Run() override;

  virtual ~Pad2dCompute() = default;

 private:
  int mode_;
  std::vector<int> pad_h_{0, 0};
  std::vector<int> pad_w_{0, 0};
  float pad_value_ = 0.f;
  std::string data_format_{"NCHW"};
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
