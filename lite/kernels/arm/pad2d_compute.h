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
#include <vector>
#include "lite/core/kernel.h"
#include "lite/operators/pad2d_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class Pad2dCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::Pad2dParam;

  void Run() override;

  virtual ~Pad2dCompute() = default;

 private:
  /////////////////////////////
  /*     _mode是PadMode
         typedef enum{
             PAD_CONSTANT = 0,
             PAD_EDGE = 1,
             PAD_REFLECT = 2,
         } PadMode;   */
  /////////////////////////
  int _mode;
  std::vector<int> _pad_h{0, 0};
  std::vector<int> _pad_w{0, 0};
  float _pad_value = 0.f;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
