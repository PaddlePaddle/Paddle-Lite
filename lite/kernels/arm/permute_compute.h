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
#include "lite/operators/permute_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class PermuteCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::PermuteParam;

  void PrepareForRun() override;
  void Run() override;

  virtual ~PermuteCompute() = default;

 private:
  int _num_axes;
  int _count;
  bool _need_permute{false};
  bool _transpose{false};
  int _trans_num;
  int _trans_w;
  int _trans_h;
  std::vector<int> _new_steps;
  std::vector<int> _old_steps;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
