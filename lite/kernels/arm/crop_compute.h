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
#include "lite/operators/crop_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class CropCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::CropParam;

  void Run() override;
  virtual ~CropCompute() = default;
  void crop_fun(const lite::Tensor* input, lite::Tensor* output);

 private:
  std::vector<int> offsets_;
  std::vector<int> shape_;

  int c_off;
  int h_off;
  int w_off;
  int c_end;
  int h_end;
  int w_end;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
