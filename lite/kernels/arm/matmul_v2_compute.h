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
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType PType, PrecisionType OutType>
class MatMulV2Compute : public KernelLite<TARGET(kARM), PType> {
 public:
  using param_t = operators::MatMulParam;

  void PrepareForRun() { auto& ctx = this->ctx_->template As<ARMContext>(); }

  void ReInitWhenNeeded() override;

  void Run() override;

  virtual ~MatMulV2Compute() = default;

 private:
  int m_{1};
  int n_{1};
  int k_{1};
  int lda_{1};
  int ldb_{1};
  int ldc_{1};
  std::vector<float> scale_;
  std::vector<float> scale_one;
  DDim last_x_shape_;
  DDim last_y_shape_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
