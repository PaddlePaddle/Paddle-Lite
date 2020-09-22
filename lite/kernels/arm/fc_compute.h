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
#include <stdint.h>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/type_trans.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType PType, PrecisionType OutType>
class FcCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  using param_t = operators::FcParam;

  virtual void ReInitWhenNeeded();
  virtual void PrepareForRun();
  virtual void Run();

  ~FcCompute() = default;

 private:
  DDim last_shape_;
  Tensor weights_;
  Tensor bias_;
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  bool flag_gemm_{true};
  int m_;
  int n_;
  int k_;
  std::vector<float> scale_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
