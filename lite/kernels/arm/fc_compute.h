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
bool check_fc_use_gemm(int m, const std::vector<float>& scale, bool has_bias) {
  return m > 1;
}

template <>
bool check_fc_use_gemm<PRECISION(kInt8), PRECISION(kFloat)>(
    int m, const std::vector<float>& scale, bool has_bias) {
  CHECK(scale.size() > 0) << "Int8 FC param must has weight_scale";
  return m > 1 && scale.size() == 1;
}

template <>
bool check_fc_use_gemm<PRECISION(kInt8), PRECISION(kInt8)>(
    int m, const std::vector<float>& scale, bool has_bias) {
  CHECK(scale.size() > 0) << "Int8 FC param must has weight_scale";
  return m > 1 && scale.size() == 1 && !has_bias;
}

template <PrecisionType PType, PrecisionType OutType>
class FcCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  using param_t = operators::FcParam;

  virtual void ReInitWhenNeeded() {
    auto& param = this->template Param<operators::FcParam>();
    auto x_dims = param.input->dims();
    if (last_shape_ == x_dims) {
      return;
    }
    last_shape_ = x_dims;
    auto w_dims = param.w->dims();
    auto& ctx = this->ctx_->template As<ARMContext>();

    CHECK_GE(x_dims.size(), 2UL);
    CHECK_EQ(w_dims.size(), 2UL);
    CHECK_GE(param.output->dims().size(), 2UL);

    m_ = x_dims.Slice(0, param.in_num_col_dims).production();
    k_ = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
    CHECK_EQ(k_, w_dims[0]);
    n_ = w_dims[1];
    CHECK_EQ(k_, static_cast<int>(w_dims[0]));
    flag_gemm_ = check_fc_use_gemm<PType, OutType>(
        m_, param.weight_scale, param.bias != nullptr);
  }

  virtual void PrepareForRun();
  virtual void Run();

  ~FcCompute() = default;

 private:
  DDim last_shape_;
  Tensor bias_;
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
