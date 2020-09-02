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
namespace xpu {

template <typename T>
class Reshape2Compute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ReshapeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto x = param.x;
    auto output = param.output;
    auto xshape = param.xshape;
    auto x_dims = x->dims();
    auto x_dims_data = x_dims.Vectorize();
    auto out_dims = output->dims();
    output->ShareDataWith(*x);
    output->Resize(out_dims);
    auto* xshape_data = xshape->mutable_data<int64_t>(TARGET(kXPU));
    TargetWrapperXPU::MemcpySync(xshape_data,
                                 x_dims_data.data(),
                                 x_dims.size() * sizeof(int64_t),
                                 IoDirection::HtoD);
  }

  virtual ~Reshape2Compute() = default;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
