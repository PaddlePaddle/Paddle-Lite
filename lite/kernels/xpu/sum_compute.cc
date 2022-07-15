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

#include "lite/kernels/xpu/sum_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SumCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  std::vector<lite::Tensor*>& inputs = param.X;
  auto* out_data = param.Out->mutable_data<float>(TARGET(kXPU));
  if (inputs.size() == 1) {
    if (!param.inplace) {
      int r = xdnn::copy<float>(ctx.GetRawContext(),
                                inputs[0]->data<float>(),
                                out_data,
                                param.Out->numel());
      CHECK_EQ(r, 0);
    }
    return;
  }
  auto output_dim = param.Out->dims();
  std::vector<int> data_shape(output_dim.size(), 0);
  for (size_t i = 0; i < output_dim.size(); i++) {
    data_shape[i] = static_cast<int>(output_dim[i]);
  }
  int start_index = 0;
  if (param.inplace) {  // inplace add
    start_index = 1;
  } else {
    int ret = xdnn::broadcast_add<float>(ctx.GetRawContext(),
                                         inputs[0]->data<float>(),
                                         inputs[1]->data<float>(),
                                         out_data,
                                         data_shape,
                                         data_shape);
    CHECK_EQ(ret, 0);
    start_index = 2;
  }
  for (auto it = inputs.begin() + start_index; it != inputs.end(); ++it) {
    const auto& x_data = (*it)->data<float>();
    int ret = xdnn::broadcast_add<float>(ctx.GetRawContext(),
                                         x_data,
                                         out_data,
                                         out_data,
                                         data_shape,
                                         data_shape);
    CHECK_EQ(ret, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    sum, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::SumCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();
