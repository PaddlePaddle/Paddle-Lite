// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/unstack_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void UnstackCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& dout = param.Out;
  auto in_dim = param.X->dims();
  int axis = param.axis;
  if (axis < 0) {
    axis += in_dim.size();
  }
  int height = 1;
  for (int i = 0; i < axis; i++) {
    height = height * in_dim[i];
  }
  int width = param.X->numel() / height;
  std::vector<float*> out_ptrs;
  std::vector<int> width_out;
  for (auto out : dout) {
    out->set_lod(param.X->lod());
    out_ptrs.push_back(out->mutable_data<float>(TARGET(kXPU)));
    width_out.push_back(out->numel() / height);
  }
  int r = xdnn::split<float>(ctx.GetRawContext(),
                             param.X->data<float>(),
                             out_ptrs,
                             {height, width},
                             width_out,
                             1);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(unstack,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::UnstackCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
