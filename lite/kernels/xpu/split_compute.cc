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

#include "lite/kernels/xpu/split_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SplitCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& dout = param.output;
  auto in_dim = param.x->dims();
  auto axis = param.axis;
  int height = 1;
  for (int i = 0; i < axis; i++) {
    height = height * in_dim[i];
  }
  int width = param.x->numel() / height;
  std::vector<float*> out_ptrs;
  std::vector<int> width_out;
  for (auto out : dout) {
    out->set_lod(param.x->lod());
    out_ptrs.push_back(out->mutable_data<float>(TARGET(kXPU)));
    width_out.push_back(out->numel() / height);
  }
  int r = xdnn::split<float>(ctx.GetRawContext(),
                             param.x->data<float>(),
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

REGISTER_LITE_KERNEL(
    split, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::SplitCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("SectionsTensorList",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
