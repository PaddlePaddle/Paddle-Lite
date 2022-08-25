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

template <typename InType, PrecisionType PType>
void SplitCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& dout = param.output;
  const lite::Tensor* x = param.x;
  auto in_dim = x->dims();
  auto axis = param.axis;
  int height = 1;
  for (int i = 0; i < axis; i++) {
    height = height * in_dim[i];
  }
  int width = x->numel() / height;
  std::vector<InType*> out_ptrs;
  std::vector<int> width_out;
  for (lite::Tensor* out : dout) {
    out->set_lod(x->lod());
    out_ptrs.push_back(out->mutable_data<InType>(TARGET(kXPU)));
    width_out.push_back(out->numel() / height);
  }
  int r = xdnn::split<InType>(ctx.GetRawContext(),
                              x->data<InType>(),
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

namespace xpu = paddle::lite::kernels::xpu;
using splitFP32 = xpu::SplitCompute<float, PRECISION(kFloat)>;
using splitFP16 = xpu::SplitCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(split, kXPU, kFloat, kNCHW, splitFP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("SectionsTensorList",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(split, kXPU, kFP16, kNCHW, splitFP16, float16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("SectionsTensorList",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
