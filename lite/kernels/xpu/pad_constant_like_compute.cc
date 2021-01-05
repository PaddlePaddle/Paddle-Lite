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

#include "lite/kernels/xpu/pad_constant_like_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

union TypeUnion {
  float fp32;
  int32_t int32;
};

void PadConstantLikeCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto x_dims = param.x->dims();
  auto y_dims = param.y->dims();
  float* out = param.output->mutable_data<float>(TARGET(kXPU));

  TypeUnion value;
  value.fp32 = param.pad_value;

  if (x_dims.size() == 2 && x_dims[1] == y_dims[1]) {
    int r = xdnn::memset_4_byte(ctx.GetRawContext(), /* context */
                                reinterpret_cast<void*>(out),
                                value.int32,
                                param.x->numel());
    CHECK_EQ(r, 0);

    r = xdnn::elementwise_add(ctx.GetRawContext(),    /* context */
                              param.y->data<float>(), /* x */
                              out,                    /* y */
                              out,
                              param.y->numel());

    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "Unsupport shape";
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pad_constant_like,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::PadConstantLikeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
