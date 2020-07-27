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

#include "lite/kernels/xpu/concat_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void ConcatCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto ins = param.x;
  auto out = param.output;
  int64_t axis = param.axis;

  int n = ins.size();
  int h = 1;
  int w_except_axis = 1;
  CHECK(n <= 8) << "XPU only surpport at most 8 tensors for now";
  for (int i = 0; i < axis; ++i) {
    h *= (ins[0]->dims())[i];
  }
  for (int i = axis + 1; i < ins[0]->dims().size(); ++i) {
    w_except_axis *= (ins[0]->dims())[i];
  }
  CHECK(axis >= 0) << "concat: axis shoud >= 0!";
  CHECK(axis < ins[0]->dims().size()) << "concat: axis shoud < ins[0]->dims()!";
  for (int i = 0; i < n; ++i) {
    int hh = 1;
    int ww = 1;
    for (int j = 0; j < axis; ++j) {
      hh *= (ins[i]->dims())[j];
    }
    for (int j = axis + 1; j < ins[i]->dims().size(); ++j) {
      ww *= (ins[i]->dims())[j];
    }
    CHECK(hh == h) << "concat: h should be eual!";
    CHECK(ww == w_except_axis) << "concat: w should be eual except for axis!";
  }

  int in_w_host[n];      // NOLINT
  const float* ptrs[n];  // NOLINT

  for (int i = 0; i < n; ++i) {
    ptrs[i] = ins[i]->data<float>();
    in_w_host[i] = w_except_axis * (ins[i]->dims())[axis];
  }

  int r = xdnn::concat<float>(ctx.GetRawContext(), /* ctx */
                              h,                   /* height */
                              in_w_host,           /* width_x */
                              n,                   /* n */
                              ptrs,                /* lm_ptrs */
                              out->mutable_data<float>(TARGET(kXPU)) /*y*/);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    concat, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::ConcatCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
