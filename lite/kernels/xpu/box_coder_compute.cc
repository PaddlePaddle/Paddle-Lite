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

#include "lite/kernels/xpu/box_coder_compute.h"
#include <string>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void BoxCoderCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto prior_box_var_size = 0;
  auto* prior_box = param.prior_box;
  auto* prior_box_var = param.prior_box_var;
  if (prior_box_var) {
    prior_box_var_size = prior_box_var->dims().size();
  }
  auto* target_box = param.target_box;
  auto* output_box = param.proposals;
  std::vector<float> variance = param.variance;
  const int axis = param.axis;
  std::string code_type = param.code_type;
  bool normalized = param.box_normalized;

  auto row = target_box->dims()[0];
  auto col = prior_box->dims()[0];
  if (code_type == "decode_center_size") {
    col = target_box->dims()[1];
  }
  auto len = prior_box->dims()[1];
  output_box->Resize({row, col, len});
  auto* output = output_box->mutable_data<float>(TARGET(kXPU));

  if (code_type == "encode_center_size") {
    int r = xdnn::box_coder_encode(ctx.GetRawContext(),
                                   prior_box->data<float>(),
                                   prior_box_var->data<float>(),
                                   target_box->data<float>(),
                                   row,
                                   col,
                                   len,
                                   normalized,
                                   prior_box_var_size,
                                   variance.data(),
                                   variance.size(),
                                   output);
    CHECK_EQ(r, 0);
  } else if (code_type == "decode_center_size") {
    int r = xdnn::box_coder_decode(ctx.GetRawContext(),
                                   prior_box->data<float>(),
                                   prior_box_var->data<float>(),
                                   target_box->data<float>(),
                                   row,
                                   col,
                                   len,
                                   normalized,
                                   prior_box_var_size,
                                   variance.data(),
                                   variance.size(),
                                   axis,
                                   output);
    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "box_coder don't support this code_type: " << code_type;
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(box_coder,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::BoxCoderCompute,
                     def)
    .BindInput("PriorBox", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("PriorBoxVar", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("TargetBox", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputBox", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
