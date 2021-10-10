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

#include "lite/kernels/x86/conv_direct.h"
#include <cmath>
#include "lite/backends/x86/math/conv3x3s2_direct_fp32.h"
#include "lite/backends/x86/math/conv_bias.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <>
void DirectConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK_EQ(param.strides[0], 2);
  CHECK_EQ(param.strides[1], 2);
  // auto& ctx = this->ctx_->template As<X86Context>();

  const auto* i_data = param.x->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  const int ph = (*(param.paddings))[0];
  const int pw = (*(param.paddings))[2];

  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oc = o_dims[1];
  int oh = o_dims[2];
  int ow = o_dims[3];

  float* trans_out = static_cast<float*>(
      TargetMalloc(TARGET(kX86), sizeof(float) * bs * oc_expand_ * oh * ow));
  memset(trans_out, 0, sizeof(float) * oc * oh * ow * bs);

  auto act_param = param.activation_param;
  code_->run(i_data,
             weights_.data<float>(),
             trans_out,
             bs,
             ic,
             ih,
             iw,
             oc,
             oc_expand_,
             oh,
             ow,
             ph,
             pw);

  lite::x86::math::conv_direct_3x3s2_tranpose_out(bs,
                                                  oc,
                                                  oh,
                                                  ow,
                                                  o_data,
                                                  trans_out,
                                                  b_data,
                                                  act_param.active_type,
                                                  act_param);
  TargetFree(TARGET(kX86), trans_out);
}
}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
