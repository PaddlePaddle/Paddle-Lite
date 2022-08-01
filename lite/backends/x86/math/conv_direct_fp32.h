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

#pragma once
#include "lite/backends/x86/jit/gen/jitcode.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

struct conv_direct : lite::jit::gen::JitCode {
  conv_direct();
  void generate_code(int ic,
                     int ih,
                     int iw,
                     int oc,
                     int oc_expand,
                     int oh,
                     int ow,
                     int ph,
                     int pw,
                     int wh,
                     int ww,
                     int stridew);
  virtual void genCode() {}
  virtual ~conv_direct() {}
  void run(const float* i_data,
           const float* trans_weight,
           float* trans_out,
           int bs,
           int ic,
           int ih,
           int iw,
           int oc,
           int oc_expand,
           int oh,
           int ow,
           int ph,
           int pw,
           int wh,
           int ww,
           int strideh);
};

void conv_direct_transpose_out(int bs,
                               int oc,
                               int oh,
                               int ow,
                               float* o_data,
                               float* trans_out,
                               const float* bias,
                               lite_api::ActivationType active_type,
                               operators::ActivationParam act_param);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
