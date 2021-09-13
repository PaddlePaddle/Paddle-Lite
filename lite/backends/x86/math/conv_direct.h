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
#include "lite/core/context.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void conv_direct_3x3s2(const float* i_data,
                       const float* trans_weight,
                       int bs,
                       int ic,
                       int ih,
                       int iw,
                       int oc,
                       int oc_expand,
                       float* o_data,
                       int oh,
                       int ow,
                       int ph,
                       int pw,
                       const float* bias,
                       lite_api::ActivationType active_type);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
