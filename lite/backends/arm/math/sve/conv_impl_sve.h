// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace sve {
template <typename Dtype>
void conv1x1s1_gemm_sve(const Dtype* din,
                        Dtype* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        const Dtype* weights,
                        const Dtype* bias,
                        const operators::ConvParam& param,
                        ARMContext* ctx);

template <typename Dtype>
void conv_im2col_gemm_sve(const Dtype* din,
                          Dtype* dout,
                          int num,
                          int chout,
                          int hout,
                          int wout,
                          int chin,
                          int hin,
                          int win,
                          const Dtype* weights,
                          const Dtype* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx);

}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
