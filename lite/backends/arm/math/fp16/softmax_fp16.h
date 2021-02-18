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

#pragma once

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
typedef __fp16 float16_t;

void softmax_basic_fp16(const float16_t* din,
                        float16_t* dout,
                        const int axis_size,
                        const int inner_num,
                        const int outer_num);

void softmax_inner8_axis4_fp16(const float16_t* din,
                               float16_t* dout,
                               const int axis_size,
                               const int inner_num,
                               const int outer_num);

void softmax_inner8_axis1_fp16(const float16_t* din,
                               float16_t* dout,
                               const int axis_size,
                               const int inner_num,
                               const int outer_num);

void softmax_inner1_large_axis_fp16(const float16_t* din,
                                    float16_t* dout,
                                    const int outer_size,
                                    const int axis_size);

void softmax_inner1_small_axis_fp16(const float16_t* din,
                                    float16_t* dout,
                                    const int outer_size,
                                    const int axis_size);

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
