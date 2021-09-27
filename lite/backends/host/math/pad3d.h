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

#include <algorithm>
#include <string>
#include <vector>
#include "lite/operators/op_params.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace host {
namespace math {

void pad_ncdhw_constant(const float* din,
                        float* dout,
                        int n,
                        int c,
                        int in_d,
                        int in_h,
                        int in_w,
                        int out_d,
                        int out_h,
                        int out_w,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right,
                        const int pad_front,
                        const int pad_back,
                        const float pad_value);
void pad_ndhwc_constant(const float* din,
                        float* dout,
                        int n,
                        int c,
                        int in_d,
                        int in_h,
                        int in_w,
                        int out_d,
                        int out_h,
                        int out_w,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right,
                        const int pad_front,
                        const int pad_back,
                        const float pad_value);

void pad_ncdhw_reflect(const float* din,
                       float* dout,
                       int n,
                       int c,
                       int in_d,
                       int in_h,
                       int in_w,
                       int out_d,
                       int out_h,
                       int out_w,
                       const int pad_top,
                       const int pad_bottom,
                       const int pad_left,
                       const int pad_right,
                       const int pad_front,
                       const int pad_back);
void pad_ndhwc_reflect(const float* din,
                       float* dout,
                       int n,
                       int c,
                       int in_d,
                       int in_h,
                       int in_w,
                       int out_d,
                       int out_h,
                       int out_w,
                       const int pad_top,
                       const int pad_bottom,
                       const int pad_left,
                       const int pad_right,
                       const int pad_front,
                       const int pad_back);
void pad_ncdhw_replicate(const float* din,
                         float* dout,
                         int n,
                         int c,
                         int in_d,
                         int in_h,
                         int in_w,
                         int out_d,
                         int out_h,
                         int out_w,
                         const int pad_top,
                         const int pad_bottom,
                         const int pad_left,
                         const int pad_right,
                         const int pad_front,
                         const int pad_back);
void pad_ndhwc_replicate(const float* din,
                         float* dout,
                         int n,
                         int c,
                         int in_d,
                         int in_h,
                         int in_w,
                         int out_d,
                         int out_h,
                         int out_w,
                         const int pad_top,
                         const int pad_bottom,
                         const int pad_left,
                         const int pad_right,
                         const int pad_front,
                         const int pad_back);

void pad_ncdhw_circular(const float* din,
                        float* dout,
                        int n,
                        int c,
                        int in_d,
                        int in_h,
                        int in_w,
                        int out_d,
                        int out_h,
                        int out_w,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right,
                        const int pad_front,
                        const int pad_back);
void pad_ndhwc_circular(const float* din,
                        float* dout,
                        int n,
                        int c,
                        int in_d,
                        int in_h,
                        int in_w,
                        int out_d,
                        int out_h,
                        int out_w,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right,
                        const int pad_front,
                        const int pad_back);

void pad3d_ncdhw_func(const lite::Tensor* input,
                      lite::Tensor* output,
                      int n,
                      int c,
                      int in_d,
                      int in_h,
                      int in_w,
                      int out_d,
                      int out_h,
                      int out_w,
                      int mode,
                      std::vector<int> pad_h,
                      std::vector<int> pad_w,
                      std::vector<int> pad_d,
                      float pad_value);

void pad3d_ndhwc_func(const lite::Tensor* input,
                      lite::Tensor* output,
                      int n,
                      int c,
                      int in_d,
                      int in_h,
                      int in_w,
                      int out_d,
                      int out_h,
                      int out_w,
                      int mode,
                      std::vector<int> pad_h,
                      std::vector<int> pad_w,
                      std::vector<int> pad_d,
                      float pad_value);
}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
