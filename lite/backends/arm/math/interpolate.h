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
#include <string>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void bilinear_interp(const float* src,
                     int w_in,
                     int h_in,
                     float* dst,
                     int w_out,
                     int h_out,
                     float scale_x,
                     float scale_y,
                     bool with_align);

void nearest_interp(const float* src,
                    int w_in,
                    int h_in,
                    float* dst,
                    int w_out,
                    int h_out,
                    float scale_x,
                    float scale_y,
                    bool with_align);

void interpolate(lite::Tensor* X,
                 lite::Tensor* OutSize,
                 std::vector<const lite::Tensor*> SizeTensor,
                 lite::Tensor* Scale,
                 lite::Tensor* Out,
                 int out_height,
                 int out_width,
                 float scale,
                 bool with_align,
                 std::string interpolate_type);

} /* namespace math */
} /* namespace arm */
} /* namespace lite */
} /* namespace paddle */
