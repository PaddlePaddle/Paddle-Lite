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

template <typename Dtype>
void col2im(const Dtype* data_col,
            const int channels,
            const int height,
            const int width,
            const int kernel_h,
            const int kernel_w,
            const int pad_h0,
            const int pad_h1,
            const int pad_w0,
            const int pad_w1,
            const int stride_h,
            const int stride_w,
            const int dilation_h,
            const int dilation_w,
            Dtype* data_im);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
