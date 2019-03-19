/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef CONV_OP

#pragma once

#include <vector>
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

int ConvOutputSize(int input_size, int filter_size, int dilation, int padding,
                   int stride);

bool IsExpand(const std::vector<int64_t> &filter_dim,
              const std::vector<int> &strides, const std::vector<int> &paddings,
              const std::vector<int> &dilations);

template <typename Itype, typename Otype>
void GemmConv(const ConvParam<CPU> &param);

template <int tile, int kernel>
void WinogradConv3x3(const ConvParam<CPU> &param);

template <typename Itype, typename Otype>
void DepthwiseConv3x3(const ConvParam<CPU> &param);

template <typename Itype, typename Otype>
void DepthwiseConv5x5(const ConvParam<CPU> &param);

template <typename Itype, typename Otype>
void SlidingwindowConv3x3(const ConvParam<CPU> &param);

}  // namespace operators
}  // namespace paddle_mobile

#endif
