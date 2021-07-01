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

#include <vector>
#include "core/hal/types.h"
#include "rknpu/rknpu_pub.h"

namespace nnadapter {
namespace rockchip_npu {

rk::nn::PrecisionType ConvertPrecision(
    NNAdapterOperandPrecisionCode input_precision);
rk::nn::DataLayoutType ConvertDataLayout(
    NNAdapterOperandLayoutCode input_layout);
std::vector<int32_t> ConvertDimensions(int32_t* input_dimensions,
                                       uint32_t input_dimensions_count);

}  // namespace rockchip_npu
}  // namespace nnadapter
