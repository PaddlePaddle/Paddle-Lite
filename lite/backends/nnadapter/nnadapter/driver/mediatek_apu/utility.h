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
#include "driver/mediatek_apu/neuron_adapter_wrapper.h"

namespace nnadapter {
namespace mediatek_apu {

int ConvertPrecision(NNAdapterOperandPrecisionCode input_precision);
int ConvertDataLayout(NNAdapterOperandLayoutCode input_layout);
std::vector<uint32_t> ConvertDimensions(int32_t* input_dimensions,
                                        uint32_t input_dimensions_count);
int32_t ConvertFuseCode(int32_t input_code);
int PrecisionLength(int precision);

}  // namespace mediatek_apu
}  // namespace nnadapter
