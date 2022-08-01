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

#include <imgdnn.h>
#include <vector>
#include "core/types.h"

namespace nnadapter {
namespace imagination_nna {

// Convert NNAdapter types to imgdnn types
imgdnn_type ConvertToImgdnnPrecision(
    NNAdapterOperandPrecisionCode precision_code);
imgdnn_dimensions_order ConvertToImgdnnDataLayout(
    NNAdapterOperandLayoutCode layout_code);
void ConvertToImgdnnDimensions(int32_t* input_dimensions,
                               uint32_t input_dimensions_count,
                               size_t* output_dimensions,
                               unsigned int* output_dimensions_count);

}  // namespace imagination_nna
}  // namespace nnadapter
