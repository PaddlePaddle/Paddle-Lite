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

#include <float.h>
#include "core/types.h"

namespace nnadapter {

// For the quatized model, folding batch norm causes a large difference in the
// quantized scale of each channel of the filter of conv2d. Some hardware is
// more sensitive to this deviation, which may lead to wrong results. In order
// to solve the problem, we use a threshold of max_scale/min_scale to determine
// whether to perform conv+bn folding.
void FuseConv2DBatchNormIntoConv2D(
    core::Model *model, double max_allowed_quant_scale_deviation = -1.0f);

}  // namespace nnadapter
