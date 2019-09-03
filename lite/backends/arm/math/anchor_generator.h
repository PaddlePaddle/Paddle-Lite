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
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void anchor_generator_func(int feature_height,
                           int feature_widht,
                           std::vector<float> anchor_sizes,
                           std::vector<float> aspect_ratios,
                           std::vector<float> stride,
                           std::vector<float> variances,
                           float offset,
                           float* anchors_data,
                           float* variances_data);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
