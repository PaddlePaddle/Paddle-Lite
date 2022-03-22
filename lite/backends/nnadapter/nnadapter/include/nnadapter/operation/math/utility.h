// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

namespace nnadapter {
namespace operation {
namespace math {

// Get the slice of the shape
std::vector<int32_t> slice_of_shape(const std::vector<int32_t>& input_shape,
                                    int start,
                                    int end = -1);
// Get the production of the shape
int64_t production_of_shape(const std::vector<int32_t>& input_shape);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
