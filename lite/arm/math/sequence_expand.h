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

#include <vector>
#include "lite/core/tensor.h"

#pragma once

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void SequenceExpandImpl(const T* x_data,
                        const LoD& x_lod,
                        int width,
                        const std::vector<uint64_t>& ref_lod,
                        lite::Tensor* output);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
