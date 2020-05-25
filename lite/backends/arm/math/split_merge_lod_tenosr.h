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

#include <utility>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

std::pair<LoD, std::pair<size_t, size_t>> GetSubLoDAndAbsoluteOffset(
    const LoD &lod, size_t start_idx, size_t end_idx, size_t start_level);

void AppendLoD(LoD *lod, const LoD &lod_length);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
