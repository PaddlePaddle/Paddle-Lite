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
#include "lite/core/context.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void topk(const float* din,
          float* out_val,
          int64_t* out_ind,
          int m,
          int n,
          int k,
          Context<TARGET(kARM)>* ctx);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
