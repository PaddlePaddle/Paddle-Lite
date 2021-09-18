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

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void group_norm(const float* in,
                float* out,
                const int n,
                const int c,
                const int height,
                const int width,
                const float epsilon,
                const int groups,
                const float* scale,
                const float* bias,
                float* saved_mean,
                float* saved_variance);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
