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

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void compute_across_channels(const T* din,
                             T* dout,
                             int num,
                             int channel,
                             int h,
                             int w,
                             int local_size,
                             float alpha,
                             float beta,
                             float k);

template <typename T>
void compute_within_channels(const T* din,
                             T* dout,
                             int num,
                             int channel,
                             int h,
                             int w,
                             int local_size,
                             float alpha,
                             float beta,
                             float k);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
