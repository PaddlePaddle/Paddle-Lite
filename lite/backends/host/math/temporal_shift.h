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
namespace host {
namespace math {

template <typename InType>
void temporalshiftNCHW_func(const InType* input,
                            InType* output,
                            const int ntchw,
                            const int tchw,
                            const int chw,
                            const int hw,
                            const int t,
                            const int c1,
                            const int c2);

template <typename InType>
void temporalshiftNHWC_func(const InType* input,
                            InType* output,
                            const int nthwc,
                            const int thwc,
                            const int hwc,
                            const int t,
                            const int c,
                            const int c1,
                            const int c2);
}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
