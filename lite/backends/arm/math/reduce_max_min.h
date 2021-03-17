/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

namespace paddle {
namespace lite {
namespace arm {
namespace math {

enum class MaxMinType : bool { kMin = false, kMax = true };
template <typename DataType>
void reduce_first_of_two(const float* src,
                         float* dst,
                         int first_in,
                         int second_in,
                         MaxMinType compare_functor);

template <typename DataType>
void reduce_second_of_two(const float* src,
                          float* dst,
                          int first_in,
                          int second_in,
                          MaxMinType max_min_selector);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
