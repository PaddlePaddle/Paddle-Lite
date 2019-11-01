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

#include <random>
template <typename Dtype>
inline void fill_data_const(Dtype* dio, Dtype value, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    dio[i] = value;
  }
}

template <typename Dtype>
inline void fill_data_rand(Dtype* dio, Dtype vstart, Dtype vend, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1.f);
  for (size_t i = 0; i < size; ++i) {
    dio[i] = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
  }
}
