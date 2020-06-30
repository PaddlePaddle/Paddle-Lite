// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T>
void SequenceUnpadding(T* seq_data,
                       const T* pad_data,
                       const size_t* seq_offsets_data,
                       int seq_num,
                       int pad_seq_len,
                       int step_width,
                       cudaStream_t* stream);

template <typename T>
void SequencePadding(T* pad_data,
                     const T* seq_data,
                     const T* pad_value_data,
                     bool is_constant_pad,
                     const size_t* seq_offsets_data,
                     int seq_num,
                     int pad_seq_len,
                     int step_width,
                     cudaStream_t* stream);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
