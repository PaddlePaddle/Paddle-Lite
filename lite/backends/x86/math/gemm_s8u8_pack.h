/* Copyright (c) 2021 paddlepaddle Authors. All Rights Reserved.

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

#include <stdint.h>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

#define TRANS_INT8_UINT8_OFFT (128)

// PackA 's K dim need 4-aligned,
// so it needs M * K_4aligned Bytes.
void gemm_s8u8s8_prepackA(
    int M, int K, const int8_t* A, int8_t* pack_A, bool is_trans);

void gemm_s8u8s8_runpackB(
    int N, int K, int stride, const int8_t* B, uint8_t* pack_B, bool is_trans);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
