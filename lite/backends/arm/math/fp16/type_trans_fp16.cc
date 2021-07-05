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
#include "lite/backends/arm/math/fp16/type_trans_fp16.h"

namespace paddle {
namespace lite_metal {
namespace arm {
namespace math {
namespace fp16 {
void fp16_to_fp32(const float16_t* in, float* out, int size) {
  int cnt = size >> 6;
  int remain = size & 63;
  int remain_cnt = remain >> 3;
  int remain_remain = remain & 7;
  asm volatile(
      "cmp %w[cnt], #1\n"
      "blt 1f\n"
      "0: \n"
      "ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [%[in]], #64\n"
      "ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [%[in]], #64\n"
      // 16bit->32bit
      "fcvtl v8.4s, v0.4h\n"
      "fcvtl2 v9.4s, v0.8h\n"
      "fcvtl v10.4s, v1.4h\n"
      "fcvtl2 v11.4s, v1.8h\n"
      "fcvtl v12.4s, v2.4h\n"
      "fcvtl2 v13.4s, v2.8h\n"
      "fcvtl v14.4s, v3.4h\n"
      "fcvtl2 v15.4s, v3.8h\n"
      "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[out]], #64\n"
      "fcvtl v16.4s, v4.4h\n"
      "fcvtl2 v17.4s, v4.8h\n"
      "fcvtl v18.4s, v5.4h\n"
      "fcvtl2 v19.4s, v5.8h\n"
      "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[out]], #64\n"
      "subs %w[cnt], %w[cnt], #1\n"
      "fcvtl v20.4s, v6.4h\n"
      "fcvtl2 v21.4s, v6.8h\n"
      "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[out]], #64\n"
      "fcvtl v22.4s, v7.4h\n"
      "fcvtl2 v23.4s, v7.8h\n"
      "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[out]], #64\n"
      "bne 0b\n"
      "1: \n"
      "cmp %w[remain_cnt], #1\n"
      "blt 2f\n"
      "4: \n"
      "ld1 {v0.8h}, [%[in]], #16\n"
      "subs %w[remain_cnt], %w[remain_cnt], #1\n"
      // 16bit->32bit
      "fcvtl v8.4s, v0.4h\n"
      "fcvtl2 v9.4s, v0.8h\n"
      "st1 {v8.4s, v9.4s}, [%[out]], #32\n"
      "bne 4b\n"
      "2: \n"
      "cmp %w[remain_remain], #1\n"
      "blt 3f\n"
      "5: \n"
      "ldr h0, [%[in]], #2\n"
      "subs %w[remain_remain], %w[remain_remain], #1\n"
      "fcvt s0, h0\n"
      "str s0, [%[out]], #4\n"
      "bne 5b\n"
      "3: \n"
      : [in] "+r"(in),
        [out] "+r"(out),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt),
        [remain_remain] "+r"(remain_remain)
      :
      : "cc",
        "memory",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23");
}

void fp32_to_fp16(const float* in, float16_t* out, int size) {
  int cnt = size >> 6;
  int remain = size & 63;
  int remain_cnt = remain >> 3;
  int remain_remain = remain & 7;
  asm volatile(
      "cmp %w[cnt], #1\n"
      "blt 1f\n"
      "0: \n"
      "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[in]], #64\n"
      "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[in]], #64\n"
      "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[in]], #64\n"
      "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[in]], #64\n"
      // 32bit->16bit
      "fcvtn v16.4h, v0.4s\n"
      "fcvtn2 v16.8h, v1.4s\n"
      "fcvtn v17.4h, v2.4s\n"
      "fcvtn2 v17.8h, v3.4s\n"
      "fcvtn v18.4h, v4.4s\n"
      "fcvtn2 v18.8h, v5.4s\n"
      "fcvtn v19.4h, v6.4s\n"
      "fcvtn2 v19.8h, v7.4s\n"
      "fcvtn v20.4h, v8.4s\n"
      "fcvtn2 v20.8h, v9.4s\n"
      "fcvtn v21.4h, v10.4s\n"
      "fcvtn2 v21.8h, v11.4s\n"
      "subs %w[cnt], %w[cnt], #1\n"
      "st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [%[out]], #64\n"
      "fcvtn v22.4h, v12.4s\n"
      "fcvtn2 v22.8h, v13.4s\n"
      "fcvtn v23.4h, v14.4s\n"
      "fcvtn2 v23.8h, v15.4s\n"
      "st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [%[out]], #64\n"
      "bne 0b\n"
      "1: \n"
      "cmp %w[remain_cnt], #1\n"
      "blt 2f\n"
      "4: \n"
      "ld1 {v0.4s, v1.4s}, [%[in]], #32\n"
      "subs %w[remain_cnt], %w[remain_cnt], #1\n"
      // 32bit->16bit
      "fcvtn v16.4h, v0.4s\n"
      "fcvtn2 v16.8h, v1.4s\n"
      "st1 {v16.8h}, [%[out]], #16\n"
      "bne 4b\n"
      "2: \n"
      "cmp %w[remain_remain], #1\n"
      "blt 3f\n"
      "5: \n"
      "ldr s0, [%[in]], #4\n"
      "subs %w[remain_remain], %w[remain_remain], #1\n"
      "fcvt h0, s0\n"
      "str h0, [%[out]], #2\n"
      "bne 5b\n"
      "3: \n"
      : [in] "+r"(in),
        [out] "+r"(out),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt),
        [remain_remain] "+r"(remain_remain)
      :
      : "cc",
        "memory",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23");
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
