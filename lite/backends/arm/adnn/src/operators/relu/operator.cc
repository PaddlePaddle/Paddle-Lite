// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "adnn/core/types.h"  // NOLINT
#include "operators/relu/kernels.h"

namespace adnn {

template <>
ADNN_DLL_EXPORT Status
relu<float>(Context* context, const float* x_data, float* y_data, size_t size) {
  Status status = SUCCESS;
#ifdef ANN_WITH_AACH32
  status = kernels::relu_fp32_aarch32_neon_x8(context, x_data, y_data, size);
#elif defined(ANN_WITH_AACH64)
  status = kernels::relu_fp32_aarch64_neon_x16(context, x_data, y_data, size);
#else
  ADNN_VLOG(5) << "relu<float>() is not accelerated on the current "
                  "architecture! Only accelerated on aarch32 and aarch64 !";
  status = kernel::relu<float>(context, x_data, y_data, size);
#endif
  return status;
}

ADNN_DLL_EXPORT Status relu_qs8(Context* context,
                                const int8_t* x_data,
                                float x_scale,
                                int8_t* y_data,
                                float y_scale,
                                size_t size) {
  Status status = SUCCESS;
  ADNN_CHECK_EQ(x_scale, y_scale);
  ADNN_VLOG(5) << "relu_qs8() is not accelerated on the current architecture!";
  status = kernels::relu<int8_t>(context, x_data, y_data, size);
  return status;
}

}  // namespace adnn
