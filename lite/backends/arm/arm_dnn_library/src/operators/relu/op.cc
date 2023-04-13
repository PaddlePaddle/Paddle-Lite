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

#include "operators/relu/op.h"
#include "arm_dnn_library/core/types.h"
#include "arm_dnn_library/operators/nn_ops.h"
#include "operators/relu/kernels.h"
#include "runtime/context.h"
#include "utilities/cpu_info.h"
#include "utilities/logging.h"

namespace armdnnlibrary {

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status
relu<float>(void* context, const float* x_data, float* y_data, size_t size) {
  Status status = SUCCESS;
  auto ctx = reinterpret_cast<Context*>(context);
  ARM_DNN_LIBRARY_CHECK(ctx);
#if ARM_DNN_LIBRARY_ARCH_ARM64
  status = kernels::relu_fp32_aarch64_neon_x16(ctx, x_data, y_data, size);
#elif ARM_DNN_LIBRARY_ARCH_ARM
  status = kernels::relu_fp32_aarch32_neon_x8(ctx, x_data, y_data, size);
#else
  ARM_DNN_LIBRARY_VLOG(5)
      << "relu<float>() is not accelerated on the current "
         "architecture! Only accelerated on ARM and ARM64 !";
  status = kernels::relu<float>(ctx, x_data, y_data, size);
#endif
  return status;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status relu<float16>(void* context,
                                                const float16* x_data,
                                                float16* y_data,
                                                size_t size) {
  auto ctx = reinterpret_cast<Context*>(context);
  ARM_DNN_LIBRARY_CHECK(ctx);
#if ARM_DNN_LIBRARY_ARM_WITH_FP16
  if (ctx->enable_arm_fp16())
    return kernels::relu_fp16_neon_x32(ctx, x_data, y_data, size);
#endif
  ARM_DNN_LIBRARY_VLOG(5)
      << "relu<float16>() is not accelerated on the current "
         "architecture! Only accelerated on ARM and ARM64 !";
  return kernels::relu<float16>(ctx, x_data, y_data, size);
}

ARM_DNN_LIBRARY_DLL_EXPORT Status relu_qs8(void* context,
                                           const int8_t* x_data,
                                           float x_scale,
                                           int8_t* y_data,
                                           float y_scale,
                                           size_t size) {
  Status status = SUCCESS;
  auto ctx = reinterpret_cast<Context*>(context);
  ARM_DNN_LIBRARY_CHECK(ctx);
  ARM_DNN_LIBRARY_CHECK_EQ(x_scale, y_scale);
  ARM_DNN_LIBRARY_VLOG(5)
      << "relu_qs8() is not accelerated on the architecture!";
  status = kernels::relu<int8_t>(ctx, x_data, y_data, size);
  return status;
}

}  // namespace armdnnlibrary
