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
#include "operators/concat/kernels.h"

namespace adnn {

template <>
ADNN_DLL_EXPORT Status
concat<float>(Context* context,
              const std::vector<const float*>& x_datas,
              const std::vector<std::vector<int64_t>>& x_shapes,
              float* y_data,
              int64_t axis) {
  Status status = SUCCESS;
  ADNN_VLOG(5)
      << "concat<float>() is not accelerated on the current architecture!";
  status = kernels::concat<float>(context, x_datas, x_shapes, y_data, axis);
  return status;
}

ADNN_DLL_EXPORT Status
concat_qs8(Context* context,
           const std::vector<const int8_t*>& x_datas,
           const std::vector<std::vector<int64_t>>& x_shapes,
           const std::vector<std::vector<float>>& x_scales,
           int8_t* y_data,
           float y_scale,
           int64_t axis) {
  Status status = SUCCESS;
  ADNN_VLOG(5)
      << "concat_qs8() is not accelerated on the current architecture!";
  status = kernels::concat<int8_t>(context, x_datas, x_shapes, y_data, axis);
  return status;
}

}  // namespace adnn
