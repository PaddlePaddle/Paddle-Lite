/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef SPLIT_OP
#pragma once

#include <vector>
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

// Strided numel memory copy from src to dst by the specified axis
//
// For example, for a tensor dims [4, 20, 100], the strieded numel is
// [8000, 2000, 100]
//
// NOTE: The src and dst tensor should have the same elements
// except the specified axis.
template <typename T>
inline void StridedNumelCopyWithAxis(int64_t axis, T* dst,
                                     const framework::DDim& dst_stride_numel,
                                     const T* src,
                                     const framework::DDim& src_stride_numel,
                                     int64_t size) {
  int64_t before = dst_stride_numel[0] / dst_stride_numel[axis];
  int64_t src_after = src_stride_numel[axis];
  int64_t dst_after = dst_stride_numel[axis];

  PADDLE_MOBILE_ENFORCE(src_stride_numel.size() == dst_stride_numel.size(),
                        "src and dst tensor should have the same dims size.");

  for (int64_t i = 0; i < axis; ++i) {
    if (i < axis) {
      PADDLE_MOBILE_ENFORCE(src_stride_numel[i] / src_stride_numel[axis] ==
                                dst_stride_numel[i] / dst_stride_numel[axis],
                            "src and dst should have the same elements "
                            "except the specified axis.");
    } else if (i == axis) {
      continue;
    } else {
      PADDLE_MOBILE_ENFORCE(src_stride_numel[i] == dst_stride_numel[i],
                            "src and dst should have the same elements "
                            "except the specified axis.");
    }
  }

  for (int64_t i = 0; i < before; ++i) {
    memory::Copy(dst + i * dst_after, src + i * src_after, sizeof(T) * size);
  }
}

template <typename P>
void SplitCompute(const SplitParam<CPU>& param) {
  auto* in = param.InputX();
  auto outs = param.Outs();
  auto in_stride = framework::stride_numel(in->dims());
  int64_t axis = param.Axis();

  size_t input_offset = 0;
  for (auto& out : outs) {
    out->mutable_data<float>();
    auto out_stride = framework::stride_numel(out->dims());

    StridedNumelCopyWithAxis<float>(axis, out->data<float>(), out_stride,
                                    in->data<float>() + input_offset, in_stride,
                                    out_stride[axis]);
    input_offset += out_stride[axis];
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
