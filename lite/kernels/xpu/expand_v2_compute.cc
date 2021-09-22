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

#include "lite/kernels/xpu/expand_v2_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
void ExpandV2Compute<T>::Run() {
  auto& param = this->template Param<operators::ExpandV2Param>();
  auto& ctx = this->ctx_->As<XPUContext>();
  const auto* x = param.X;
  auto* out = param.Out;
  std::vector<int> expand_shape;
  if (param.Shape != nullptr) {
    auto Shape_data = param.Shape->template data<int>();
    for (int64_t i = 0; i < param.Shape->numel(); i++) {
      expand_shape.push_back(Shape_data[i]);
    }
  } else if (!param.expand_shapes_tensor.empty()) {
    for (size_t i = 0; i < param.expand_shapes_tensor.size(); i++) {
      expand_shape.push_back(
          param.expand_shapes_tensor[i]->template data<int>()[0]);
    }
  } else {
    expand_shape = param.shape;
  }
  std::vector<int> vec_in_dims;
  DDim in_shape = x->dims();
  for (int i = 0; i < in_shape.size(); ++i) {
    vec_in_dims.push_back(static_cast<int>(in_shape[i]));
  }
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> repeat_times(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    if (i < diff) {
      repeat_times[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {
      if (vec_in_dims[i] != 1) {
        repeat_times[i] = 1;
      } else {
        repeat_times[i] = expand_shape[i];
      }
    } else {
      repeat_times[i] = 1;
    }
  }
  std::vector<int> vec_out_dims(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); i++) {
    vec_out_dims[i] = vec_in_dims[i] * repeat_times[i];
  }

  int r = xdnn::broadcast<T>(ctx.GetRawContext(),
                             x->template data<T>(),
                             out->template mutable_data<T>(TARGET(kXPU)),
                             vec_in_dims,
                             vec_out_dims);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using expand_v2_xpu_float = paddle::lite::kernels::xpu::ExpandV2Compute<float>;
REGISTER_LITE_KERNEL(expand_v2, kXPU, kFloat, kAny, expand_v2_xpu_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_shapes_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();
