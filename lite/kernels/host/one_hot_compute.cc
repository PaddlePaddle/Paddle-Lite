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

#include <map>
#include <utility>
#include <vector>

#include "lite/kernels/host/one_hot_compute.h"
#include "lite/utils/paddle_enforce.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void OneHotCompute::Run() {
  auto& param = Param<operators::OneHotParam>();
  param.Out->mutable_data<float>();
  int depth = param.depth;
  if (param.depth_tensor) {
    auto* depth_tensor = param.depth_tensor;
    auto* depth_data = depth_tensor->data<int32_t>();
    depth = depth_data[0];
    auto in_dims = param.X->dims();
    DDim out_dims(in_dims);
    out_dims[out_dims.size() - 1] = depth;
    param.Out->Resize(out_dims);
  }

  auto* p_in_data = param.X->data<float>();
  auto numel = param.X->numel();
  auto* p_out_data = param.Out->mutable_data<float>();

  for (int i = 0; i < param.Out->numel(); ++i) {
    p_out_data[i] = 0;
  }

  if (param.allow_out_of_range) {
    for (int i = 0; i < numel; ++i) {
      if (p_in_data[i] >= 0 && p_in_data[i] < param.depth) {
        *(p_out_data + i * param.depth + (int)(p_in_data[i])) = 1.0;  // NOLINT
      }
    }
  } else {
    for (int i = 0; i < numel; ++i) {
      PADDLE_ENFORCE_GE(
          p_in_data[i], 0, "Illegal index value, should be at least 0.");
      PADDLE_ENFORCE_LT(p_in_data[i],
                        param.depth,
                        "Illegal index value, should be less than depth (%d).",
                        param.depth);
      *(p_out_data + i * param.depth + (int)(p_in_data[i])) = 1.0;  // NOLINT
    }
  }
}
}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(one_hot,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::OneHotCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
