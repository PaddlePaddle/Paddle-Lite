// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/tile_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void TileCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto repeat_times = param.repeat_times;
  if (param.RepeatTimes) {
    auto repeat_times_size = param.RepeatTimes->data_size();
    for (int64_t i = 0; i < repeat_times_size; i++) {
      repeat_times.push_back(param.RepeatTimes->template data<int>()[i]);
    }
  } else if (param.repeat_times_tensor.size() != 0) {
    for (int i = 0; i < param.repeat_times_tensor.size(); i++) {
      auto temp = param.repeat_times_tensor[i];
      repeat_times.push_back(*(temp->template data<int>()));
    }
  }
  auto in_dims = param.X->dims();
  auto vec_in_dims = in_dims.Vectorize();
  // broadcast for vec_in_dims.size() equal to repeat_times.size()
  if (repeat_times.size() < vec_in_dims.size()) {
    int diff = vec_in_dims.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  }

  std::vector<int> new_in_dims(vec_in_dims.begin(), vec_in_dims.end());
  std::vector<int> out_dims(param.Out->dims().data().begin(),
                            param.Out->dims().data().end());
  int r = xdnn::broadcast<T>(ctx.GetRawContext(),
                             param.X->template data<T>(),
                             param.Out->template mutable_data<T>(TARGET(kXPU)),
                             new_in_dims,
                             out_dims);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using tile_float =
    paddle::lite::kernels::xpu::TileCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kXPU, kFloat, kNCHW, tile_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using tile_fp16 =
    paddle::lite::kernels::xpu::TileCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(tile, kXPU, kFP16, kNCHW, tile_fp16, fp16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
