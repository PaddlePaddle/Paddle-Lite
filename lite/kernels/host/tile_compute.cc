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
#include "lite/kernels/host/tile_compute.h"
#include <algorithm>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void TileCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::TileParam>();
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

  DDim new_in_dims{vec_in_dims};
  DDim out_dims(new_in_dims);
  std::vector<int> bcast_dims(vec_in_dims.size() + 1);
  std::vector<int> in_stride(vec_in_dims.size() + 1);

  in_stride[0] = 1;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    bcast_dims[i] = repeat_times[i];
    out_dims[i] *= repeat_times[i];
    if (i > 0) {
      in_stride[i + 1] = in_stride[i] / new_in_dims[i - 1];
    } else {
      in_stride[i + 1] = new_in_dims.production();
    }
  }
  bcast_dims[repeat_times.size()] = 1;
  auto& in = param.X;
  auto& out = param.Out;
  out->Resize(out_dims);
  Tensor tmp_src_tensor;
  Tensor tmp_dst_tensor;
  auto in_data = in->template data<T>();
  tmp_src_tensor.Resize(out_dims);
  tmp_dst_tensor.Resize(out_dims);
  auto tmp_src = tmp_src_tensor.mutable_data<T>();
  auto tmp_dst = tmp_dst_tensor.mutable_data<T>();
  for (int i = 0; i < in_dims.production(); i++) {
    tmp_src[i] = in_data[i];
    tmp_dst[i] = in_data[i];
  }

  int right = 1;
  for (int i = bcast_dims.size() - 1; i >= 0; i--) {
    right *= bcast_dims[i];
    if (bcast_dims[i] > 1) {
      int num = in_stride[1] / in_stride[i + 1];
      int dst_stride = in_stride[i + 1] * right;
      for (int m = 0; m < num; m++) {
        for (int j = 0; j < bcast_dims[i]; j++) {
          std::memcpy(
              tmp_dst + j * (dst_stride / bcast_dims[i]) + m * dst_stride,
              tmp_src + m * (dst_stride / bcast_dims[i]),
              dst_stride / bcast_dims[i] * sizeof(T));
        }
      }
      tmp_src_tensor.CopyDataFrom(tmp_dst_tensor);
    }
  }
  out->CopyDataFrom(tmp_dst_tensor);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using tile_float =
    paddle::lite::kernels::host::TileCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kNCHW, tile_float, def_float)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();
using tile_int32 =
    paddle::lite::kernels::host::TileCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kNCHW, tile_int32, def_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
using tile_int64 =
    paddle::lite::kernels::host::TileCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kNCHW, tile_int64, def_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

using tile_int8 =
    paddle::lite::kernels::host::TileCompute<int8_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kNCHW, tile_int8, def_int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .Finalize();

using tile_bool =
    paddle::lite::kernels::host::TileCompute<bool, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kHost, kFloat, kNCHW, tile_bool, def_bool)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();
