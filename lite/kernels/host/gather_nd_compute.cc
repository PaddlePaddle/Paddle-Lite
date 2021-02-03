// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/gather_nd_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, typename IndexT = int32_t>
void GatherNd(const Tensor& x, const Tensor& index, Tensor* out) {
  auto index_dims = index.dims();
  auto index_dims_size = index_dims.size();
  auto x_dims = x.dims();
  auto x_dims_size = x_dims.size();

  const T* x_data = x.data<T>();
  const IndexT* index_data = index.data<IndexT>();
  T* out_data = out->template mutable_data<T>();

  int64_t gather_time = 1;
  for (size_t i = 0; i < index_dims_size - 1; i++) {
    gather_time *= index_dims[i];
  }

  int64_t end_size = index_dims[index_dims_size - 1];
  int64_t gather_size = 1;
  for (size_t i = end_size; i < x_dims_size; i++) {
    gather_size *= x_dims[i];
  }
  const size_t gather_bytes = gather_size * sizeof(T);

  for (int64_t i = 0; i < gather_time; i++) {
    int64_t x_index = 0;
    int64_t step = 1;
    for (int64_t j = end_size - 1; j >= 0; j--) {
      x_index += (index_data[i * end_size + j] * step);
      step *= x_dims[j];
    }
    memcpy(out_data, x_data + x_index * gather_size, gather_bytes);
    out_data += gather_size;
  }
  return;
}

template <typename T, PrecisionType Ptype>
void GatherNdCompute<T, Ptype>::Run() {
  auto& param = this->template Param<operators::GatherNdParam>();
  auto* x = param.x;
  auto* index = param.index;
  auto* out = param.out;

  switch (index->precision()) {
    case PRECISION(kInt32):
      GatherNd<T, int32_t>(*x, *index, out);
      break;
    case PRECISION(kInt64):
      GatherNd<T, int64_t>(*x, *index, out);
      break;
    default:
      LOG(FATAL) << "unsupported index type: "
                 << lite_api::PrecisionToStr(index->precision());
      break;
  }
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using GatherNdFloat32 =
    paddle::lite::kernels::host::GatherNdCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(gather_nd, kHost, kFloat, kAny, GatherNdFloat32, float32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();
