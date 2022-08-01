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

template <typename DataT, typename IndexT = int32_t>
void GatherNd(const Tensor& x, const Tensor& index, Tensor* out) {
  auto index_dims = index.dims();
  auto index_dims_size = index_dims.size();
  auto x_dims = x.dims();
  auto x_dims_size = x_dims.size();

  const DataT* x_data = x.data<DataT>();
  const IndexT* index_data = index.data<IndexT>();
  DataT* out_data = out->template mutable_data<DataT>();

  int64_t gather_time = 1;
  for (size_t i = 0; i < index_dims_size - 1; i++) {
    gather_time *= index_dims[i];
  }

  int64_t end_size = index_dims[index_dims_size - 1];
  int64_t gather_size = 1;
  for (size_t i = end_size; i < x_dims_size; i++) {
    gather_size *= x_dims[i];
  }
  const size_t gather_bytes = gather_size * sizeof(DataT);

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

void GatherNdCompute::Run() {
  auto& param = this->template Param<operators::GatherNdParam>();
  auto* x = param.x;
  auto* index = param.index;
  auto* out = param.out;

#define SELECT_GATHERND(index_data_type)                      \
  switch (x->precision()) {                                   \
    case PRECISION(kFloat):                                   \
      GatherNd<float, index_data_type>(*x, *index, out);      \
      break;                                                  \
    case PRECISION(kFP64):                                    \
      GatherNd<double, index_data_type>(*x, *index, out);     \
      break;                                                  \
    case PRECISION(kInt64):                                   \
      GatherNd<int64_t, index_data_type>(*x, *index, out);    \
      break;                                                  \
    case PRECISION(kInt32):                                   \
      GatherNd<int32_t, index_data_type>(*x, *index, out);    \
      break;                                                  \
    case PRECISION(kUInt8):                                   \
      GatherNd<uint8_t, index_data_type>(*x, *index, out);    \
      break;                                                  \
    case PRECISION(kInt8):                                    \
      GatherNd<int8_t, index_data_type>(*x, *index, out);     \
      break;                                                  \
    case PRECISION(kBool):                                    \
      GatherNd<bool, index_data_type>(*x, *index, out);       \
      break;                                                  \
    default:                                                  \
      LOG(FATAL) << "unsupported input(x) type: "             \
                 << lite_api::PrecisionToStr(x->precision()); \
      break;                                                  \
  }

  switch (index->precision()) {
    case PRECISION(kInt32): {
      SELECT_GATHERND(int32_t)
      break;
    }
    case PRECISION(kInt64): {
      SELECT_GATHERND(int64_t)
      break;
    }
    default: {
      LOG(FATAL) << "unsupported index type: "
                 << lite_api::PrecisionToStr(index->precision());
    }
  }
#undef SELECT_GATHERND
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using GatherNdCompute_ = paddle::lite::kernels::host::GatherNdCompute;
REGISTER_LITE_KERNEL(gather_nd, kHost, kAny, kAny, GatherNdCompute_, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
