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

#include "lite/kernels/host/where_index_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

static void where_index_rank4(const int64_t* true_index,
                              int true_num,
                              const int64_t* stride,
                              int64_t* out) {
  int cnt = true_num >> 1;
  register int64_t stride0 = stride[0];
  register int64_t stride1 = stride[1];
  register int64_t stride2 = stride[2];
  register int64_t stride3 = stride[3];
  for (int i = 0; i < cnt; ++i) {
    int64_t index0 = true_index[i * 2];
    int64_t index1 = true_index[i * 2 + 1];
    int out_index = i * 8;
    // rank0
    register int64_t oindex0 = index0 / stride0;
    register int64_t oindex1 = index1 / stride0;
    out[out_index] = oindex0;
    index0 -= oindex0 * stride0;
    index1 -= oindex1 * stride0;
    out[out_index + 4] = oindex1;
    out_index++;
    // rank1
    oindex0 = index0 / stride1;
    oindex1 = index1 / stride1;
    out[out_index] = oindex0;
    index0 -= oindex0 * stride1;
    index1 -= oindex1 * stride1;
    out[out_index + 4] = oindex1;
    out_index++;
    // rank2
    oindex0 = index0 / stride2;
    oindex1 = index1 / stride2;
    out[out_index] = oindex0;
    index0 -= oindex0 * stride2;
    index1 -= oindex1 * stride2;
    out[out_index + 4] = oindex1;
    out_index++;
    // rank3
    oindex0 = index0 / stride3;
    oindex1 = index1 / stride3;
    out[out_index] = oindex0;
    out[out_index + 4] = oindex1;
  }
  // remain
  for (int r = cnt * 2; r < true_num; ++r) {
    int out_index = r * 4;
    int64_t index = true_index[r];
    for (int i = 0; i < 4; ++i) {
      out[out_index + i] = index / stride[i];
      index -= out[out_index + i] * stride[i];
    }
  }
}

inline void where_index_rank1(const int64_t* true_index,
                              int true_num,
                              int64_t* out) {
  memcpy(out, true_index, true_num * sizeof(int64_t));
}

static void where_index_rankn(const int64_t* true_index,
                              int true_num,
                              const int64_t* stride,
                              int rank,
                              int64_t* out) {
  int out_index = 0;
  for (int i = 0; i < true_num; ++i) {
    int64_t index = true_index[i];
    for (int r = 0; r < rank; ++r) {
      out[out_index] = index / stride[r];
      index -= out[out_index++] * stride[r];
    }
  }
}

template <typename T>
void WhereIndexKernel(const operators::WhereIndexParam& param) {
  auto* input = param.input;
  auto* output = param.output;
  auto dims = input->dims();
  auto numel = dims.production();
  int64_t rank = static_cast<int64_t>(dims.size());
  const T* cond_data = input->template data<T>();
  int64_t true_num = 0;
  std::vector<int64_t> true_index(numel);
  for (auto i = 0; i < numel; i++) {
    if (static_cast<bool>(cond_data[i])) {
      true_index[true_num] = i;
      true_num++;
    }
  }
  output->Resize({true_num, rank});
  if (true_num == 0) {
    return;
  }
  auto* out_ptr = output->template mutable_data<int64_t>();
  std::vector<int64_t> stride(rank);
  stride[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * dims[i + 1];
  }
  if (rank == 1) {
    where_index_rank1(true_index.data(), true_num, out_ptr);
  } else if (rank == 4) {
    where_index_rank4(true_index.data(), true_num, stride.data(), out_ptr);
  } else {
    where_index_rankn(
        true_index.data(), true_num, stride.data(), rank, out_ptr);
  }
}

void WhereIndexCompute::Run() {
  auto& param = this->Param<operators::WhereIndexParam>();
  switch (param.input->precision()) {
    case PRECISION(kFloat):
      WhereIndexKernel<float>(param);
      break;
    case PRECISION(kInt32):
      WhereIndexKernel<int32_t>(param);
      break;
    case PRECISION(kInt64):
      WhereIndexKernel<int64_t>(param);
      break;
    case PRECISION(kInt8):
      WhereIndexKernel<int8_t>(param);
      break;
    case PRECISION(kBool):
      WhereIndexKernel<bool>(param);
      break;
    default:
      LOG(FATAL) << "WhereIndex does not implement for the "
                 << "input type:" << static_cast<int>(param.input->precision());
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using whereindex = paddle::lite::kernels::host::WhereIndexCompute;

REGISTER_LITE_KERNEL(where_index, kHost, kAny, kAny, whereindex, def)
    .BindInput("Condition",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
