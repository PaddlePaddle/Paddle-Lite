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

#include "lite/kernels/host/sequence_expand_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void SequenceExpandFunc(const Tensor& x,
                        const std::vector<uint64_t>& x_lod,
                        const std::vector<uint64_t>& ref_lod,
                        Tensor* out) {
  uint64_t out_offset = 0;
  int64_t x_item_length = x.numel() / x.dims()[0];
  auto out_data = out->mutable_data<T>();
  auto x_data = x.data<T>();
  for (size_t i = 1; i < ref_lod.size(); ++i) {
    uint64_t repeat_num = ref_lod[i] - ref_lod[i - 1];
    uint64_t x_start = x_lod[i - 1];
    uint64_t x_end = x_lod[i];
    uint64_t x_seq_len = x_end - x_start;
    if (repeat_num > 0) {
      uint64_t out_start = out_offset;
      if (out->lod().size() == 1) {
        out_start = out->lod()[0][out_offset];
      }
      for (uint64_t j = 0; j < repeat_num; j++) {
        for (uint64_t k = 0; k < x_seq_len; k++) {
          for (int l = 0; l < x_item_length; l++) {
            out_data[(out_start + j * x_seq_len + k) * x_item_length + l] =
                x_data[(x_start + k) * x_item_length + l];
          }
        }
      }
    }
    out_offset += repeat_num;
  }
}

template <typename T, PrecisionType PType>
void SequenceExpandCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::SequenceExpandParam>();
  auto* x = param.X;
  auto* y = param.Y;
  auto* out = param.Out;
  int ref_level = param.ref_level;
  auto x_lod = x->lod();
  auto y_lod = y->lod();

  if (ref_level == -1) ref_level = y_lod.size() - 1;

  out->template mutable_data<T>();
  if (y_lod[ref_level].size() <= 1) {
    out->CopyDataFrom(*x);
    return;
  }

  std::vector<uint64_t> out_lod;
  if (x_lod.size() == 1) {
    out_lod.push_back(0);
    uint64_t out_offset = 0;
    for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
      uint64_t repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
      uint64_t x_start = x_lod[0][i - 1];
      uint64_t x_end = x_lod[0][i];
      uint64_t x_seq_len = x_end - x_start;
      for (uint64_t j = 0; j < repeat_num; ++j) {
        out_lod.push_back(out_lod.back() + x_seq_len);
        out_offset++;
      }
    }
    // write lod to out if x has lod
    auto& ref_lod = *out->mutable_lod();
    ref_lod[0] = out_lod;
  }

  std::vector<uint64_t> ref_x_lod;
  if (x->lod().size() == 1) {
    ref_x_lod = x->lod()[0];
  } else {
    ref_x_lod.resize(x->dims()[0] + 1);
    std::iota(ref_x_lod.begin(), ref_x_lod.end(), 0);
  }

  SequenceExpandFunc<T>(*x, ref_x_lod, y_lod[ref_level], out);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using sequence_expand_float32 =
    paddle::lite::kernels::host::SequenceExpandCompute<float,
                                                       PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    sequence_expand, kHost, kFloat, kNCHW, sequence_expand_float32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

using sequence_expand_int32 =
    paddle::lite::kernels::host::SequenceExpandCompute<int32_t,
                                                       PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    sequence_expand, kHost, kFloat, kNCHW, sequence_expand_int32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using sequence_expand_int64 =
    paddle::lite::kernels::host::SequenceExpandCompute<int64_t,
                                                       PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    sequence_expand, kHost, kFloat, kNCHW, sequence_expand_int64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
