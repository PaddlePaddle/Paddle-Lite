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

#include "lite/kernels/host/ctc_align_compute.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

LoD ToAbs(const LoD& in) {
  if (in.empty()) return in;
  LoD result;
  for (auto& src : in) {
    std::vector<uint64_t> dest(src.size() + 1, 0);
    for (int i = 0; i < src.size(); i++) {
      dest[i + 1] = dest[i] + src[i];
    }
    result.emplace_back(dest);
  }
  return result;
}

LoD ToNorm(const LoD& in) {
  if (in.empty()) return in;
  LoD result;
  for (auto& src : in) {
    std::vector<uint64_t> dest(src.size() - 1, 0);
    for (int i = 0; i < dest.size(); i++) {
      dest[i] = src[i + 1] - src[i];
    }
    result.emplace_back(dest);
  }
  return result;
}

LoD ToAbsOffset(const LoD& in) {
  // the lowest level stores relative offsets
  if (in.empty() || in.size() == 1) return in;
  LoD result = in;
  for (auto level = static_cast<int>(in.size() - 2); level >= 0; level--) {
    for (size_t i = 0; i < in[level].size(); ++i) {
      size_t index = in[level][i];
      result[level][i] = result[level + 1][index];
    }
  }
  return result;
}

template <typename T, PrecisionType PT>
void CtcAlignCompute<T, PT>::Run() {
  auto& param = this->template Param<operators::CtcAlignParam>();
  auto* input = param.input;
  auto* output = param.output;
  size_t blank = static_cast<size_t>(param.blank);
  bool merge_repeated = param.merge_repeated;
  size_t padding_value = static_cast<size_t>(param.padding_value);

  const auto* input_data = input->template data<T>();
  auto input_dims = input->dims();
  auto* output_data = output->template mutable_data<T>();

  if (input->lod().empty()) {
    auto* input_length = param.input_length;
    auto* output_length = param.output_length;
    CHECK(input_length != nullptr);
    CHECK(output_length != nullptr);
    const auto* input_length_data = input_length->template data<T>();
    auto* output_length_data = output_length->template mutable_data<T>();

    for (size_t batch_id = 0; batch_id < (unsigned)input_dims[0]; batch_id++) {
      T prev_token = -1;
      size_t output_idx = 0;
      for (size_t i = 0; i < (unsigned)input_length_data[batch_id]; i++) {
        size_t input_ind = batch_id * input_dims[1] + i;
        if ((unsigned)input_data[input_ind] != blank &&
            !(merge_repeated && input_data[input_ind] == prev_token)) {
          output_data[batch_id * input_dims[1] + output_idx] =
              input_data[input_ind];
          ++output_idx;
        }
        prev_token = input_data[input_ind];
      }
      output_length_data[batch_id] = output_idx;
      for (size_t j = output_idx; j < (unsigned)input_dims[1]; j++)
        output_data[batch_id * input_dims[1] + j] = padding_value;
    }
  } else {
    const size_t level = 0;

    auto input_lod = input->lod();
    input_lod = ToAbs(input->lod());
    input_lod = ToAbsOffset(input_lod);
    CHECK_EQ(input_dims[0], static_cast<int64_t>(input_lod[level].back()));

    const size_t num_sequences = input_lod[level].size() - 1;
    // merge repeated tokens and delete blank
    size_t output_idx = 0;
    std::vector<uint64_t> output_lod0(1, 0);
    for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
      T prev_token = -1;
      for (size_t i = input_lod[level][seq_idx];
           i < input_lod[level][seq_idx + 1];
           ++i) {
        if ((unsigned)input_data[i] != blank &&
            !(merge_repeated && input_data[i] == prev_token)) {
          output_data[output_idx] = input_data[i];
          ++output_idx;
        }
        prev_token = input_data[i];
      }
      output_lod0.push_back(static_cast<uint64_t>(output_idx));
    }

    LoD output_lod;
    output_lod.push_back(output_lod0);
    output_lod = ToNorm(output_lod);
    output->set_lod(output_lod);
    output->Resize({static_cast<int64_t>(output_lod0.back()), 1});
    if (output_lod0.back() == 0) {
      output->Resize({1, 1});
      output_data = output->template mutable_data<T>();
      output_data[0] = -1;
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
using ctc_align_int64 =
    paddle::lite::kernels::host::CtcAlignCompute<int64_t, PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(ctc_align, kHost, kInt64, kNCHW, ctc_align_int64, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("InputLength",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("OutputLength",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

using ctc_align_int32 =
    paddle::lite::kernels::host::CtcAlignCompute<int32_t, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(ctc_align, kHost, kInt32, kNCHW, ctc_align_int32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("InputLength",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("OutputLength",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
