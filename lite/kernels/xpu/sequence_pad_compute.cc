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

#include "lite/kernels/xpu/sequence_pad_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SequencePadCompute::PrepareForRun() {
  lod_cpu.reset(new int[XPU_MAX_LOD_SIZE_64]);
}

void SequencePadCompute::Run() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();

  auto* x = param.X;
  const float* x_ptr = x->data<float>();
  CHECK(!x->lod().empty()) << "Input X should have lod data.";
  const auto& x_lod_0 = x->lod()[0];
  int seq_num = x_lod_0.size() - 1;
  int max_seq_len = 0;
  for (int i = 0; i < seq_num; ++i) {
    max_seq_len =
        (std::max)(max_seq_len, static_cast<int>(x_lod_0[i + 1] - x_lod_0[i]));
  }
  auto& x_dims = x->dims();
  int dim = x->numel() / x_dims[0];

  int lod_len = x_lod_0.size();
  CHECK_LE(lod_len, XPU_MAX_LOD_SIZE_64)
      << "lod length is expand XPU_MAX_LOD_SIZE_64";
  for (int i = 0; i < lod_len; ++i) {
    lod_cpu[i] = static_cast<int>(x_lod_0[i]);
  }

  auto* pad_value = param.PadValue;
  float* pad_value_ptr = const_cast<float*>(pad_value->data<float>());
  int pad_value_size = pad_value->numel();
  CHECK_EQ(pad_value_size, 1) << "The numel of pad_value only be 1 for XPU.";

  auto* out = param.Out;
  float* out_ptr = out->mutable_data<float>(TARGET(kXPU));

  int padded_length = param.padded_length;
  CHECK_EQ(((padded_length == -1) || (padded_length >= max_seq_len)), true)
      << "padded_length(" << padded_length
      << ") should be -1 or  not less than max_seq_len(" << max_seq_len
      << ") for XPU.";

  int ret = xdnn::sequence_pad<float, int>(ctx.GetRawContext(),
                                           x_ptr,
                                           out_ptr,
                                           {lod_cpu.get(), lod_len, nullptr},
                                           seq_num,
                                           padded_length,
                                           dim,
                                           pad_value_ptr[0]);
  CHECK_EQ(ret, 0) << "call xdnn::sequence_pad failed!";

  auto* length = param.Length;
  auto* length_ptr = length->template mutable_data<int64_t>();
  for (size_t i = 1; i < x_lod_0.size(); i++) {
    length_ptr[i - 1] = x_lod_0[i] - x_lod_0[i - 1];
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_pad,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SequencePadCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("PadValue",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Length",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
