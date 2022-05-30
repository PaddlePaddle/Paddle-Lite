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

#include "lite/kernels/xpu/sequence_reverse_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
void SequenceReverseCompute<T>::PrepareForRun() {
  lod_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
}

template <typename T>
void SequenceReverseCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* x = param.X;
  auto* y = param.Out;

  auto lod = x->lod()[0];
  size_t limit = x->numel();
  size_t ele_cnt_in_4_byte = limit / x->dims()[0];
  auto* x_data = x->template data<T>();
  auto* y_data = y->template mutable_data<T>(TARGET(kXPU));
  int lod_len = lod.size();

  if (std::is_same<T, uint8_t>::value) {
    ele_cnt_in_4_byte /= 4;
  } else if (std::is_same<T, int>::value) {
    // remain the same
  } else if (std::is_same<T, int64_t>::value) {
    ele_cnt_in_4_byte *= 2;
  } else if (std::is_same<T, float>::value) {
    // remain the same
  } else if (std::is_same<T, double>::value) {
    ele_cnt_in_4_byte *= 2;
  }

  for (size_t i = 0; i < lod.size(); ++i) {
    lod_cpu[i] = lod[i];
  }

  int r =
      xdnn::sequence_reverse<float, int>(ctx.GetRawContext(),
                                         reinterpret_cast<const float*>(x_data),
                                         reinterpret_cast<float*>(y_data),
                                         {lod_cpu.get(), lod_len, nullptr},
                                         ele_cnt_in_4_byte);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using SequenceReverseFp32 = xpu::SequenceReverseCompute<float>;
using SequenceReverseInt32 = xpu::SequenceReverseCompute<int>;
using SequenceReverseInt64 = xpu::SequenceReverseCompute<int64_t>;

REGISTER_LITE_KERNEL(
    sequence_reverse, kXPU, kFloat, kNCHW, SequenceReverseFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    sequence_reverse, kXPU, kFloat, kNCHW, SequenceReverseInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(
    sequence_reverse, kXPU, kFloat, kNCHW, SequenceReverseInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
