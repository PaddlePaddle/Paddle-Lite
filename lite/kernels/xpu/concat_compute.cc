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

#include "lite/kernels/xpu/concat_compute.h"

#include <algorithm>
#include <vector>

#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType, PrecisionType PType>
void ConcatCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto ins = param.x;
  auto out = param.output;
  int64_t axis = param.axis < 0
                     ? param.axis + static_cast<int>(ins[0]->dims().size())
                     : param.axis;

  std::vector<const InType*> x_list;
  std::vector<std::vector<int>> xdims_list;
  for (size_t i = 0; i < ins.size(); i++) {
    if (ins[i]->numel() > 0) {
      xdims_list.push_back(std::vector<int>());
      for (size_t j = 0; j < ins[i]->dims().size(); j++) {
        xdims_list.back().push_back(ins[i]->dims()[j]);
      }
      if (std::is_same<InType, int64_t>::value) {
        xdims_list[i].back() = xdims_list[i].back() * 2;
      }
      x_list.push_back(ins[i]->template data<InType>());
    }
  }
  if (x_list.size() > 1) {
    int r = 0;
    // int64 is not supported in xpu1, use float.
    if (std::is_same<InType, int64_t>::value) {
      std::vector<const float*> x_list_f(x_list.size());
      for (size_t i = 0; i < x_list.size(); ++i) {
        x_list_f[i] = reinterpret_cast<const float*>(x_list[i]);
      }
      r = xdnn::concat<float>(
          ctx.GetRawContext(),
          x_list_f,
          reinterpret_cast<float*>(
              out->template mutable_data<InType>(TARGET(kXPU))),
          xdims_list,
          axis);

    } else {
      r = xdnn::concat<InType>(
          ctx.GetRawContext(),
          x_list,
          reinterpret_cast<InType*>(
              out->template mutable_data<InType>(TARGET(kXPU))),
          xdims_list,
          axis);
    }
    CHECK_EQ(r, 0);
  } else if (x_list.size() == 1) {
    int r = xdnn::copy<InType>(ctx.GetRawContext(),
                               reinterpret_cast<const InType*>(x_list[0]),
                               out->template mutable_data<InType>(TARGET(kXPU)),
                               out->numel());

    CHECK_EQ(r, 0);
  } else {
    out->set_target(TARGET(kXPU));
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
using concatfp32 =
    paddle::lite::kernels::xpu::ConcatCompute<float, PRECISION(kFloat)>;
using concatfp16 =
    paddle::lite::kernels::xpu::ConcatCompute<float16, PRECISION(kFP16)>;
using concati16 =
    paddle::lite::kernels::xpu::ConcatCompute<int, PRECISION(kInt16)>;
using concati32 =
    paddle::lite::kernels::xpu::ConcatCompute<int, PRECISION(kFloat)>;
using concati64 =
    paddle::lite::kernels::xpu::ConcatCompute<int64_t, PRECISION(kFloat)>;
using concati8 =
    paddle::lite::kernels::xpu::ConcatCompute<int8_t, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(concat, kXPU, kFloat, kNCHW, concatfp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(concat, kXPU, kFP16, kNCHW, concatfp16, concat_FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    concat, kXPU, kInt16, kNCHW, concati16, DISABLE_XPU1_concat_INT16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt16))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt16))})
    .Finalize();

REGISTER_LITE_KERNEL(concat, kXPU, kFloat, kNCHW, concati32, concat_INT32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(concat, kXPU, kFloat, kNCHW, concati64, concat_INT64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(concat, kXPU, kInt8, kNCHW, concati8, concat_INT8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();
