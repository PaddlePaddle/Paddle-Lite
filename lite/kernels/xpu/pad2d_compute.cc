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

#include "lite/kernels/xpu/pad2d_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void Pad2dCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto pads = param.paddings;
  auto mode = param.mode;
  auto data_format = param.data_format;
  T value = static_cast<T>(param.pad_value);

  auto* x = param.X;
  auto in_dims = x->dims();
  auto* in_data = x->template data<T>();
  auto* out = param.Out;
  if (data_format == "NCHW") {
    out->Resize({in_dims[0],
                 in_dims[1],
                 in_dims[2] + pads[0] + pads[1],
                 in_dims[3] + pads[2] + pads[3]});
  } else {
    out->Resize({in_dims[0],
                 in_dims[1] + pads[0] + pads[1],
                 in_dims[2] + pads[2] + pads[3],
                 in_dims[3]});
  }
  T* out_data = out->template mutable_data<T>(TARGET(kXPU));

  if (mode == "reflect" || mode == "constant" || mode == "edge") {
    int r = xdnn::pad2d<T>(ctx.GetRawContext(),
                           in_data,
                           out_data,
                           in_dims[0],
                           in_dims[1],
                           in_dims[2],
                           in_dims[3],
                           pads,
                           mode.c_str(),
                           value,
                           (data_format == "NCHW"));
    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "xpu unsupport mode: " << mode;
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pad2d,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::Pad2dCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
