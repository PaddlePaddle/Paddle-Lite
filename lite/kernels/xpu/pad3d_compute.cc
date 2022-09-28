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

#include "lite/kernels/xpu/pad3d_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void Pad3dCompute<T>::Run() {
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
  T* out_data = out->template mutable_data<T>(TARGET(kXPU));
  bool is_ncdhw;
  int n, c, d, h, w;
  if (data_format == "NCDHW") {
    is_ncdhw = true;
    n = in_dims[0];
    c = in_dims[1];
    d = in_dims[2];
    h = in_dims[3];
    w = in_dims[4];
  } else if (data_format == "NDHWC") {
    is_ncdhw = false;
    n = in_dims[0];
    c = in_dims[4];
    d = in_dims[1];
    h = in_dims[2];
    w = in_dims[3];
  } else {
    LOG(FATAL) << "xpu unsupport data_format: " << data_format;
  }
  // trans pad format
  std::vector<int> padding(6);
  padding[0] = pads[4];
  padding[1] = pads[5];
  padding[2] = pads[2];
  padding[3] = pads[3];
  padding[4] = pads[0];
  padding[5] = pads[1];

  if (mode == "constant") {
    int r = xdnn::constant_pad3d<T>(ctx.GetRawContext(),
                                    in_data,
                                    out_data,
                                    n,
                                    c,
                                    d,
                                    h,
                                    w,
                                    padding,
                                    value,
                                    is_ncdhw);
    CHECK_EQ(r, 0);
  } else if (mode == "reflect") {
    int r = xdnn::reflection_pad3d<T>(ctx.GetRawContext(),
                                      in_data,
                                      out_data,
                                      n,
                                      c,
                                      d,
                                      h,
                                      w,
                                      padding,
                                      is_ncdhw);
    CHECK_EQ(r, 0);
  } else if (mode == "replicate") {
    int r = xdnn::replication_pad3d<T>(ctx.GetRawContext(),
                                       in_data,
                                       out_data,
                                       n,
                                       c,
                                       d,
                                       h,
                                       w,
                                       padding,
                                       is_ncdhw);
    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "xpu unsupport mode: " << mode;
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pad3d,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::Pad3dCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
