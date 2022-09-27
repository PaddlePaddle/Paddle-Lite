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
  if (data_format == "NCDHW") {
    is_ncdhw = true;
  } else if (data_format == "NDHWC") {
    is_ncdhw = false;
  } else {
    LOG(FATAL) << "xpu unsupport data_format: " << data_format;
  }
  // trans format
  std::vector<int> padding(6);
  padding[0] = pads[4];
  padding[1] = pads[5];
  padding[2] = pads[2];
  padding[3] = pads[3];
  padding[4] = pads[0];
  padding[5] = pads[1];

  if (mode == "constant") {
    if (is_ncdhw) {
      std::vector<int> pad_left = {0, 0, pads[4], pads[2], pads[0]};
      std::vector<int> pad_right = {0, 0, pads[5], pads[3], pads[1]};

      int n_shape = in_dims[0];
      int c_shape = in_dims[1];
      int d_shape = in_dims[2];
      int h_shape = in_dims[3];
      int w_shape = in_dims[4];

      std::vector<int> xshape = {n_shape, c_shape, d_shape, h_shape, w_shape};
      int r = xdnn::pad<T>(ctx.GetRawContext(),
                           in_data,
                           out_data,
                           xshape,
                           pad_left,
                           pad_right,
                           value);
      CHECK_EQ(r, 0);
    } else {
      std::vector<int> pad_left = {0, pads[4], pads[2], pads[0], 0};
      std::vector<int> pad_right = {0, pads[5], pads[3], pads[1], 0};

      int n_shape = in_dims[0];
      int d_shape = in_dims[1];
      int h_shape = in_dims[2];
      int w_shape = in_dims[3];
      int c_shape = in_dims[4];
      std::vector<int> xshape = {n_shape, d_shape, h_shape, w_shape, c_shape};

      int r = xdnn::pad<T>(ctx.GetRawContext(),
                           in_data,
                           out_data,
                           xshape,
                           pad_left,
                           pad_right,
                           value);
      CHECK_EQ(r, 0);
    }
  } else if (mode == "reflect") {
    int r = 0;
    if (is_ncdhw) {
      int r = xdnn::reflection_pad3d<T>(ctx.GetRawContext(),
                                        in_data,
                                        out_data,
                                        in_dims[0],
                                        in_dims[1],
                                        in_dims[2],
                                        in_dims[3],
                                        in_dims[4],
                                        padding,
                                        is_ncdhw);
    } else {
      int r = xdnn::reflection_pad3d<T>(ctx.GetRawContext(),
                                        in_data,
                                        out_data,
                                        in_dims[0],
                                        in_dims[4],
                                        in_dims[1],
                                        in_dims[2],
                                        in_dims[3],
                                        padding,
                                        is_ncdhw);
    }
    CHECK_EQ(r, 0);
  } else if (mode == "replicate") {
    int r = 0;
    if (is_ncdhw) {
      int r = xdnn::replication_pad3d<T>(ctx.GetRawContext(),
                                         in_data,
                                         out_data,
                                         in_dims[0],
                                         in_dims[1],
                                         in_dims[2],
                                         in_dims[3],
                                         in_dims[4],
                                         padding,
                                         is_ncdhw);
    } else {
      int r = xdnn::replication_pad3d<T>(ctx.GetRawContext(),
                                         in_data,
                                         out_data,
                                         in_dims[0],
                                         in_dims[4],
                                         in_dims[1],
                                         in_dims[2],
                                         in_dims[3],
                                         padding,
                                         is_ncdhw);
    }
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
