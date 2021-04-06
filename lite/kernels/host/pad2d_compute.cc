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

#include "lite/kernels/host/pad2d_compute.h"
#include <algorithm>
#include <string>
#include <vector>
#include "lite/backends/host/math/pad2d.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void Pad2dCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
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
  auto out_dims = out->dims();
  T* out_data = out->template mutable_data<T>();

  const int pad_top = pads[0];
  const int pad_left = pads[2];
  const int num = in_dims[0];
  if (data_format == "NCHW") {
    const int channels = in_dims[1];
    const int in_height = in_dims[2];
    const int in_width = in_dims[3];
    const int out_height = out_dims[2];
    const int out_width = out_dims[3];
    if (mode == "reflect") {
      lite::host::math::Pad2DReflectNCHW(in_data,
                                         num,
                                         channels,
                                         in_height,
                                         in_width,
                                         out_height,
                                         out_width,
                                         pad_top,
                                         pad_left,
                                         out_data);
    } else if (mode == "edge") {
      lite::host::math::Pad2DEdgeNCHW(in_data,
                                      num,
                                      channels,
                                      in_height,
                                      in_width,
                                      out_height,
                                      out_width,
                                      pad_top,
                                      pad_left,
                                      out_data);
    } else {
      lite::host::math::Pad2DConstNCHW(in_data,
                                       num,
                                       channels,
                                       in_height,
                                       in_width,
                                       out_height,
                                       out_width,
                                       pad_top,
                                       pad_left,
                                       value,
                                       out_data);
    }
  } else {
    const int channels = in_dims[3];
    const int in_height = in_dims[1];
    const int in_width = in_dims[2];
    const int out_height = out_dims[1];
    const int out_width = out_dims[2];
    if (mode == "reflect") {
      lite::host::math::Pad2DReflectNHWC(in_data,
                                         num,
                                         channels,
                                         in_height,
                                         in_width,
                                         out_height,
                                         out_width,
                                         pad_top,
                                         pad_left,
                                         out_data);
    } else if (mode == "edge") {
      lite::host::math::Pad2DEdgeNHWC(in_data,
                                      num,
                                      channels,
                                      in_height,
                                      in_width,
                                      out_height,
                                      out_width,
                                      pad_top,
                                      pad_left,
                                      out_data);
    } else {
      lite::host::math::Pad2DConstNHWC(in_data,
                                       num,
                                       channels,
                                       in_height,
                                       in_width,
                                       out_height,
                                       out_width,
                                       pad_top,
                                       pad_left,
                                       value,
                                       out_data);
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pad2d,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::Pad2dCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
