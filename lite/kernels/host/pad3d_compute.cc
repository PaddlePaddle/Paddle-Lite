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

#include "lite/kernels/host/pad3d_compute.h"
#include <string>
#include <vector>
#include "lite/backends/host/math/pad3d.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void Pad3dCompute::Run() {
  auto& param = Param<operators::Pad2dParam>();
  const lite::Tensor* inputs = param.X;
  auto* out = param.Out;

  if (param.mode == "constant") {
    mode_ = 0;
  } else if (param.mode == "reflect") {
    mode_ = 1;
  } else if (param.mode == "replicate") {
    mode_ = 2;
  } else if (param.mode == "circular") {
    mode_ = 3;
  } else {
    LOG(FATAL) << "Unknown mode type";
  }

  pad_w_ = {param.paddings[0], param.paddings[1]};
  pad_h_ = {param.paddings[2], param.paddings[3]};
  pad_depth_ = {param.paddings[4], param.paddings[5]};
  pad_value_ = param.pad_value;
  data_format_ = param.data_format;
  auto x_dims = inputs->dims();
  auto out_dims = out->dims();
  // default is NCDHW
  int batch = x_dims[0];
  int channels = x_dims[1];
  int in_depth = x_dims[2];
  int in_height = x_dims[3];
  int in_width = x_dims[4];
  int out_depth = out_dims[2];
  int out_height = out_dims[3];
  int out_width = out_dims[4];
  if (data_format_ == "NDHWC") {
    channels = x_dims[4];
    in_depth = x_dims[1];
    in_height = x_dims[2];
    in_width = x_dims[3];
    out_depth = out_dims[1];
    out_height = out_dims[2];
    out_width = out_dims[3];
  }

  if (param.mode == "reflect") {
    CHECK_GT(in_depth, param.paddings[4])
        << "The depth of Input(X)'s dimension should be "
           "greater than pad_front";
    CHECK_GT(in_depth, param.paddings[5])
        << "The depth of Input(X)'s dimension should be "
           "greater than pad_back";
    CHECK_GT(in_height, param.paddings[2])
        << "The height of Input(X)'s dimension should be "
           "greater than pad_top";
    CHECK_GT(in_height, param.paddings[3])
        << "The height of Input(X)'s dimension should be "
           "greater than pad_bottom";
    CHECK_GT(in_width, param.paddings[0])
        << "The width of Input(X)'s dimension should be "
           "greater than pad_left";
    CHECK_GT(in_width, param.paddings[1])
        << "The width of Input(X)'s dimension should be "
           "greater than pad_right";
  } else if (param.mode == "circular" || param.mode == "replicate") {
    CHECK_NE(in_depth * in_height * in_width, 0)
        << "The input tensor size can not be 0 for circular "
           "or replicate padding mode.";
  }

  if (data_format_ == "NCDHW") {
    lite::host::math::pad3d_ncdhw_func(inputs,
                                       out,
                                       batch,
                                       channels,
                                       in_depth,
                                       in_height,
                                       in_width,
                                       out_depth,
                                       out_height,
                                       out_width,
                                       mode_,
                                       pad_h_,
                                       pad_w_,
                                       pad_depth_,
                                       pad_value_);
  } else if (data_format_ == "NDHWC") {
    lite::host::math::pad3d_ndhwc_func(inputs,
                                       out,
                                       batch,
                                       channels,
                                       in_depth,
                                       in_height,
                                       in_width,
                                       out_depth,
                                       out_height,
                                       out_width,
                                       mode_,
                                       pad_h_,
                                       pad_w_,
                                       pad_depth_,
                                       pad_value_);
  } else {
    LOG(FATAL) << "This dataformat:" << data_format_ << " doesn't support!";
  }
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    pad3d, kHost, kFloat, kNCHW, paddle::lite::kernels::host::Pad3dCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
