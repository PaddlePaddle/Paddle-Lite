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

#include "lite/kernels/xpu/yolo_box_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void YoloBoxCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto input_dims = param.X->dims();
  std::vector<int> anchors = param.anchors;
  CHECK_LE(anchors.size(), 6UL);
  const int n = input_dims[0];
  const int h = input_dims[2];
  const int w = input_dims[3];
  const int box_num = param.Boxes->dims()[1];
  const int an_num = anchors.size() / 2;
  int downsample_ratio = param.downsample_ratio;
  int class_num = param.class_num;
  float scale_x_y = param.scale_x_y;
  float bias = -0.5 * (scale_x_y - 1.);
  CHECK_EQ(box_num, an_num * h * w);

  int r = xdnn::yolo_box<float>(ctx.GetRawContext(),
                                param.X->data<float>(),
                                param.ImgSize->data<int>(),
                                param.Boxes->mutable_data<float>(TARGET(kXPU)),
                                param.Scores->mutable_data<float>(TARGET(kXPU)),
                                n,
                                h,
                                w,
                                anchors,
                                an_num,
                                class_num,
                                param.conf_thresh,
                                downsample_ratio,
                                scale_x_y,
                                bias,
                                false);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(yolo_box,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::YoloBoxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Scores", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
