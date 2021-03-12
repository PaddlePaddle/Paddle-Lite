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

#include "lite/kernels/xpu/polygon_box_transform_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void PolygonBoxTransformCompute<T>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto x = param.input;
  auto in_dims = x->dims();
  auto out = param.output;

  int batch_size = in_dims[0];
  int geo_channel = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  int r = xdnn::polygon_box_transform<T>(
      ctx.GetRawContext(),
      x->template data<T>(),
      out->template mutable_data<T>(TARGET(kXPU)),
      batch_size,
      geo_channel,
      height,
      width);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    polygon_box_transform,
    kXPU,
    kFloat,
    kNCHW,
    paddle::lite::kernels::xpu::PolygonBoxTransformCompute<float>,
    def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
