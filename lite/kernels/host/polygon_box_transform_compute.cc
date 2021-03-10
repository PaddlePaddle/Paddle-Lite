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

#include "lite/kernels/host/polygon_box_transform_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void PolygonBoxTransformCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  const lite::Tensor* in = param.input;
  auto in_dims = in->dims();
  const T* in_data = in->template data<T>();
  lite::Tensor* out = param.output;
  T* out_data = out->template mutable_data<T>();

  int batch_size = in_dims[0];
  int geo_channel = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  int id = 0;
  for (int id_n = 0; id_n < batch_size * geo_channel; ++id_n) {
    for (int id_h = 0; id_h < height; ++id_h) {
      for (int id_w = 0; id_w < width; ++id_w) {
        id = id_n * height * width + width * id_h + id_w;
        if (id_n % 2 == 0) {
          out_data[id] = id_w * 4 - in_data[id];
        } else {
          out_data[id] = id_h * 4 - in_data[id];
        }
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    polygon_box_transform,
    kHost,
    kFloat,
    kNCHW,
    paddle::lite::kernels::host::PolygonBoxTransformCompute<float>,
    def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
