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

#include "lite/kernels/host/layout_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <>
void ImageToNCHWCompute<PRECISION(kFloat)>::Run() {
  LOG(INFO) << "image -> nchw on host";
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::host::ImageToNCHWCompute<PRECISION(kFloat)>
    Image_fp32;

REGISTER_LITE_KERNEL(
    layout_once, kHost, kFloat, kNCHW, Image_fp32, fp32_image2nchw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
