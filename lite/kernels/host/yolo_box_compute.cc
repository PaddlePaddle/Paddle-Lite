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

#include "lite/kernels/host/yolo_box_compute.h"
#include <vector>

#ifdef ENABLE_ARM_FP16
using fp16_yolo = paddle::lite::kernels::host::YoloBoxCompute<float16_t,
                                                              TARGET(kARM),
                                                              PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(yolo_box, kARM, kFP16, kNCHW, fp16_yolo, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Boxes",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Scores",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

using fp32_yolo =
    paddle::lite::kernels::host::YoloBoxCompute<float,
                                                TARGET(kHost),
                                                PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(yolo_box, kHost, kFloat, kNCHW, fp32_yolo, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Scores", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
