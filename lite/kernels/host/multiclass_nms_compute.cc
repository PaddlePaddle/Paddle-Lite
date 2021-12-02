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

#include "lite/kernels/host/multiclass_nms_compute.h"

namespace paddle {
namespace lite {
namespace kernels {}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using nmsfp32 =
    paddle::lite::kernels::host::MulticlassNmsCompute<float,
                                                      TARGET(kHost),
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(multiclass_nms, kHost, kFloat, kNCHW, nmsfp32, def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(multiclass_nms2, kHost, kFloat, kNCHW, nmsfp32, def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(multiclass_nms3, kHost, kFloat, kNCHW, nmsfp32, def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("RoisNum",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("NmsRoisNum",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
