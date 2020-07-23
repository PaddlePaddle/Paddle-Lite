// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.ddNod
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

#include "lite/kernels/mlu/layout_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    layout,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::mlu::LayoutNhwcToNchwCompute<PRECISION(kFloat)>,
    def_layout_nhwc2nchw_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(
    layout,
    kX86,
    kFP16,
    kNCHW,
    paddle::lite::kernels::mlu::LayoutNhwcToNchwCompute<PRECISION(kFP16)>,
    def_layout_nhwc2nchw_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(
    layout,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::mlu::LayoutNchwToNhwcCompute<PRECISION(kFloat)>,
    def_layout_nchw2nhwc_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    layout,
    kX86,
    kFP16,
    kNCHW,
    paddle::lite::kernels::mlu::LayoutNchwToNhwcCompute<PRECISION(kFP16)>,
    def_layout_nchw2nhwc_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    layout,
    kX86,
    kInt8,
    kNCHW,
    paddle::lite::kernels::mlu::LayoutNchwToNhwcCompute<PRECISION(kInt8)>,
    def_layout_nchw2nhwc_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
