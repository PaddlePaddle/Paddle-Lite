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

#include "lite/kernels/x86/elementwise_compute.h"

REGISTER_LITE_KERNEL(elementwise_add,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseAddCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseAddCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseAddCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseSubCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseSubCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseSubCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMulCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMulCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMulCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseDivCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseDivCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseDivCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_floordiv,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseFloorDivCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_floordiv,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseFloorDivCompute<int>,
    int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_floordiv,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseFloorDivCompute<int64_t>,
    int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_pow,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwisePowCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mod,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseModCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mod,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseModCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMaxCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMaxCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMaxCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_min,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMinCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_min,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMinCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_min,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMinCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();
