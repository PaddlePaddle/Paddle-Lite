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

#include "lite/kernels/loongarch/reduce_compute.h"

namespace loongarch = paddle::lite::kernels::loongarch;

using ReduceMeanFloat32 = loongarch::ReduceCompute<float, loongarch::MeanFunctor>;
REGISTER_LITE_KERNEL(reduce_mean, kLoongArch, kFloat, kNCHW, ReduceMeanFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

#ifdef LITE_BUILD_EXTRA
using ReduceSumFloat32 = loongarch::ReduceCompute<float, loongarch::SumFunctor>;
REGISTER_LITE_KERNEL(reduce_sum, kLoongArch, kFloat, kNCHW, ReduceSumFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

using ReduceSumInt32 = loongarch::ReduceCompute<int, loongarch::SumFunctor>;
REGISTER_LITE_KERNEL(reduce_sum, kLoongArch, kFloat, kNCHW, ReduceSumInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .Finalize();

using ReduceSumInt64 = loongarch::ReduceCompute<int64_t, loongarch::SumFunctor>;
REGISTER_LITE_KERNEL(reduce_sum, kLoongArch, kFloat, kNCHW, ReduceSumInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .Finalize();

using ReduceProdFloat32 = loongarch::ReduceCompute<float, loongarch::ProdFunctor>;
REGISTER_LITE_KERNEL(reduce_prod, kLoongArch, kFloat, kNCHW, ReduceProdFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

using ReduceProdInt32 = loongarch::ReduceCompute<int, loongarch::ProdFunctor>;
REGISTER_LITE_KERNEL(reduce_prod, kLoongArch, kFloat, kNCHW, ReduceProdInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .Finalize();

using ReduceProdInt64 = loongarch::ReduceCompute<int64_t, loongarch::ProdFunctor>;
REGISTER_LITE_KERNEL(reduce_prod, kLoongArch, kFloat, kNCHW, ReduceProdInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .Finalize();

using ReduceMaxFloat32 = loongarch::ReduceCompute<float, loongarch::MaxFunctor>;
REGISTER_LITE_KERNEL(reduce_max, kLoongArch, kFloat, kNCHW, ReduceMaxFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

using ReduceMaxInt32 = loongarch::ReduceCompute<int, loongarch::MaxFunctor>;
REGISTER_LITE_KERNEL(reduce_max, kLoongArch, kFloat, kNCHW, ReduceMaxInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .Finalize();

using ReduceMaxInt64 = loongarch::ReduceCompute<int64_t, loongarch::MaxFunctor>;
REGISTER_LITE_KERNEL(reduce_max, kLoongArch, kFloat, kNCHW, ReduceMaxInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .Finalize();

using ReduceMinFloat32 = loongarch::ReduceCompute<float, loongarch::MinFunctor>;
REGISTER_LITE_KERNEL(reduce_min, kLoongArch, kFloat, kNCHW, ReduceMinFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

using ReduceMinInt32 = loongarch::ReduceCompute<int, loongarch::MinFunctor>;
REGISTER_LITE_KERNEL(reduce_min, kLoongArch, kFloat, kNCHW, ReduceMinInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .Finalize();

using ReduceMinInt64 = loongarch::ReduceCompute<int64_t, loongarch::MinFunctor>;
REGISTER_LITE_KERNEL(reduce_min, kLoongArch, kFloat, kNCHW, ReduceMinInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA
