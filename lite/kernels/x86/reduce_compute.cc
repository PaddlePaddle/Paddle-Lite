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

#include "lite/kernels/x86/reduce_compute.h"

namespace x86 = paddle::lite::kernels::x86;

using ReduceMeanFloat32 = x86::ReduceCompute<float, x86::MeanFunctor>;
REGISTER_LITE_KERNEL(reduce_mean, kX86, kFloat, kNCHW, ReduceMeanFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

#ifdef LITE_BUILD_EXTRA
using ReduceSumFloat32 = x86::ReduceCompute<float, x86::SumFunctor>;
REGISTER_LITE_KERNEL(reduce_sum, kX86, kFloat, kNCHW, ReduceSumFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

using ReduceSumInt32 = x86::ReduceCompute<int, x86::SumFunctor>;
REGISTER_LITE_KERNEL(reduce_sum, kX86, kFloat, kNCHW, ReduceSumInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

using ReduceSumInt64 = x86::ReduceCompute<int64_t, x86::SumFunctor>;
REGISTER_LITE_KERNEL(reduce_sum, kX86, kFloat, kNCHW, ReduceSumInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

using ReduceProdFloat32 = x86::ReduceCompute<float, x86::ProdFunctor>;
REGISTER_LITE_KERNEL(reduce_prod, kX86, kFloat, kNCHW, ReduceProdFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

using ReduceProdInt32 = x86::ReduceCompute<int, x86::ProdFunctor>;
REGISTER_LITE_KERNEL(reduce_prod, kX86, kFloat, kNCHW, ReduceProdInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

using ReduceProdInt64 = x86::ReduceCompute<int64_t, x86::ProdFunctor>;
REGISTER_LITE_KERNEL(reduce_prod, kX86, kFloat, kNCHW, ReduceProdInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

using ReduceMaxFloat32 = x86::ReduceCompute<float, x86::MaxFunctor>;
REGISTER_LITE_KERNEL(reduce_max, kX86, kFloat, kNCHW, ReduceMaxFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

using ReduceMaxInt32 = x86::ReduceCompute<int, x86::MaxFunctor>;
REGISTER_LITE_KERNEL(reduce_max, kX86, kFloat, kNCHW, ReduceMaxInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

using ReduceMaxInt64 = x86::ReduceCompute<int64_t, x86::MaxFunctor>;
REGISTER_LITE_KERNEL(reduce_max, kX86, kFloat, kNCHW, ReduceMaxInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

using ReduceMinFloat32 = x86::ReduceCompute<float, x86::MinFunctor>;
REGISTER_LITE_KERNEL(reduce_min, kX86, kFloat, kNCHW, ReduceMinFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

using ReduceMinInt32 = x86::ReduceCompute<int, x86::MinFunctor>;
REGISTER_LITE_KERNEL(reduce_min, kX86, kFloat, kNCHW, ReduceMinInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

using ReduceMinInt64 = x86::ReduceCompute<int64_t, x86::MinFunctor>;
REGISTER_LITE_KERNEL(reduce_min, kX86, kFloat, kNCHW, ReduceMinInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA
