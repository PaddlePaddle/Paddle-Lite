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

#include "lite/kernels/x86/sequence_expand_as_compute.h"

using sequence_expand_as_float32 =
    paddle::lite::kernels::x86::SequenceExpandAsCompute<float,
                                                        PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    sequence_expand_as, kX86, kFloat, kNCHW, sequence_expand_as_float32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

using sequence_expand_as_int32 =
    paddle::lite::kernels::x86::SequenceExpandAsCompute<int32_t,
                                                        PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    sequence_expand_as, kX86, kFloat, kNCHW, sequence_expand_as_int32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

using sequence_expand_as_int64 =
    paddle::lite::kernels::x86::SequenceExpandAsCompute<int64_t,
                                                        PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    sequence_expand_as, kX86, kFloat, kNCHW, sequence_expand_as_int64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();
