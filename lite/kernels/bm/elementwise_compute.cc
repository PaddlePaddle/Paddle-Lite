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

#include "lite/kernels/bm/elementwise_compute.h"
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {

void ElementwiseAddCompute::Run() {
}

template <PrecisionType Ptype_out>
void ElementwiseAddComputeInt8<Ptype_out>::Run() {
}

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kBM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::bm::ElementwiseAddCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kBM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kBM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kBM))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
                     kBM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::bm::ElementwiseAddComputeInt8<PRECISION(kInt8)>,
                     def)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kBM))})
.BindInput("Y", {LiteType::GetTensorTy(TARGET(kBM))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kBM))})
.Finalize();
