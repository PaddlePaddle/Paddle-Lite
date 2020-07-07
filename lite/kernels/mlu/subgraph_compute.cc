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

#include "lite/kernels/mlu/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/mlu/bridges/paddle_use_bridges.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    subgraph,
    kMLU,
    kFloat,
    kNHWC,
    paddle::lite::kernels::mlu::SubgraphCompute<PRECISION(kFloat)>,
    def_kFloat)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kMLU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kMLU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    subgraph,
    kMLU,
    kFP16,
    kNHWC,
    paddle::lite::kernels::mlu::SubgraphCompute<PRECISION(kFP16)>,
    def_FP16)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kMLU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kMLU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
