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

#include "lite/kernels/bm/graph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {

void GraphCompute::PrepareForRun() {
}

void GraphCompute::Run() {
}

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(graph_op,
                     kBM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::bm::GraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
