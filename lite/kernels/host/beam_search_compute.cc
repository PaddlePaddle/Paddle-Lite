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

#include "lite/kernels/host/beam_search_compute.h"
#include "lite/backends/host/math/beam_search.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void BeamSearchCompute::Run() {
  auto& param = this->Param<operators::BeamSearchParam>();
  lite::host::math::beam_search(param.pre_ids,
                                param.pre_scores,
                                param.ids,
                                param.scores,
                                param.selected_ids,
                                param.selected_scores,
                                param.parent_idx,
                                param.level,
                                param.beam_size,
                                param.end_id,
                                param.is_accumulated);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(beam_search,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::BeamSearchCompute,
                     def)
    .BindInput("pre_ids",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("pre_scores",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("ids", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("scores",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("selected_ids",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("selected_scores",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("parent_idx",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
