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

#include "lite/kernels/nnadapter/feed_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

void FeedCompute::Run() {
  LOG(FATAL) << "Feed op is only used for graph optimization and should not be "
                "called!";
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(feed,
                     kNNAdapter,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::nnadapter::FeedCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kAny), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kAny), PRECISION(kAny))})
    .Finalize();
