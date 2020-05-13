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

#include "lite/core/context.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_XPU
std::string Context<TargetType::kXPU>::_multi_encoder_precision;  // NOLINT
thread_local xdnn::Context* Context<TargetType::kXPU>::_tls_raw_ctx{nullptr};
int Context<TargetType::kXPU>::_workspace_l3_size_per_thread{0};
#endif

}  // namespace lite
}  // namespace paddle
