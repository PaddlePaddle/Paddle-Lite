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
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_HUAWEI_ASCEND_NPU
LITE_THREAD_LOCAL std::string
    Context<TargetType::kHuaweiAscendNPU>::subgraph_model_cache_dir_{
        ""};  // NOLINT
LITE_THREAD_LOCAL int
    Context<TargetType::kHuaweiAscendNPU>::huawei_ascend_device_id_{
        0};  // NOLINT
#endif

#ifdef LITE_WITH_MLU
int Context<TargetType::kMLU>::next_queue_id_{0};
std::map<int, int> Context<TargetType::kMLU>::queue_id_map_;
std::mutex Context<TargetType::kMLU>::map_mutex_;
#endif

}  // namespace lite
}  // namespace paddle
