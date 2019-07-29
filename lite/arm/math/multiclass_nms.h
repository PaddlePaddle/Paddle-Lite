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

#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename dtype>
void multiclass_nms(const dtype* bbox_cpu_data,
                    const dtype* conf_cpu_data,
                    std::vector<dtype>* result,
                    const std::vector<int>& priors,
                    int class_num,
                    int background_id,
                    int keep_topk,
                    int nms_topk,
                    float conf_thresh,
                    float nms_thresh,
                    float nms_eta,
                    bool share_location);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
