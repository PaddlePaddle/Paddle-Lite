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

// Parts of the following code in this file refs to
// https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp
// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "lite/backends/bm/device_info.h"
#include <algorithm>
#include <limits>

namespace paddle {
namespace lite {

#ifdef LITE_WITH_BM
void Device<TARGET(kBM)>::SetId(int device_id) {
  LOG(INFO) << "Set bm device " << device_id;
  TargetWrapper<TARGET(kBM)>::SetDevice(device_id);
  idx_ = device_id;
}

void Device<TARGET(kBM)>::Init() { SetId(idx_); }
int Device<TARGET(kBM)>::core_num() {
  return TargetWrapper<TARGET(kBM)>::num_devices();
}
#endif  // LITE_WITH_BM

}  // namespace lite
}  // namespace paddle
