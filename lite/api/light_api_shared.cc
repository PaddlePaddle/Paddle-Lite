/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#ifndef LITE_ON_TINY_PUBLISH
#include "lite/api/paddle_use_passes.h"
#endif

namespace paddle {
namespace lite_api {

void RunModel() {
  // 1. Set MobileConfig
  MobileConfig mobile_config;

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> mobile_predictor =
      CreatePaddlePredictor<MobileConfig>(mobile_config);
}

}  // namespace lite_api
}  // namespace paddle
