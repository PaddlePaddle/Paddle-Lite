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
//
// ncnn license
// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#pragma once

#include <math.h>
#include <stdint.h>
#include "lite/utils/cv/paddle_image_preprocess.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
class ImageResize {
 public:
  void choose(const uint8_t* src,
              uint8_t* dst,
              ImageFormat srcFormat,
              int srcw,
              int srch,
              int dstw,
              int dsth);
};
void resize(const uint8_t* src,
            uint8_t* dst,
            ImageFormat srcFormat,
            int srcw,
            int srch,
            int dstw,
            int dsth);

}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
