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

#include <vector>
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
enum ImageFormat {
  RGBA = 0,
  BGRA,
  RGB,
  BGR,
  GRAY,
  NV21 = 11,
  NV12,
};
enum FlipParam { X = 0, Y, XY };
enum LayOut { CHW = 0, HWC };
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
