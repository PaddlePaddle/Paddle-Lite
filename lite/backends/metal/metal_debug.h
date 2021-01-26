// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_BACKENDS_METAL_METAL_DEBUG_H_
#define LITE_BACKENDS_METAL_METAL_DEBUG_H_

#include <string>
#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_image.h"

namespace paddle {
namespace lite {

class MetalDebug {
 public:
  enum class DumpMode {
    kFile,
    kStd,
    kBoth,
  };

  static void DumpImage(std::string name,
                        MetalImage* image,
                        int length,
                        DumpMode mode = DumpMode::kBoth);
  static void DumpImage(std::string name,
                        const MetalImage* image,
                        int length,
                        DumpMode mode = DumpMode::kBoth);
  static void DumpBuffer(std::string name,
                         MetalBuffer* image,
                         int length,
                         DumpMode mode = DumpMode::kBoth);
  static void DumpBuffer(std::string name,
                         const MetalBuffer* image,
                         int length,
                         DumpMode mode = DumpMode::kBoth);
  void DumpNCHWFloat(std::string name,
                     float* data,
                     int length,
                     DumpMode mode = DumpMode::kBoth);
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_DEBUG_H_
