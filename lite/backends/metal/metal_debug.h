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

class metal_debug {
 public:
  enum class dump_mode {
    TO_FILE,
    TO_STDOUT,
    TO_BOTH,
  };

  static void dump_image(std::string name,
                         metal_image* image,
                         int length,
                         dump_mode mode = dump_mode::TO_BOTH);
  static void dump_image(std::string name,
                         const metal_image* image,
                         int length,
                         dump_mode mode = dump_mode::TO_BOTH);
  static void dump_buffer(std::string name,
                          metal_buffer* image,
                          int length,
                          dump_mode mode = dump_mode::TO_BOTH);
  static void dump_buffer(std::string name,
                          const metal_buffer* image,
                          int length,
                          dump_mode mode = dump_mode::TO_BOTH);
  void dump_nchw_float(std::string name,
                       float* data,
                       int length,
                       dump_mode mode = dump_mode::TO_BOTH);
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_DEBUG_H_
