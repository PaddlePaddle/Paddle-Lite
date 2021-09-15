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

#include <map>
#include <memory>
#include <string>

#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_image.h"
#include "lite/utils/macros.h"

#define LITE_METAL_SAVE_TENSOR 0

namespace paddle {
namespace lite {

class MetalDebug {
   public:
    enum class DumpMode {
        kFile,
        kStd,
        kBoth,
    };

    static bool enable() {
        return enable_;
    }
    static void set_enable(bool flag) {
        enable_ = flag;
    }

    static void SaveOutput(std::string name,
        const MetalImage* image,
        DumpMode mode = DumpMode::kBoth) {
        print_log(name, image);
    }

    static void SaveOutput(std::string name, MetalBuffer* buffer, DumpMode mode = DumpMode::kBoth) {
        print_log(name, buffer);
    }

    static void SaveOutput_(std::string name, MetalImage* image, DumpMode mode = DumpMode::kBoth);

    static void print_log(const std::string& name, const MetalImage* metalImg, int inCount = 80);

    static void print_log(const std::string& name, MetalBuffer* metalImg, int inCount = 80);

    static void print_float(const std::string& name, float* data, int size, int inCount = 80);

    static void DumpImage(const std::string& name,
        MetalImage* image,
        DumpMode mode = DumpMode::kBoth);

    static void DumpBuffer(const std::string& name,
        MetalBuffer* image,
        DumpMode mode = DumpMode::kBoth);

   private:
    static LITE_THREAD_LOCAL std::map<std::string, int> op_stats_;
    static LITE_THREAD_LOCAL bool enable_;
    static LITE_THREAD_LOCAL int layer_count_;
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_DEBUG_H_
