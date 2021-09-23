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

#ifndef LITE_BACKENDS_METAL_METAL_CONTEXT_H_
#define LITE_BACKENDS_METAL_METAL_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace lite {

class MetalContext {
   public:
    MetalContext();
    ~MetalContext();

    // external
    void wait_all_completed();

    // config
    void set_metal_device(void* device);
    void set_metal_path(std::string path);
    void set_use_mps(bool flag) {
        use_mps_ = flag;
    }
    void set_use_aggressive(bool flag) {
        use_aggressive_ = flag;
    }
    void set_use_memory_reuse(bool flag);

    bool use_mps() const {
        return use_mps_;
    }
    bool use_quadruple() const {
        return use_aggressive_;
    }
    bool use_winograde() const {
        return use_aggressive_;
    }
    bool use_memory_reuse() const {
        return use_memory_reuse_;
    }

    // ptr
    void* backend() const {
        return mContext;
    }

   private:
    bool use_mps_{false};
    bool use_aggressive_{false};
    bool use_memory_reuse_{false};
    void* mContext = nullptr;
};
}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_CONTEXT_H_
