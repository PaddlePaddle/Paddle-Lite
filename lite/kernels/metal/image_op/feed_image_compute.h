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

#ifndef LITE_KERNELS_METAL_IMAGE_OP_FEED_IMAGE_COMPUTE_H_
#define LITE_KERNELS_METAL_IMAGE_OP_FEED_IMAGE_COMPUTE_H_

#include <memory>
#include <string>

#include "lite/core/kernel.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

#include "lite/backends/metal/metal_context.h"
#include "lite/backends/metal/metal_debug.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

class FeedImageCompute : public KernelLite<TARGET(kMetal), PRECISION(kAny), DATALAYOUT(kAny)> {
    using param_t = operators::FeedParam;

   public:
    void PrepareForRun() override;
    void ReInitWhenNeeded() override;
    void Run() override;
    void SaveOutput() override {
        MetalDebug::SaveOutput(function_name_, output_buffer_);
    };
    virtual ~FeedImageCompute();

   private:
    void init_memory();
    void release_memory();
    void setup_pipeline();

    void run_raw();
    void run_int32();
    void run_float();
    void setup_float();

    void run_mtl_texture();
    void setup_mtl_texture();

   private:
    std::shared_ptr<MetalBuffer> input_buffer_;
    std::shared_ptr<MetalBuffer> params_buffer_{nullptr};
    MetalImage* output_buffer_{nullptr};

    id<MTLComputePipelineState> pipline_;
    std::string function_name_;
    MetalContext* metal_context_;
    DDim last_input_dims_{};

    void* lanczos_{nullptr};
    id<MTLTexture> resize_texture_;
};

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#endif  // LITE_KERNELS_METAL_IMAGE_OP_FEED_IMAGE_COMPUTE_H_
