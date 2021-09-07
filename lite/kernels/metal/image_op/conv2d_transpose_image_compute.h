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

class Conv2dTransposeImageCompute
    : public KernelLite<TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray)> {
    using param_t = operators::ConvParam;

   public:
    void PrepareForRun() override;
    void ReInitWhenNeeded() override;
    void Run() override;
    void SaveOutput() override {
        MetalDebug::SaveOutput("conv2d_transpose", output_buffer_);
    };

   private:
    bool use_mps_{false};

    void init_for_run();
    void init_memory();
    void release_memory();

    void setup_with_mps();
    void setup_without_mps();

    void run_with_mps();
    void run_without_mps();

    MetalImage* output_buffer_{nullptr};
    MetalImage* blank_buffer_{nullptr};
    const MetalImage* bias_buffer_;
    const MetalImage* input_buffer_;

    std::shared_ptr<MetalBuffer> param_buffer_;
    std::shared_ptr<MetalBuffer> params_buffer_;
    std::shared_ptr<MetalBuffer> filter_buffer_;

    std::string function_name_;
    int16_t activate_type_ = 0;
    int16_t relu6_thredhold_ = 6;

    MetalContext* metal_context_;
    id<MTLComputePipelineState> pipline_;
    DDim last_input_dims_{};
    static std::string KernelFunctionName(const param_t& param);
    static bool HasPrefix(const std::string& function_name, const std::string& prefix_name);
};

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
