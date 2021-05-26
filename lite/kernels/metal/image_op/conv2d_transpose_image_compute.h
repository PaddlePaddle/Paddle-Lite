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

template <typename P, PrecisionType PTYPE>
class Conv2dTransposeImageCompute
    : public KernelLite<TARGET(kMetal), PTYPE, DATALAYOUT(kMetalTexture2DArray)> {
    using param_t = operators::ConvParam;

   public:
    void PrepareForRun() override;
    void Run() override;
    void SaveOutput() override {
        MetalDebug::SaveOutput("conv2d_transpose", output_buffer_);
    };

   private:
    const MetalImage* input_buffer_;
    std::shared_ptr<MetalBuffer> param_buffer_;

    static std::string KernelFunctionName(const param_t& param,
                                          bool use_aggressive_optimization = false);

    static bool HasPrefix(const std::string& function_name, const std::string& prefix_name);

   private:
    void SetupWithMPS();
    void SetupWithoutMPS();

    MetalImage* output_buffer_;
    std::shared_ptr<MetalBuffer> filter_buffer_;
    std::shared_ptr<MetalBuffer> params_buffer_;
    const MetalImage* bias_buffer_;

    Tensor blank_tensor_;
    std::string function_name_;

    int16_t activate_type_ = 0;
    int16_t relu6_thredhold_ = 6;

    std::shared_ptr<MetalKernel> kernel_;
    std::shared_ptr<MetalQueue> queue_;
    std::shared_ptr<MetalEncoder> encoder_;
    MetalContext* metal_context_;
};

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
