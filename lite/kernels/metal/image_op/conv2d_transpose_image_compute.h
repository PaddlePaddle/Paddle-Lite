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
#include "lite/backends/metal/mps_conv_datasource.h"
#include "lite/kernels/metal/image_op/metal_params.h"

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
        MetalDebug::SaveOutput(
            (use_mps_ ? "MPS_conv2d_transpose" : function_name_), output_buffer_);
    };
    virtual ~Conv2dTransposeImageCompute();

   private:
    bool use_mps_{false};
    void* mps_conv_trans_op_{nullptr};
    void* mps_conv_op_{nullptr};
    void* mps_input_image_{nullptr};
    void* mps_output_image_{nullptr};

    void init_for_run();
    void init_attention();
    void init_memory();
    void release_memory();
    void release_mps_memory();
    void release_intermediate();

    void setup_with_mps();
    void setup_without_mps();

    void run_with_mps();
    void run_without_mps();

    void run_2x2();
    void run_3x3();
    void run_4x4();

    bool canAddUseMPS();
    bool canMPSAddByChannel();
    bool canMPSAddByElement();

    std::string KernelFunctionName(const param_t& param);
    static bool HasPrefix(const std::string& function_name, const std::string& prefix_name);
    static bool HasSuffix(const std::string& function_name, const std::string& suffix);

   private:
    int16_t activate_type_ = 0;
    int16_t relu6_thredhold_ = 6;

    id<MTLComputePipelineState> pipline_;
    std::string function_name_;
    MetalContext* metal_context_;
    DDim last_input_dims_{};

    MetalImage* output_buffer_{nullptr};
    const MetalImage* input_buffer_;
    const MetalImage* bias_buffer_;
    MetalImage* blank_buffer_{nullptr};
    std::shared_ptr<MetalBuffer> filter_buffer_;
    std::shared_ptr<MetalBuffer> params_buffer_;

    DDim filter_metal_dims_{};
    MetalImage* intermediate_shift_left_{nullptr};
    MetalImage* intermediate_shift_right_{nullptr};
    MetalImage* intermediate_bias_relu_output_{nullptr};
    id<MTLComputePipelineState> pipline_shift_left_;
    id<MTLComputePipelineState> pipline_shift_right_;
    id<MTLComputePipelineState> pipline_bias_relu_output_;
};

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
