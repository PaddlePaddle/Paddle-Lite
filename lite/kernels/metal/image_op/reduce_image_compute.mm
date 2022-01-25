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

#include <cmath>

#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/kernels/metal/image_op/reduce_image_compute.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace metal {

void ReduceImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.X->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    // use mps or not
    bool should_use_mps = false;
    if (@available(iOS 11.3, macOS 10.13.4, macCatalyst 13.0, *)) {
        if (metal_context_->use_mps()) {
            should_use_mps = true;
        }
    }
    
    auto irank = input_buffer_->tensor_dim_.size();
    if (irank != 4) {
        LOG(FATAL) << "reduce: only support input is 4-dim";
    }

    if (param.keep_dim == true) {
        auto orank = output_buffer_->tensor_dim_.size();
        if (orank != 4) {
            LOG(FATAL) << "reduce: error keep_dime=true but output isn't 4-dim";
        }
    } else {
        should_use_mps = false;
    }

    use_mps_ = should_use_mps;
    if (use_mps_) {
        setup_with_mps();
    } else {
        setup_without_mps();
    }
}

void ReduceImageCompute::Run() {
    @autoreleasepool {
        if (use_mps_) {
            run_with_mps();
        } else {
            run_without_mps();
        }
    }
}

#pragma mark - SELF

void ReduceImageCompute::run_without_mps() {
    if (!pipline_) {
        LOG(FATAL) << "reduce: don't support " << function_name_;
        return;
    }
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder setTexture:(output_buffer_->image()) atIndex:(1)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void ReduceImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    bool isValid = false;
    if (param.keep_dim) {
        if (param.dim.size() == 1) {
            if (param.dim[0] == 1) {
                // support
                isValid = true;
            }
        }
        else if(param.dim.size() == 2) {
            if (param.dim[0] == 2 && param.dim[1] == 3) {
                // support
                isValid = true;
            }
        }
    } else {
        if (param.dim.size() == 1) {
            
        }
        else if(param.dim.size() == 2) {
            if (param.dim[0] == 1 && param.dim[1] == 2) {
                // support
                isValid = true;
            }
        }
    }
    
    if (!isValid) {
        return;
    }

    std::string xx = "";
    if (param.dim.size() == 1) {
        xx = "c";
    } else if (param.dim.size() == 2) {
        // reduce by CH,HW
        if (param.dim[0] == 1 && param.dim[1] == 2) {
            xx = "ch";
        } else if (param.dim[0] == 2 && param.dim[1] == 3) {
            xx = "hw";
        } else {
            LOG(INFO) << "reduce: only support CH,HW";
        }
    }
    
    auto reduce_type_ = KernelBase::op_type();
    function_name_ = reduce_type_ + "_" + xx + (param.keep_dim ? "" : "unkeep");

    auto irank = input_buffer_->tensor_dim_.size();
    std::vector<int> idm = {1, 1, 1, 1};
    for (int i = 0; i < irank; i++) {
        idm[4 - irank + i] = (int)(input_buffer_->tensor_dim_[i]);
    }

    auto orank = output_buffer_->tensor_dim_.size();
    std::vector<int> odm = {1, 1, 1, 1};
    for (int i = 0; i < orank; i++) {
        odm[4 - orank + i] = (int)(output_buffer_->tensor_dim_[i]);
    }

    RankMetalParam metal_params{
        (int)irank, {idm[0], idm[1], idm[2], idm[3]}, (int)orank, {odm[0], odm[1], odm[2], odm[3]},
    };
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_params), &metal_params);

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

#pragma mark - MPS

void ReduceImageCompute::run_with_mps() {
    const auto& param = this->Param<param_t>();
    if (param.dim.size() == 1) {
        if(!mps_op_x_) {
            LOG(FATAL) << "mps_reduce: reduce x error";
        }
    }
    else if (param.dim.size() == 2) {
        if(!mps_op_x_ && !mps_op_xx_) {
            LOG(FATAL) << "mps_reduce: reduce xx error";
        }
    } else {
        LOG(FATAL) << "mps_reduce: only support reduce x or xx";
    }
    
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    if (@available(iOS 11.3, macOS 10.13.4, macCatalyst 13.0, *)) {
        
        id <MTLTexture> mid_texture = nil;
        
        auto cmdbuf = [backend commandBuffer];
        [((__bridge MPSNNReduceUnary *)mps_op_x_) encodeToCommandBuffer:cmdbuf
                                          sourceImage:(__bridge MPSImage*)mps_input_image_
                                     destinationImage:(__bridge MPSImage*)mps_output_image_x_];
        [backend commit:cmdbuf];
        
        if (param.dim.size() == 1) {
            mid_texture = [(__bridge MPSImage*)mps_output_image_x_ texture];
        }
        else if (param.dim.size() == 2) {
            //second
            auto cmdbuf = [backend commandBuffer];
            [((__bridge MPSNNReduceUnary *)mps_op_xx_) encodeToCommandBuffer:cmdbuf
                                              sourceImage:(__bridge MPSImage*)mps_output_image_x_
                                         destinationImage:(__bridge MPSImage*)mps_output_image_xx_];
            [backend commit:cmdbuf];

            mid_texture = [(__bridge MPSImage*)mps_output_image_xx_ texture];
        } else {
            
        }
        //c1 -> c4
        if (param.dim[0] == 1 && mid_texture) {
            std::string function = "tex2d_c1_to_c4";
            id<MTLComputePipelineState> pipline = [backend pipline:function];

            auto outTexture = output_buffer_->image();

            auto encoder = [backend commandEncoder];
            [encoder setTexture:(mid_texture) atIndex:(0)];
            [encoder setTexture:(output_buffer_->image()) atIndex:(1)];

            [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
            [backend commit];
        }
    }
}

void ReduceImageCompute::setup_with_mps() {
    auto reduce_type_ = KernelBase::op_type();
    const auto& param = this->Param<param_t>();
    
    if (@available(iOS 11.3, macOS 10.13.4, macCatalyst 13.0, *)) {
        if (param.dim.size() == 1) {
            mps_op_x_ = init_mps_op(param.dim[0]);
        }
        else if (param.dim.size() == 2) {
            mps_op_x_ = init_mps_op(param.dim[0]);
            mps_op_xx_ = init_mps_op(param.dim[1]);
        } else {
            
        }
    }
     
    if (@available(iOS 11.3, macOS 10.13.4, macCatalyst 13.0, *)) {
        // MPS input and output
        auto input_c = MAX(4, static_cast<int>(input_buffer_->tensor_dim_[1]));
        auto output_c = MAX(4, static_cast<int>(output_buffer_->tensor_dim_[1]));
        mps_input_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_->image()
                                                       featureChannels:input_c];
        
        if (param.dim.size() == 1) {
            if (param.dim[0] == 1) {
                auto odims = param.Out->dims();
                auto metal_image = new MetalImage(metal_context_, odims);
                metal_image->use_mps_ = true;
                metal_image->initImage(metal_context_);
                mps_output_image_x_ =
                    (__bridge_retained void*)[[MPSImage alloc] initWithTexture:metal_image->image()
                                                               featureChannels:odims[1]];
                free(metal_image);
            } else {
                mps_output_image_x_ =
                    (__bridge_retained void*)[[MPSImage alloc] initWithTexture:output_buffer_->image()
                                                               featureChannels:output_c];
            }
        }
        else if (param.dim.size() == 2) {
            // first
            {
                auto odims = param.X->dims();
                odims[param.dim[0]] = 1;
                auto metal_image = new MetalImage(metal_context_, odims);
                if (param.dim[0] == 1) {
                    metal_image->use_mps_ = true;
                }
                metal_image->initImage(metal_context_);
                mps_output_image_x_ =
                    (__bridge_retained void*)[[MPSImage alloc] initWithTexture:metal_image->image()
                                                               featureChannels:odims[1]];
                free(metal_image);
            }
            //second
            {
                if (param.dim[0] == 1) {
                    auto odims = param.Out->dims();
                    auto metal_image = new MetalImage(metal_context_, odims);
                    metal_image->initImage(metal_context_);
                    mps_output_image_xx_ =
                        (__bridge_retained void*)[[MPSImage alloc] initWithTexture:metal_image->image()
                                                                   featureChannels:odims[1]];
                    free(metal_image);
                } else {
                    mps_output_image_xx_ =
                        (__bridge_retained void*)[[MPSImage alloc] initWithTexture:output_buffer_->image()
                                                                   featureChannels:output_c];
                }
            }
        } else {
            
        }
    }
}

/**
 * The reduction operations supported are:
 *                   - Reduce row min
 *                   - Reduce column min
 *                   - Reduce feature channels min
 *                   - Reduce row max
 *                   - Reduce column max
 *                   - Reduce feature channels max
 *                   - Reduce row mean
 *                   - Reduce column mean
 *                   - Reduce feature channels mean
 *                   - Reduce row sum
 *                   - Reduce column sum
 *                   - Reduce feature channels sum
 */
// NCHW = 0123
void* ReduceImageCompute::init_mps_op(int index) {
    void* mps_op_{nullptr};
    auto reduce_type_ = KernelBase::op_type();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    if (@available(iOS 11.3, macOS 10.13.4, macCatalyst 13.0, *)) {
        if (index == 1) {
            if (reduce_type_ == ("reduce_max")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceFeatureChannelsMax alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_min")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceFeatureChannelsMin alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_mean")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceFeatureChannelsMean alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_sum")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceFeatureChannelsSum alloc] initWithDevice:backend.device];
            }
        }
        else if (index == 2) {
            if (reduce_type_ == ("reduce_max")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceColumnMax alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_min")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceColumnMin alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_mean")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceColumnMean alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_sum")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceColumnSum alloc] initWithDevice:backend.device];
            }
        }
        else if (index == 3) {
            if (reduce_type_ == ("reduce_max")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceRowMax alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_min")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceRowMin alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_mean")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceRowMean alloc] initWithDevice:backend.device];
            } else if (reduce_type_ == ("reduce_sum")) {
                mps_op_ = (__bridge_retained void*)[[MPSNNReduceRowSum alloc] initWithDevice:backend.device];
            }
        }
    }
    
    return mps_op_;
}

ReduceImageCompute::~ReduceImageCompute() {
    if (mps_op_x_) {
        CFRelease(mps_op_x_);
        mps_op_x_ = nullptr;
    }
    if (mps_op_xx_) {
        CFRelease(mps_op_xx_);
        mps_op_xx_ = nullptr;
    }
    if (mps_input_image_) {
        CFRelease(mps_input_image_);
        mps_input_image_ = nullptr;
    }
    if (mps_output_image_x_) {
        CFRelease(mps_output_image_x_);
        mps_output_image_x_ = nullptr;
    }
    if (mps_output_image_xx_) {
        CFRelease(mps_output_image_xx_);
        mps_output_image_xx_ = nullptr;
    }
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#pragma mark -

REGISTER_LITE_KERNEL(max,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite_metal::kernels::metal::ReduceImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_max,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite_metal::kernels::metal::ReduceImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_min,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite_metal::kernels::metal::ReduceImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_sum,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite_metal::kernels::metal::ReduceImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_mean,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite_metal::kernels::metal::ReduceImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
