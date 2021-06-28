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

#include "lite/kernels/metal/image_op/slice_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void SliceImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.X->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    setup_without_mps();
}

void SliceImageCompute::Run() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:input_buffer_->image() atIndex:(0)];
    [encoder setTexture:output_buffer_->image() atIndex:(1)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void SliceImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    std::vector<int> axes = {};
    for (int i = 0; i < input_buffer_->tensor_dim_.size(); i++) {
        if (input_buffer_->tensor_dim_[i] != output_buffer_->tensor_dim_[i]) {
            axes.push_back(i);
        }
    }
    // only support C channel slice
    if (axes.size() == 1 && axes[0] == 1) {
    } else {
        LOG(FATAL) << "slice: only support channel axe";
    }
    auto starts = param.starts;
    auto ends = param.ends;
    std::map<int, std::vector<uint16_t>> ranges = {};
    for (int j = 0; j < axes.size(); j++) {
        ranges[uint16_t(axes[j])] = {uint16_t(starts[j]), uint16_t(ends[j])};
    }
    //
    int iC = (int)input_buffer_->tensor_dim_[1];
    int oC = (int)output_buffer_->tensor_dim_[1];
    uint16_t param_rangs[4][2] = {};
    for (int k = 0; k < 4; k++) {
        if (ranges.find(k) != ranges.end()) {
            param_rangs[k][0] = (ranges[k])[0];
            param_rangs[k][1] = (ranges[k])[1];
        } else {
            param_rangs[k][0] = 0;
            param_rangs[k][1] = (uint16_t)(input_buffer_->tensor_dim_[k]);
        }
    }
    SliceMetalParam params{param_rangs[0][0],
        param_rangs[1][0],
        param_rangs[2][0],
        param_rangs[3][0],
        param_rangs[0][1],
        param_rangs[1][1],
        param_rangs[2][1],
        param_rangs[3][1],
        iC,
        oC};
    params_buffer_ = std::make_shared<MetalBuffer>(metal_context_, sizeof(params), &params);

    function_name_ = "slice";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

SliceImageCompute::~SliceImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(slice,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::SliceImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(slice,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::SliceImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
