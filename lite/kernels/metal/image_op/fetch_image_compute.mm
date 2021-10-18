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

#include "lite/kernels/metal/image_op/fetch_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void FetchImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    init_memory();
    setup_without_mps();

    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    [backend add_fetch_kernel_ptr:this];
}

void FetchImageCompute::ReInitWhenNeeded() {
    const auto& param = this->Param<param_t>();
    auto input_dims = param.input->dims();

    if (last_input_dims_ != input_dims) {
        init_memory();
        setup_without_mps();
    }
}

void FetchImageCompute::init_memory() {
    const auto& param = this->Param<param_t>();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.input->data<MetalHalf, MetalImage>();
#endif

    // output GPU data
    auto input_tensor = param.input;
    auto dims = input_tensor->dims();
    auto count = (size_t)dims.production();
    output_buffer_ = make_shared<MetalBuffer>(metal_context_, dims, count * sizeof(float));

    // output CPU data: float, int64
    auto* fetch_list = param.fetch_list;
    if (param.col >= fetch_list->size()) {
        fetch_list->resize(param.col + 1);
    }
    Tensor* output_tensor = &fetch_list->at(param.col);
    output_tensor->Resize(dims);
    if (input_tensor->precision() == paddle::lite_api::PrecisionType::kFloat) {
        auto size = count * sizeof(float);
        auto data = output_tensor->template mutable_data<float>(TARGET(kHost), size);
        TargetWrapperMetal::MemsetSync(data, 0, size);
    } else if (input_tensor->precision() == paddle::lite_api::PrecisionType::kInt64) {
        auto size = count * sizeof(int64_t);
        auto data = output_tensor->template mutable_data<int64_t>(TARGET(kHost), size);
        TargetWrapperMetal::MemsetSync(data, 0, size);
    }

    last_input_dims_ = dims;
}

void FetchImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void FetchImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto inTexture = input_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder setBuffer:(output_buffer_->buffer()) offset:(0) atIndex:(0)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(1)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:inTexture];
    [backend commit];
}

// fetch wait completed
void FetchImageCompute::fetch_data_from_gpu() {
    const auto& param = this->Param<param_t>();

    auto input_tensor = param.input;
    auto count = (size_t)input_tensor->dims().production();
    Tensor* output_tensor = &(param.fetch_list->at(param.col));
    float* buffer = (float*)[output_buffer_->buffer() contents];

    // output CPU data: float, int64
    if (input_tensor->precision() == paddle::lite_api::PrecisionType::kFloat) {
        auto data = output_tensor->data<float>();
        TargetWrapperMetal::MemcpySync((void*)data, (void*)buffer, (size_t)(count * sizeof(float)));
    } else if (input_tensor->precision() == paddle::lite_api::PrecisionType::kInt64) {
        auto data = const_cast<int64_t*>(output_tensor->data<int64_t>());
        for (int i = 0; i < count; i++) {
            data[i] = (int64_t)buffer[i];
        }
    }
}

void FetchImageCompute::setup_without_mps() {
    auto irank = input_buffer_->tensor_dim_.size();
    std::vector<int> idm = {1, 1, 1, 1};
    for (int i = 0; i < irank; i++) {
        idm[4 - irank + i] = (int)(input_buffer_->tensor_dim_[i]);
    }
    FetchMetalParam fetch_params{(int)irank, {idm[0], idm[1], idm[2], idm[3]}};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(fetch_params), &fetch_params);

    function_name_ = "fetch";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fetch,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::FetchImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kAny), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny), DATALAYOUT(kNCHW))})
    .Finalize();
