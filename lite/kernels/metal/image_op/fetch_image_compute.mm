// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

    const auto& param = this->Param<param_t>();
#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.input->data<MetalHalf, MetalImage>();
#endif

    // output
    auto* fetch_list = param.fetch_list;
    if (param.col >= fetch_list->size()) {
        fetch_list->resize(param.col + 1);
    }
    Tensor* output_tensor = &fetch_list->at(param.col);
    auto count = param.input->dims().production();
    auto size = count * sizeof(float);
    auto output_dims = DDimLite({count});
    output_tensor->Resize(output_dims);
    auto data = output_tensor->template mutable_data<float>(TARGET(kHost), size);
    TargetWrapperMetal::MemsetSync(data, 0, size);
    // output: MTLBuffer（ps：output layout is NCHW）
    output_buffer_ = make_shared<MetalBuffer>(metal_context_, output_dims, size);
    //
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    [backend add_fetch_kernel_ptr:this];
    
    setup_without_mps();
}

void FetchImageCompute::Run() {
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

void FetchImageCompute::fetch_data_from_gpu() {
    // fetch wait completed
    const auto& param = this->Param<param_t>();
    auto* fetch_list = param.fetch_list;
    Tensor* output_tensor = &fetch_list->at(param.col);
    auto data = output_tensor->data<float>();
    auto size = param.input->dims().production();
    float* buf = (float*)[output_buffer_->buffer() contents];
    TargetWrapperMetal::MemcpySync((void*)data, (void*)buf, size * sizeof(float));
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

    std::vector<int> transpose_nhwc = {0, 2, 3, 1};
    std::vector<int> transpose_nchw = {0, 1, 2, 3};
    if (input_buffer_->transpose_ == transpose_nhwc) {
        function_name_ = "fetch";
    } else if (input_buffer_->transpose_ == transpose_nchw) {
        LOG(FATAL) << "fetch: all transpose should be {0, 2, 3, 1}";
    } else {
        LOG(FATAL) << "fetch: unsupported tensor transpose";
    }

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
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .Finalize();
