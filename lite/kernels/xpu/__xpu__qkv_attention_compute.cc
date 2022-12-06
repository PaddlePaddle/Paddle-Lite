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
#include <vector>

#include "lite/kernels/xpu/__xpu__qkv_attention_compute.h"
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename TQ, 
          typename TK, 
          typename TQK, 
          typename TGEMM, 
          typename TZ>
void XPUQkvAttentionCompute<TQ, TK, TQK, TGEMM, TZ>::PrepareForRun() {}

template <typename TQ, 
          typename TK, 
          typename TQK, 
          typename TGEMM, 
          typename TZ>
void XPUQkvAttentionCompute<TQ, TK, TQK, TGEMM, TZ>::Run() {
    auto& param = this->template Param<param_t>();
    auto& ctx = this->ctx_->template As<XPUContext>();
    int r = 0;
    std::vector<int> lod_vec = {0};
    for (int i = 1; i < param.input_k->dims()[0] + 1; i++) {
        lod_vec.push_back(lod_vec[i - 1] + param.input_k->dims()[2]);
    }
    xdnn::VectorParam<int> lods = {lod_vec.data(), static_cast<int>(lod_vec.size()), nullptr};

    std::vector<int> sqlod_vec(lod_vec.size(), 0);
    for (int i = 1; i < lod_vec.size(); i++) {
        sqlod_vec[i] = sqlod_vec[i - 1] + (lod_vec[i] - lod_vec[i - 1]) * (lod_vec[i] - lod_vec[i - 1]);
    }
    xdnn::VectorParam<int> sqlod = {sqlod_vec.data(), (int) sqlod_vec.size(), nullptr};

    xdnn::QKVAttnParam attr(lods,
                            param.input_k->dims()[1],
                            param.input_k->dims()[3],
                            xdnn::Activation_t::LINEAR,
                            -1,
                            false,
                            -1,
                            param.input_k->dims()[1] * param.input_k->dims()[3],
                            false,
                            false,
                            0,
                            {},
                            AttnMacMaxPtrType_t::ATTN_WHOLE_BATCH,
                            -1,
                            sqlod);
    
    std::vector<int> shape = {};
    for (size_t i = 0; i < param.input_q->dims().size(); i++) {
        shape.push_back(param.input_q->dims()[i]);
    }

    size_t size = shape[0] * shape[1] * shape[2] * shape[2];
    float* qk_output = nullptr;
    r = xpu_malloc(reinterpret_cast<void**>(&qk_output), size * sizeof(float));
    CHECK_EQ(r, 0);

    size = (size / shape[2]) * shape[3];
    float* q_input = nullptr;
    float* k_input = nullptr;
    float* v_input = nullptr;
    r = xpu_malloc(reinterpret_cast<void**>(&q_input), size * sizeof(float));
    CHECK_EQ(r, 0);
    r = xpu_malloc(reinterpret_cast<void**>(&k_input), size * sizeof(float));
    CHECK_EQ(r, 0);
    r = xpu_malloc(reinterpret_cast<void**>(&v_input), size * sizeof(float));
    CHECK_EQ(r, 0);
    std::vector<int> permute = {0, 2, 1, 3};
    r = xdnn::transpose<float>(ctx.GetRawContext(), 
                    param.input_q->template data<TQ>(),
                    q_input,
                    shape,
                    permute);
    CHECK_EQ(r, 0);
    r = xdnn::transpose<float>(ctx.GetRawContext(), 
                    param.input_k->template data<TQ>(),
                    k_input,
                    shape,
                    permute);
    CHECK_EQ(r, 0);
    r = xdnn::transpose<float>(ctx.GetRawContext(), 
                    param.input_v->template data<TQ>(),
                    v_input,
                    shape,
                    permute);
    CHECK_EQ(r, 0);

    r = xdnn::qk_attention<TQ, TK, TQK, TGEMM, TZ>(
        ctx.GetRawContext(), 
        k_input,
        q_input,
        (TQK*)qk_output,    
        nullptr,                       
        nullptr,                       
        nullptr,    
        attr,                           
        nullptr);        
    CHECK_EQ(r, 0);
    r = xdnn::qk_v_attention<TQ, TK, TQK, TGEMM, TZ>(
        ctx.GetRawContext(),
        (TQK*)qk_output, 
        v_input,
        q_input,
        nullptr,
        nullptr, 
        nullptr,
        attr); 
    CHECK_EQ(r, 0);
    r = xdnn::transpose<float>(ctx.GetRawContext(), 
                    q_input,
                    param.output->template mutable_data<TQK>(paddle::lite_api::TargetType::kXPU),
                    {shape[0], shape[2], shape[1], shape[3]},
                    permute);
    CHECK_EQ(r, 0);

    CHECK_EQ(xpu_free(qk_output), 0);
    CHECK_EQ(xpu_free(q_input), 0);
    CHECK_EQ(xpu_free(k_input), 0);
    CHECK_EQ(xpu_free(v_input), 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using XPUQKV_ATTENDION_FP32_LOCAL_QUANT =
        xpu::XPUQkvAttentionCompute<float, float, float, int16_t, float>;

REGISTER_LITE_KERNEL(__xpu__qkv_attention,
                     kXPU,
                     kFloat,
                     kNCHW,
                     XPUQKV_ATTENDION_FP32_LOCAL_QUANT,
                     def)
    .BindInput("input_q", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("input_k", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("input_v", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

