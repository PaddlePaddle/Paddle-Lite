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

#include "lite/kernels/xpu/__xpu__multi_encoder_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
static std::vector<const T*> prepare_weight(
    const std::vector<lite::Tensor*>& fc_weight) {
  std::vector<const T*> result;
  for (auto* weight : fc_weight) {
    result.push_back(reinterpret_cast<const T*>(weight->data<float>()));
  }
  return result;
}

template <typename T>
std::vector<const T*>* XPUMultiEncoderCompute::get_weight() {
  LOG(FATAL) << "Invalid Weight Type";
  return nullptr;
}

template <>
std::vector<const int16_t*>* XPUMultiEncoderCompute::get_weight() {
  return &arg_fc_weight_int16_;
}

template <>
std::vector<const int8_t*>* XPUMultiEncoderCompute::get_weight() {
  return &arg_fc_weight_int8_;
}

template <>
std::vector<const float*>* XPUMultiEncoderCompute::get_weight() {
  return &arg_fc_weight_fp32_;
}

void XPUMultiEncoderCompute::prepare_quant_max(
    const std::vector<float>& max_value,
    int n_layers,
    int max_ptr_len,
    std::vector<const float*>& max_xpu_ptrs) {
  bool mul_quant = false;
  bool matmul_quant = false;
  if (max_value.size() == (n_layers * 12)) {
    mul_quant = true;
  } else if (max_value.size() == (n_layers * 18)) {
    mul_quant = true;
    matmul_quant = true;
  } else if (max_value.size() == 0) {
    // dynamic quant, find max in xpu
    return;
  } else {
    LOG(FATAL) << "invalid quant max value for xpu encoder, "
               << max_value.size() << ", n_layers: " << n_layers;
  }
  // prepare input_max
  input_max_guard_ = TargetWrapperXPU::MallocScratchPad(
      max_value.size() * max_ptr_len * sizeof(float));
  float* input_max_ptr = reinterpret_cast<float*>(input_max_guard_->addr_);
  std::vector<float> cpu_max;
  cpu_max.resize(max_value.size() * max_ptr_len);
  for (int i = 0; i < max_value.size(); ++i) {
    for (int j = 0; j < max_ptr_len; ++j) {
      cpu_max[i * max_ptr_len + j] = max_value[i];
    }
  }
  lite::TargetWrapperXPU::MemcpySync(
      input_max_ptr,
      cpu_max.data(),
      sizeof(float) * max_ptr_len * max_value.size(),
      IoDirection::HtoD);
  for (int i = 0; i < max_ptr_len * max_value.size(); i += max_ptr_len) {
    max_xpu_ptrs.push_back(input_max_ptr + i);
  }
  if (matmul_quant) {
    CHECK_EQ(max_xpu_ptrs.size(), (n_layers * 18));
  } else {
    CHECK_EQ(max_xpu_ptrs.size(), (n_layers * 12));
  }
  return;
}

void XPUMultiEncoderCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();
  // prepare bias
  for (auto* fc_bias : param.fc_bias) {
    arg_fc_bias_.push_back(fc_bias->data<float>());
  }
  // prepare scale
  for (auto* ln_scale : param.ln_scale) {
    arg_ln_scale_.push_back(ln_scale->data<float>());
  }
  // prepare ln_bias
  for (auto* ln_bias : param.ln_bias) {
    arg_ln_bias_.push_back(ln_bias->data<float>());
  }
  // prepare weights
  if (param.precision == "int16") {
    arg_fc_weight_int16_ = prepare_weight<int16_t>(param.fc_weight);
  } else if (param.precision == "int8") {
    arg_fc_weight_int8_ = prepare_weight<int8_t>(param.fc_weight);
  } else if (param.precision == "int31") {
    arg_fc_weight_fp32_ = prepare_weight<float>(param.fc_weight);
  }
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  // prepare weight_max
  weight_max_guard_ = TargetWrapperXPU::MallocScratchPad(
      param.fc_weight_max->numel() * XPU_QUANT_SCALE_NUM * sizeof(float));
  float* weight_max_ptr = reinterpret_cast<float*>(weight_max_guard_->addr_);
  for (int i = 0; i < param.fc_weight_max->numel(); i++) {
    float* cur_weight_max_ptr = weight_max_ptr + i * XPU_QUANT_SCALE_NUM;
    std::vector<float> cpu_max(XPU_QUANT_SCALE_NUM,
                               param.fc_weight_max->data<float>()[i]);
    lite::TargetWrapperXPU::MemcpySync(cur_weight_max_ptr,
                                       cpu_max.data(),
                                       sizeof(float) * XPU_QUANT_SCALE_NUM,
                                       IoDirection::HtoD);
    fc_weight_max_.push_back(cur_weight_max_ptr);
  }
  // prepare quant max, mul&matmul input/output max
  const int n_layers = param.fc_weight.size() / 6;
  prepare_quant_max(
      param.input_max, n_layers, XPU_QUANT_SCALE_NUM, fc_input_max_);
  // prepare act_type
  if (param.act_type == "gelu") {
    qkv_act = xdnn::Activation_t::GELU;
  } else if (param.act_type != "relu") {
    CHECK(false) << "Invalid QKV Activation Type: " << param.act_type;
  }
  // prepare with sice
  if ((param.slice_starts.size() > 0 && param.slice_starts[0] == 0) &&
      (param.slice_ends.size() > 0 && param.slice_ends[0] == 1) &&
      (param.slice_axes.size() > 0 && param.slice_axes[0] == 1)) {
    slice_idx = 0;
  }
  // prepare input_cast and output_cast guard_
  cast_in_guard_ = TargetWrapperXPU::MallocScratchPad(4 * 1024 * 1024);
  cast_out_guard_ = TargetWrapperXPU::MallocScratchPad(4 * 1024 * 1024);
}

template <typename T, typename TW, typename TGEMM>
void XPUMultiEncoderCompute::run_encoder(const T* in, T* out) {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  xdnn::VectorParam<int> query_lod;
  if (param.SeqLod && param.SeqLod->data<int>()) {
    // vsl
    query_lod = {param.SeqLod->data<int>(),
                 static_cast<int>(param.SeqLod->numel()),
                 nullptr};
    int max_pad_seqlen = slice_idx == -1 ? param.PadSeqLen->data<int>()[0] : -1;
    xdnn::QKVAttnParam qkv_attn_param(query_lod, /* lod */
                                      param.head_num,
                                      param.size_per_head,
                                      qkv_act,
                                      slice_idx,
                                      true /* qkv fusion */,
                                      max_pad_seqlen,
                                      param.hidden_dim);
    if (std::is_same<TGEMM, int8_t>::value) {
      CHECK_GT(fc_input_max_.size(), 0);
    }
    int r = xdnn::transformer_encoder<T, TW, TGEMM>(
        ctx.GetRawContext(),
        in,
        *(XPUMultiEncoderCompute::get_weight<TW>()),
        out,
        fc_input_max_,
        fc_weight_max_,
        arg_fc_bias_,
        arg_ln_scale_,
        arg_ln_bias_,
        qkv_attn_param);
    CHECK_EQ(r, 0);
  } else {
    // no vsl
    int batch = static_cast<int>(param.input->dims()[0]);
    int max_seqlen = static_cast<int>(param.input->dims()[1]);
    std::vector<int64_t> mask_shape = param.mask->dims().Vectorize();
    std::vector<int> encoder_mask_shape =
        std::vector<int>(mask_shape.begin(), mask_shape.end());

    xdnn::QKVAttnParam qkv_attn_param(batch,
                                      max_seqlen,
                                      param.head_num,
                                      param.size_per_head,
                                      encoder_mask_shape,
                                      qkv_act,
                                      slice_idx,
                                      true,
                                      param.hidden_dim);
    int r = xdnn::transformer_encoder<T, TW, TGEMM>(
        ctx.GetRawContext(),
        in,
        *(XPUMultiEncoderCompute::get_weight<TW>()),
        out,
        fc_input_max_,
        fc_weight_max_,
        arg_fc_bias_,
        arg_ln_scale_,
        arg_ln_bias_,
        qkv_attn_param,
        param.mask->data<float>());
    CHECK_EQ(r, 0);
  }
}

void XPUMultiEncoderCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const float* in = param.input->data<float>();
  float* out = param.output->mutable_data<float>(TARGET(kXPU));
  if (ctx.GetRawContext()->dev().type() == xdnn::kXPU1) {
    if (param.precision == "int8") {
      run_encoder<float, int8_t, int8_t>(in, out);
    } else if (param.precision == "int16") {
      run_encoder<float, int16_t, int16_t>(in, out);
    } else if (param.precision == "int31") {
      run_encoder<float, float, int>(in, out);
    } else {
      CHECK(false);
    }
  } else {
    cast_in_guard_->Reserve(param.input->numel() * sizeof(float));
    cast_out_guard_->Reserve(param.output->numel() * sizeof(float));
    if (param.precision == "int8") {
      int r = xdnn::cast_v2<float, float16>(
          ctx.GetRawContext(),
          in,
          reinterpret_cast<float16*>(cast_in_guard_->addr_),
          param.input->numel());
      CHECK_EQ(r, 0);
      run_encoder<float16, int8_t, int8_t>(
          reinterpret_cast<const float16*>(cast_in_guard_->addr_),
          reinterpret_cast<float16*>(cast_out_guard_->addr_));
      r = xdnn::cast_v2<float16, float>(
          ctx.GetRawContext(),
          reinterpret_cast<float16*>(cast_out_guard_->addr_),
          out,
          param.output->numel());
      CHECK_EQ(r, 0);
    } else if (param.precision == "int16") {
      int r = xdnn::cast_v2<float, float16>(
          ctx.GetRawContext(),
          in,
          reinterpret_cast<float16*>(cast_in_guard_->addr_),
          param.input->numel());
      CHECK_EQ(r, 0);
      run_encoder<float16, int16_t, int16_t>(
          reinterpret_cast<const float16*>(cast_in_guard_->addr_),
          reinterpret_cast<float16*>(cast_out_guard_->addr_));
      r = xdnn::cast_v2<float16, float>(
          ctx.GetRawContext(),
          reinterpret_cast<float16*>(cast_out_guard_->addr_),
          out,
          param.output->numel());
      CHECK_EQ(r, 0);
    } else if (param.precision == "int31") {
      run_encoder<float, float, int>(in, out);
    } else {
      CHECK(false);
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__multi_encoder,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMultiEncoderCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("SeqLod",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("PadSeqLen",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeightMax", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
