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
    const std::vector<lite::Tensor*>& weight) {
  std::vector<const T*> result;
  for (auto* w : weight) {
    result.push_back(reinterpret_cast<const T*>(w->data<float>()));
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

template <>
std::vector<const float16*>* XPUMultiEncoderCompute::get_weight() {
  return &arg_fc_weight_fp16_;
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
  max_xpu_ptrs.resize(max_value.size());
  for (int i = 0; i < max_value.size(); i += 1) {
    max_xpu_ptrs[i] = input_max_ptr + i * max_ptr_len;
  }
  return;
}

void XPUMultiEncoderCompute::prepare_weight_max(
    bool per_channel,
    const std::vector<lite::Tensor*>& weight_max,
    int max_ptr_len,
    std::vector<const float*>& max_xpu_ptrs,
    const std::vector<std::string>& quant_types) {
  int max_value_num = 0;
  // weight_max per mul:
  // per_channel quant: 1 * numel()
  // per_tensor quant: max_ptr_len * numel()
  // not_quantized/skip quant : max_ptr_len * numel()
  for (int i = 0; i < weight_max.size(); ++i) {
    VLOG(6) << "quant_types[" << i << "] is " << quant_types[i];
    int index_mapping = (i / 6) * 8 + i % 6;
    if (per_channel && quant_types[index_mapping] != "not_quantized") {
      max_value_num += weight_max[i]->numel();
    } else {
      CHECK_EQ(weight_max[i]->numel(), 1);
      max_value_num += weight_max[i]->numel() * max_ptr_len;
    }
  }
  VLOG(3) << "Total weight max value number: " << max_value_num;
  weight_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* weight_max_ptr = reinterpret_cast<float*>(weight_max_guard_->addr_);

  int offset = 0;
  for (int i = 0; i < weight_max.size(); ++i) {
    float* cur_weight_max_ptr = weight_max_ptr + offset;
    auto len = weight_max[i]->numel();
    VLOG(6) << "weight max value: " << weight_max[i]->data<float>()[0] << " "
            << weight_max[i]->data<float>()[len - 1];
    int index_mapping = (i / 6) * 8 + i % 6;
    if (per_channel && quant_types[index_mapping] != "not_quantized") {
      lite::TargetWrapperXPU::MemcpySync(cur_weight_max_ptr,
                                         weight_max[i]->raw_data(),
                                         sizeof(float) * len,
                                         IoDirection::HtoD);
      max_xpu_ptrs.push_back(cur_weight_max_ptr);
      offset += len;
    } else {
      std::vector<float> cpu_max(max_ptr_len, weight_max[i]->data<float>()[0]);
      lite::TargetWrapperXPU::MemcpySync(cur_weight_max_ptr,
                                         cpu_max.data(),
                                         sizeof(float) * max_ptr_len,
                                         IoDirection::HtoD);
      max_xpu_ptrs.push_back(cur_weight_max_ptr);
      offset += max_ptr_len;
    }
  }
}

void XPUMultiEncoderCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();
  const int n_layers = param.fc_weight.size() / 6;
  // prepare bias
  if (param.already_qkv_fusion) {
    // only 3 or 4 bias per layer if qkv is already be fusioned.
    CHECK((param.fc_bias.size() == 3 * n_layers) ||
          (param.fc_bias.size() == 4 * n_layers))
        << "bias num per layer shouble be 3 or 4";
    int num_per_layer = param.fc_bias.size() / n_layers;
    for (int i = 0; i < n_layers; i++) {
      for (int k = 0; k < num_per_layer; k++) {
        arg_fc_bias_.push_back(
            param.fc_bias[num_per_layer * i + k]->data<float>());
      }
      // insert 2/3 nullptr
      arg_fc_bias_.insert(arg_fc_bias_.end() - 3, 6 - num_per_layer, nullptr);
    }
  } else {
    for (auto* fc_bias : param.fc_bias) {
      arg_fc_bias_.push_back(fc_bias->data<float>());
    }
  }
  // prepare scale
  for (auto* ln_scale : param.ln_scale) {
    arg_ln_scale_.push_back(ln_scale->data<float>());
  }
  // prepare ln_bias
  for (auto* ln_bias : param.ln_bias) {
    arg_ln_bias_.push_back(ln_bias->data<float>());
  }
  // prepare smooth_quant_scale
  is_smooth_quant_ = param.is_smooth_quant;
  if (is_smooth_quant_) {
    smooth_quant_scale_ = prepare_weight<float16>(param.smooth_quant_scale);
  }
  relative_type_ = param.relative_type;
  // prepare roformer embedding
  if (relative_type_ == 1) {
    for (auto* emb : param.roformer_embedding) {
      roformer_embedding_.push_back(emb->data<float>());
    }
  }
  // prepare weights
  CHECK(lite::TargetWrapperXPU::xpu_runtime_ptr)
      << "xpu_runtime_ptr null in run";
  local_quant_ = GetBoolFromEnv(
      "XPU_LOCAL_QUANT", lite::TargetWrapperXPU::xpu_runtime_ptr->local_quant);
  if (param.precision == "int16") {
    if (local_quant_) {
      arg_fc_weight_fp16_ = prepare_weight<float16>(param.fc_weight);
    } else {
      arg_fc_weight_int16_ = prepare_weight<int16_t>(param.fc_weight);
    }
  } else if (param.precision == "int8") {
    arg_fc_weight_int8_ = prepare_weight<int8_t>(param.fc_weight);
  } else if (param.precision == "int31") {
    arg_fc_weight_fp32_ = prepare_weight<float>(param.fc_weight);
  }

  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  prepare_weight_max(param.per_channel,
                     param.weight_max,
                     XPU_QUANT_SCALE_NUM,
                     fc_weight_max_,
                     param.quant_types);
  // prepare quant max, mul&matmul input/output max
  prepare_quant_max(
      param.input_max, n_layers, XPU_QUANT_SCALE_NUM, fc_input_max_);
  // prepare quant type
  for (auto quant_type : param.quant_types) {
    if (quant_type == "enable_int8") {
      quant_types_.push_back(xdnn::QuantType::QUANT_INT8);
    } else if (quant_type == "enable_int16") {
      quant_types_.push_back(xdnn::QuantType::QUANT_INT16);
    } else {
      quant_types_.push_back(xdnn::QuantType::NOT_QUANT);
    }
  }
  // prepare act_type
  if (param.act_type == "gelu") {
    qkv_act = xdnn::Activation_t::GELU;
  } else if (param.act_type == "__xpu__quick_gelu") {
    qkv_act = xdnn::Activation_t::QUICK_GELU;
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
  const bool has_token_sliced_layer = param.has_token_sliced_layer;
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
                                      param.hidden_dim,
                                      param.norm_before, /*is_pre_norm*/
                                      param.per_channel);
    if (has_token_sliced_layer) {
      qkv_attn_param.with_token_slice = true;
      qkv_attn_param.attn_sliced_length.assign(
          param.token_sliced_length.begin(), param.token_sliced_length.end());
    }
    if (param.softmax_max.size()) {
      qkv_attn_param.ptq_max_value = param.softmax_max;
    }

    qkv_attn_param.quant_type_.assign(quant_types_.begin(), quant_types_.end());
    if (is_smooth_quant_) {
      qkv_attn_param.is_smooth_quant = true;
      qkv_attn_param.smooth_scale.assign(smooth_quant_scale_.begin(),
                                         smooth_quant_scale_.end());
    }
    if (relative_type_ == 1) {
      qkv_attn_param.relative_type = relative_type_;
      qkv_attn_param.max_pos_len = param.max_pos_len;
      qkv_attn_param.relative_pos.assign(roformer_embedding_.begin(),
                                         roformer_embedding_.end());
    }
    qkv_attn_param.scale_of_hidden_units = param.ffn_hidden_dim_scale;
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
  } else if (param.mask == nullptr) {
    // When no mask input, like VIT, create LOD to act as vsl.
    int batch = static_cast<int>(param.input->dims()[0]);
    int max_seqlen = static_cast<int>(param.input->dims()[1]);
    std::vector<int> lod;
    for (int i = 0; i < batch + 1; i++) {
      lod.push_back(i * max_seqlen);
    }
    query_lod = {lod.data(), static_cast<int>(lod.size()), nullptr};
    // No need to pad, no matter slice or not
    int max_pad_seqlen = -1;
    xdnn::QKVAttnParam qkv_attn_param(query_lod, /* lod */
                                      param.head_num,
                                      param.size_per_head,
                                      qkv_act,
                                      slice_idx,
                                      true /* qkv fusion */,
                                      max_pad_seqlen,
                                      param.hidden_dim,
                                      param.norm_before, /*is_pre_norm*/
                                      param.per_channel);
    if (has_token_sliced_layer) {
      qkv_attn_param.with_token_slice = true;
      qkv_attn_param.attn_sliced_length.assign(
          param.token_sliced_length.begin(), param.token_sliced_length.end());
    }
    if (param.softmax_max.size()) {
      qkv_attn_param.ptq_max_value = param.softmax_max;
    }
    qkv_attn_param.quant_type_.assign(quant_types_.begin(), quant_types_.end());
    if (is_smooth_quant_) {
      qkv_attn_param.is_smooth_quant = true;
      qkv_attn_param.smooth_scale.assign(smooth_quant_scale_.begin(),
                                         smooth_quant_scale_.end());
    }
    if (relative_type_ == 1) {
      qkv_attn_param.relative_type = relative_type_;
      qkv_attn_param.max_pos_len = param.max_pos_len;
      qkv_attn_param.relative_pos.assign(roformer_embedding_.begin(),
                                         roformer_embedding_.end());
    }
    qkv_attn_param.scale_of_hidden_units = param.ffn_hidden_dim_scale;
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
    // xpu1 don't support ffn_hidden_dim_scale!=4 when no vsl
    if (ctx.GetRawContext()->dev().type() == xdnn::kXPU1) {
      CHECK_EQ(param.ffn_hidden_dim_scale, 4)
          << "xpu don't support ffn_hidden_dim_scale!=4 when no vsl";
    }
    xdnn::QKVAttnParam qkv_attn_param(batch,
                                      max_seqlen,
                                      param.head_num,
                                      param.size_per_head,
                                      encoder_mask_shape,
                                      qkv_act,
                                      slice_idx,
                                      true,
                                      param.hidden_dim,
                                      param.norm_before,
                                      param.per_channel);
    if (has_token_sliced_layer) {
      qkv_attn_param.with_token_slice = true;
      qkv_attn_param.attn_sliced_length.assign(
          param.token_sliced_length.begin(), param.token_sliced_length.end());
    }
    if (param.softmax_max.size()) {
      qkv_attn_param.ptq_max_value = param.softmax_max;
    }
    qkv_attn_param.quant_type_.assign(quant_types_.begin(), quant_types_.end());
    if (is_smooth_quant_) {
      qkv_attn_param.is_smooth_quant = true;
      qkv_attn_param.smooth_scale.assign(smooth_quant_scale_.begin(),
                                         smooth_quant_scale_.end());
    }
    if (relative_type_ == 1) {
      qkv_attn_param.relative_type = relative_type_;
      qkv_attn_param.max_pos_len = param.max_pos_len;
      qkv_attn_param.relative_pos.assign(roformer_embedding_.begin(),
                                         roformer_embedding_.end());
    }
    qkv_attn_param.scale_of_hidden_units = param.ffn_hidden_dim_scale;
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
      if (local_quant_) {
        run_encoder<float, float16, float>(in, out);
      } else {
        run_encoder<float, int16_t, int16_t>(in, out);
      }
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

      if (local_quant_) {
        run_encoder<float16, float16, float>(
            reinterpret_cast<const float16*>(cast_in_guard_->addr_),
            reinterpret_cast<float16*>(cast_out_guard_->addr_));
      } else {
        run_encoder<float16, int16_t, int16_t>(
            reinterpret_cast<const float16*>(cast_in_guard_->addr_),
            reinterpret_cast<float16*>(cast_out_guard_->addr_));
      }

      r = xdnn::cast_v2<float16, float>(
          ctx.GetRawContext(),
          reinterpret_cast<float16*>(cast_out_guard_->addr_),
          out,
          param.output->numel());
      CHECK_EQ(r, 0);
    } else if (param.precision == "int31") {
      if (local_quant_) {
        run_encoder<float, float, float>(in, out);
      } else {
        run_encoder<float, float, int>(in, out);
      }
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
    .BindInput("RoformerEmbedding", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("SmoothQuantScaleWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
