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
#include <memory>
#include <string>
#include <utility>
#include "lite/backends/xpu/math.h"

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

void XPUMultiEncoderCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  const std::vector<lite::Tensor*>& fc_weight = param.fc_weight;
  bool enable_qkv_fusion = param.enable_qkv_fusion;
  const std::string& fc_precision = param.precision;
  arg_fc_weight_max_.assign(fc_weight.size(), 0.0f);
  for (size_t i = 0; i < fc_weight.size(); ++i) {
    if (enable_qkv_fusion && (i % 6 == 0)) {
      // q/k/v FCWeight fusion
      auto* weight_q = fc_weight[i];
      auto* weight_k = fc_weight[i + 1];
      auto* weight_v = fc_weight[i + 2];
      auto weight_q_dims = weight_q->dims();
      auto weight_k_dims = weight_k->dims();
      auto weight_v_dims = weight_v->dims();
      int weight_q_len = weight_q->numel();
      int weight_k_len = weight_k->numel();
      int weight_v_len = weight_v->numel();
      float* weight_q_on_host = weight_q->mutable_data<float>();
      float* weight_k_on_host = weight_k->mutable_data<float>();
      float* weight_v_on_host = weight_v->mutable_data<float>();
      int qkv_len = weight_q_len + weight_k_len + weight_v_len;
      int qkv_offset = 0;
      CHECK_EQ(weight_q_dims[0], weight_k_dims[0]);
      CHECK_EQ(weight_q_dims[0], weight_v_dims[0]);

      // 1. transpose
      std::unique_ptr<float[]> weight_q_trans(new float[weight_q_len]);
      std::unique_ptr<float[]> weight_k_trans(new float[weight_k_len]);
      std::unique_ptr<float[]> weight_v_trans(new float[weight_v_len]);
      std::unique_ptr<float[]> weight_qkv_trans(new float[qkv_len]);
      paddle::lite::xpu::math::Transpose(weight_q_on_host,
                                         weight_q_trans.get(),
                                         weight_q_dims[0],
                                         weight_q_dims[1]);
      paddle::lite::xpu::math::Transpose(weight_k_on_host,
                                         weight_k_trans.get(),
                                         weight_k_dims[0],
                                         weight_k_dims[1]);
      paddle::lite::xpu::math::Transpose(weight_v_on_host,
                                         weight_v_trans.get(),
                                         weight_v_dims[0],
                                         weight_v_dims[1]);

      // 2. concat
      memcpy(weight_qkv_trans.get() + qkv_offset,
             weight_q_trans.get(),
             weight_q_len * sizeof(float));
      qkv_offset += weight_q_len;
      memcpy(weight_qkv_trans.get() + qkv_offset,
             weight_k_trans.get(),
             weight_k_len * sizeof(float));
      qkv_offset += weight_k_len;
      memcpy(weight_qkv_trans.get() + qkv_offset,
             weight_v_trans.get(),
             weight_v_len * sizeof(float));
      qkv_offset += weight_v_len;
      CHECK_EQ(qkv_offset, qkv_len);

      // 3. int31 or int16 or int8
      int size_factor = 1;
      std::unique_ptr<int8_t[]> arg_buf(new int8_t[qkv_len * sizeof(float)]);
      void* arg_data = arg_buf.get();
      float max_f =
          paddle::lite::xpu::math::FindMaxAbs(weight_qkv_trans.get(), qkv_len);
      arg_fc_weight_max_[i] = max_f;
      VLOG(3) << "QKV fused FC-" << i << ", weight_max:" << max_f;

      if (fc_precision == "int31") {
        size_factor = sizeof(float);
        arg_data = weight_qkv_trans.get();
      } else if (fc_precision == "int8") {
        size_factor = sizeof(int8_t);
        paddle::lite::xpu::math::ConvertFP32ToInt8(
            weight_qkv_trans.get(), arg_data, max_f, qkv_len);
      } else {
        size_factor = sizeof(int16_t);
        paddle::lite::xpu::math::ConvertFP32ToInt16(
            weight_qkv_trans.get(), arg_data, max_f, qkv_len);
      }

      auto arg_guard =
          TargetWrapperXPU::MallocScratchPad(qkv_len * size_factor);
      arg_fc_weight_.push_back(arg_guard->addr_);
      XPU_CALL(xpu_memcpy(arg_guard->addr_,
                          arg_data,
                          qkv_len * size_factor,
                          XPUMemcpyKind::XPU_HOST_TO_DEVICE));
      fc_weight_guard_.push_back(std::move(arg_guard));

      continue;
    } else if (enable_qkv_fusion && ((i % 6 == 1) || (i % 6 == 2))) {
      // to fool param validation check in xdnn, it's not used actually
      arg_fc_weight_.push_back(arg_fc_weight_.back());
      continue;
    }

    // no q/k/v fusion
    auto* weight_t = fc_weight[i];
    auto weight_dims = weight_t->dims();
    int weight_len = weight_t->numel();
    float* weight_on_host = weight_t->mutable_data<float>();

    std::unique_ptr<float[]> weight_trans(new float[weight_len]);
    paddle::lite::xpu::math::Transpose(
        weight_on_host, weight_trans.get(), weight_dims[0], weight_dims[1]);

    int size_factor = 1;
    std::unique_ptr<int8_t[]> arg_buf(new int8_t[weight_len * sizeof(float)]);
    void* arg_data = arg_buf.get();
    float max_f =
        paddle::lite::xpu::math::FindMaxAbs(weight_on_host, weight_len);
    arg_fc_weight_max_[i] = max_f;
    VLOG(3) << "FC-" << i << ", weight_max:" << max_f;

    // i ranges from 0 to 6*encoder_num, so we need to do i%6 to get relative
    // position in the encoder
    if (fc_precision == "int31") {
      // FCs in encoder use int31
      size_factor = sizeof(float);
      arg_data = weight_trans.get();
    } else if (fc_precision == "int8") {
      size_factor = sizeof(int8_t);
      paddle::lite::xpu::math::ConvertFP32ToInt8(
          weight_trans.get(), arg_data, max_f, weight_len);
    } else {
      size_factor = sizeof(int16_t);
      paddle::lite::xpu::math::ConvertFP32ToInt16(
          weight_trans.get(), arg_data, max_f, weight_len);
    }

    auto arg_guard =
        TargetWrapperXPU::MallocScratchPad(weight_len * size_factor);
    arg_fc_weight_.push_back(arg_guard->addr_);
    XPU_CALL(xpu_memcpy(arg_guard->addr_,
                        arg_data,
                        weight_len * size_factor,
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    fc_weight_guard_.push_back(std::move(arg_guard));
  }

  const std::vector<lite::Tensor*>& fc_bias = param.fc_bias;
  for (size_t i = 0; i < fc_bias.size(); ++i) {
    if (enable_qkv_fusion && (i % 6 == 0)) {
      // q/k/v FCBias fusion
      VLOG(3) << "Copy bias in QKV fused FC-" << i << ", " << i / 6 << "-"
              << i % 6;
      auto* bias_q = fc_bias[i];
      auto* bias_k = fc_bias[i + 1];
      auto* bias_v = fc_bias[i + 2];
      auto bias_q_dims = bias_q->dims();
      auto bias_k_dims = bias_k->dims();
      auto bias_v_dims = bias_v->dims();
      int bias_q_len = bias_q->numel();
      int bias_k_len = bias_k->numel();
      int bias_v_len = bias_v->numel();
      float* bias_q_on_host = bias_q->mutable_data<float>();
      float* bias_k_on_host = bias_k->mutable_data<float>();
      float* bias_v_on_host = bias_v->mutable_data<float>();
      int qkv_len = bias_q_len + bias_k_len + bias_v_len;
      int qkv_offset = 0;
      CHECK_EQ(bias_q_dims.size(), 1);
      CHECK_EQ(bias_k_dims.size(), 1);
      CHECK_EQ(bias_v_dims.size(), 1);

      std::unique_ptr<float[]> bias_qkv(new float[qkv_len]);
      memcpy(bias_qkv.get() + qkv_offset,
             bias_q_on_host,
             bias_q_len * sizeof(float));
      qkv_offset += bias_q_len;
      memcpy(bias_qkv.get() + qkv_offset,
             bias_k_on_host,
             bias_k_len * sizeof(float));
      qkv_offset += bias_k_len;
      memcpy(bias_qkv.get() + qkv_offset,
             bias_v_on_host,
             bias_v_len * sizeof(float));
      qkv_offset += bias_v_len;
      CHECK_EQ(qkv_offset, qkv_len);

      auto arg_guard =
          TargetWrapperXPU::MallocScratchPad(qkv_len * sizeof(float));
      arg_fc_bias_.push_back(reinterpret_cast<float*>(arg_guard->addr_));
      XPU_CALL(xpu_memcpy(arg_guard->addr_,
                          bias_qkv.get(),
                          qkv_len * sizeof(float),
                          XPUMemcpyKind::XPU_HOST_TO_DEVICE));
      fc_bias_guard_.push_back(std::move(arg_guard));

      continue;
    } else if (enable_qkv_fusion && ((i % 6 == 1) || (i % 6 == 2))) {
      // to fool param validation check in xdnn, it's not used actually
      arg_fc_bias_.push_back(arg_fc_bias_.back());
      continue;
    }

    auto* bias_t = fc_bias[i];
    int bias_len = bias_t->numel();
    float* bias_on_host = bias_t->mutable_data<float>();

    auto arg_guard =
        TargetWrapperXPU::MallocScratchPad(bias_len * sizeof(float));
    arg_fc_bias_.push_back(reinterpret_cast<float*>(arg_guard->addr_));
    XPU_CALL(xpu_memcpy(arg_guard->addr_,
                        bias_on_host,
                        bias_len * sizeof(float),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    fc_bias_guard_.push_back(std::move(arg_guard));
  }

  for (auto* ln_scale : param.ln_scale) {
    arg_ln_scale_.push_back(ln_scale->data<float>());
  }
  for (auto* ln_bias : param.ln_bias) {
    arg_ln_bias_.push_back(ln_bias->data<float>());
  }

  encoder_param_.head_num = param.head_num;
  encoder_param_.size_per_head = param.size_per_head;
  encoder_param_.n_layers = param.n_layers;
  encoder_param_.pretrans_b = true;
  encoder_param_.use_l3 = true;
  encoder_param_.slice_starts = param.slice_starts;
  encoder_param_.slice_ends = param.slice_ends;
  encoder_param_.slice_axes = param.slice_axes;
  if (param.act_type == "relu") {
    encoder_param_.act_type = xdnn::Activation_t::RELU;
  } else if (param.act_type == "gelu") {
    encoder_param_.act_type = xdnn::Activation_t::GELU;
  }
}

int XPUMultiEncoderCompute::bert_encoder_run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  ctx.GetRawContext()->qkv_fusion = param.enable_qkv_fusion;

  int r = -1;
  if (param.precision == "int31") {
    r = xdnn::bert_encoder_transformer_int31(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        *reinterpret_cast<std::vector<const float*>*>(
            &arg_fc_weight_),      /* fc_weights */
        arg_fc_bias_,              /* fc_biass */
        arg_ln_scale_,             /* ln_scales */
        arg_ln_bias_,              /* ln_biass */
        arg_fc_weight_max_.data(), /* fc_weights_max */
        encoder_param_);
  } else if (param.precision == "int8") {
    r = xdnn::bert_encoder_transformer_int8<float, int8_t, float>(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        *reinterpret_cast<std::vector<const int8_t*>*>(
            &arg_fc_weight_),      /* fc_weights */
        arg_fc_bias_,              /* fc_biass */
        arg_ln_scale_,             /* ln_scales */
        arg_ln_bias_,              /* ln_biass */
        arg_fc_weight_max_.data(), /* fc_weights_max */
        encoder_param_);
  } else {
    r = xdnn::bert_encoder_transformer_int16<float, int16_t, float>(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        *reinterpret_cast<std::vector<const int16_t*>*>(
            &arg_fc_weight_),      /* fc_weights */
        arg_fc_bias_,              /* fc_biass */
        arg_ln_scale_,             /* ln_scales */
        arg_ln_bias_,              /* ln_biass */
        arg_fc_weight_max_.data(), /* fc_weights_max */
        encoder_param_);
  }
  return r;
}

int XPUMultiEncoderCompute::transformer_encoder_run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  ctx.GetRawContext()->qkv_fusion = param.enable_qkv_fusion;

  int r = -1;
  if (param.precision == "int31") {
    LOG(FATAL) << "Not support int31 at now";
  } else if (param.precision == "int8") {
    LOG(FATAL) << "Not support int8 at now";
  } else {
    r = xdnn::transformer_encoder_int16<float, int16_t, float>(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        *reinterpret_cast<std::vector<const int16_t*>*>(
            &arg_fc_weight_),      /* fc_weights */
        arg_fc_bias_,              /* fc_biass */
        arg_ln_scale_,             /* ln_scales */
        arg_ln_bias_,              /* ln_biass */
        arg_fc_weight_max_.data(), /* fc_weights_max */
        encoder_param_);
  }
  return r;
}

void XPUMultiEncoderCompute::Run() {
  auto& param = this->Param<param_t>();
  std::vector<int64_t> mask_shape = param.mask->dims().Vectorize();
  encoder_param_.mask_shape =
      std::vector<int>(mask_shape.begin(), mask_shape.end());
  encoder_param_.slice_starts = param.slice_starts;
  encoder_param_.slice_ends = param.slice_ends;
  encoder_param_.slice_axes = param.slice_axes;
  const bool norm_before_ = param.norm_before;
  if (param.SeqLod && param.SeqLod->data<int>()) {
    auto& ctx = this->ctx_->As<XPUContext>();
    ctx.GetRawContext()->batch_split_type = -1;  // disable auto split batch
    encoder_param_.seq_lod.resize(param.SeqLod->numel());
    memcpy(encoder_param_.seq_lod.data(),
           param.SeqLod->data<int>(),
           sizeof(int) * param.SeqLod->numel());
    encoder_param_.adaptive_seqlen = true;
    encoder_param_.batch_size = param.SeqLod->numel() - 1;
    encoder_param_.from_seq_len = param.PadSeqLen->data<int>()[0];
    encoder_param_.to_seq_len = param.PadSeqLen->data<int>()[0];
  } else {
    encoder_param_.adaptive_seqlen = false;
    encoder_param_.batch_size = param.input->dims()[0];
    encoder_param_.from_seq_len = param.input->dims()[1];
    encoder_param_.to_seq_len = param.input->dims()[1];
  }
  int r = -1;
  if (norm_before_) {
    r = transformer_encoder_run();
  } else {
    r = bert_encoder_run();
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// |kHost| means that tensors are managed internally
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
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
