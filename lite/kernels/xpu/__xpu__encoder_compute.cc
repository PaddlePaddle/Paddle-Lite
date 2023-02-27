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

#include "lite/kernels/xpu/__xpu__encoder_compute.h"
#include <map>
#include <string>

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUEncoderCompute::PrepareForRun() {
  auto& ctx = this->ctx_->As<XPUContext>();
  auto& param = this->Param<param_t>();
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  // prepare bias
  for (auto* fc_bias : param.fc_bias) {
    fc_bias_.push_back(fc_bias->data<float>());
  }
  // prepare scale
  for (auto* ln_scale : param.ln_scale) {
    ln_scale_.push_back(ln_scale->data<float>());
  }
  // prepare ln_bias
  for (auto* ln_bias : param.ln_bias) {
    ln_bias_.push_back(ln_bias->data<float>());
  }

  // check precisions (six fc) TODO(TingShen) : Need more consideration.
  for (int i = 0; i < 6; ++i) {
    if (param.precision[i] == "int31") {
      high_precision_ = true;
    }
  }
  // prepare input_cast and output_cast guard_
  cast_in_guard_ = TargetWrapperXPU::MallocScratchPad(4 * 1024 * 1024);
  cast_out_guard_ = TargetWrapperXPU::MallocScratchPad(4 * 1024 * 1024);

  // prepare weights
  int weight_max_id = 0;
  XPUQuantData xpu_quant_weight;
  for (int layer = 0; layer < param.n_layers; ++layer) {
    for (int i = 0; i < 6; ++i) {
      if (param.quant_type[i] == "per_tensor" ||
          param.quant_type[i] == "per_channel") {
        int max_len = max_ptr_size;
        if (param.quant_type[i] == "per_channel") {
          if (i == 0 && param.enable_qkv_fusion) {
            max_len = 3 * param.head_num * param.head_dim;
          } else if (i == 3 || i == 5) {
            max_len = param.hidden_dim;
          } else if (i == 4) {
            max_len = param.intermediate_size;
          } else {
            max_len = param.head_num * param.head_dim;
          }
        }
        if (param.precision[i] == "int8") {
          xpu_quant_weight =
              TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int8_t,
                                                                 int8_t>(
                  param.fc_weight[layer * 6 + i]->data<int8_t>(),
                  param.fc_weight[layer * 6 + i]->dims(),
                  false,
                  max_len);
        } else if (param.precision[i] == "int16") {
          xpu_quant_weight =
              TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int16_t,
                                                                 int16_t>(
                  param.fc_weight[layer * 6 + i]->data<int16_t>(),
                  param.fc_weight[layer * 6 + i]->dims(),
                  false,
                  max_len);
        } else {
          LOG(FATAL) << "Quant only supports int8 and int16.";
        }
        std::vector<float> cpu_w_max(max_len, param.weight_max[weight_max_id]);
        if (param.quant_type[i] == "per_channel") {
          for (int j = 0; j < max_len; ++j) {
            cpu_w_max[j] = param.weight_max[weight_max_id + j];
          }
        }
        lite::TargetWrapperXPU::MemcpySync(xpu_quant_weight.max_ptr_,
                                           cpu_w_max.data(),
                                           sizeof(float) * max_len,
                                           IoDirection::HtoD);
        weight_max_id += param.quant_type[i] == "per_channel" ? max_len : 1;
      } else {
        if (param.precision[i] == "int8") {
          xpu_quant_weight =
              TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int8_t>(
                  param.fc_weight[layer * 6 + i]->data<float>(),
                  param.fc_weight[layer * 6 + i]->dims(),
                  false,
                  max_ptr_size);
        } else if (param.precision[i] == "int16") {
          xpu_quant_weight =
              TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float,
                                                                 int16_t>(
                  param.fc_weight[layer * 6 + i]->data<float>(),
                  param.fc_weight[layer * 6 + i]->dims(),
                  false,
                  max_ptr_size);
        } else if (param.precision[i] == "int31" ||
                   param.precision[i] == "local_quant") {
          xpu_quant_weight =
              TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, float>(
                  param.fc_weight[layer * 6 + i]->data<float>(),
                  param.fc_weight[layer * 6 + i]->dims(),
                  false,
                  max_ptr_size);
        } else {
          LOG(FATAL) << "Unsupported precision" << param.precision[i];
        }
        ++weight_max_id;
      }
      w_list_.push_back(
          reinterpret_cast<const int16_t*>(xpu_quant_weight.data_ptr_));
      weight_max_.push_back(xpu_quant_weight.max_ptr_);
      if (i == 0 && param.enable_qkv_fusion) {
        // skip two fc weights
        for (int j = 0; j < 2; ++j) {
          w_list_.push_back(nullptr);
          weight_max_.push_back(nullptr);
        }
        if (param.quant_type[i] != "per_channel") {
          weight_max_id += 2;
        }
        i += 2;
      }
    }
  }

  // prepare io_max
  io_max_.resize(param.io_max.size());
  io_max_guard_ = TargetWrapperXPU::MallocScratchPad(
      param.io_max.size() * max_ptr_size * sizeof(float));
  float* io_max_ptr = reinterpret_cast<float*>(io_max_guard_->addr_);
  std::vector<float> cpu_max(param.io_max.size() * max_ptr_size, 0);
  for (int i = 0; i < param.io_max.size(); ++i) {
    if (param.io_max[i] < 0) {
      io_max_[i] = nullptr;
    } else {
      io_max_[i] = io_max_ptr + i * max_ptr_size;
      for (int j = 0; j < max_ptr_size; ++j) {
        cpu_max[i * max_ptr_size + j] = param.io_max[i];
      }
    }
  }
  lite::TargetWrapperXPU::MemcpySync(
      io_max_ptr,
      cpu_max.data(),
      sizeof(float) * max_ptr_size * param.io_max.size(),
      IoDirection::HtoD);

  // prepare encoder_attr
  encoder_attr_ = xdnn::EncoderAttr<int>(param.head_num, param.head_dim);
  encoder_attr_.set_basic_param(
      xdnn::Activation_t((xdnn::Activation_t::act_enum)param.act_type),
      param.hidden_dim,
      param.intermediate_size,
      param.alpha);

  encoder_attr_.set_opt_param(param.enable_qkv_fusion,
                              (param.do_slice) ? 0 : -1);
  if (param.norm_before) {
    encoder_attr_.set_struct_param(xdnn::EncoderLnStruct_t::PRE_LN);
  }
  std::vector<xdnn::QuantType_t> quant_type;
  std::vector<xdnn::Dtype> precision;
  std::map<std::string, xdnn::Dtype> precision_map{
      {"int8", xdnn::Dtype::INT8},
      {"int16", xdnn::Dtype::INT16},
      {"int31", xdnn::Dtype::INT32},
      {"local_quant", xdnn::Dtype::FLOAT32}};
  std::map<std::string, xdnn::QuantType_t> quant_type_map{
      {"per_tensor", xdnn::QuantType_t::PER_TENSOR},
      {"per_channel", xdnn::QuantType_t::PER_CHANNEL},
      {"no_quant", xdnn::QuantType_t::NO_QUANT},
      {"local_quant", xdnn::QuantType_t::LOCAL_QUANT}};
  for (int i = 0; i < 8; ++i) {
    precision.push_back(precision_map[param.precision[i]]);
    quant_type.push_back(quant_type_map[param.quant_type[i]]);
  }
  encoder_attr_.set_quant_param(quant_type, precision, io_max_, weight_max_);
}

void XPUEncoderCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  VLOG(3) << "______xpu_encoder_____run";
  std::vector<int64_t> mask_shape = {};
  if (param.mask != nullptr) {
    mask_shape = param.mask->dims().Vectorize();
  }
  if (param.adaptive_seqlen) {
    seq_lod_ = std::vector<int>(param.input->lod()[0].begin(),
                                param.input->lod()[0].end());
    xdnn::VectorParam<int> query_lod = {
        seq_lod_.data(),
        static_cast<int64_t>(param.input->lod()[0].size()),
        nullptr};
    int max_pad_seqlen =
        (param.do_padding) ? param.padSeqLen->data<int>()[0] : -1;

    encoder_attr_.set_vsl_param(
        query_lod, max_pad_seqlen, -1, {nullptr, 0, nullptr}, mask_shape);
  } else {
    encoder_attr_.set_no_vsl_param(
        static_cast<int>(param.input->dims()[0]),  // batch
        static_cast<int>(param.input->dims()[1]),  // max_seqlen
        mask_shape);
  }
  if (ctx.GetRawContext()->dev().type() == xdnn::kXPU1 || high_precision_) {
    int r = xdnn::transformer_encoder<float, int16_t, int16_t>(
        ctx.GetRawContext(),
        param.input->data<float>(),
        w_list_,
        param.output->mutable_data<float>(TARGET(kXPU)),
        io_max_,
        weight_max_,
        fc_bias_,
        ln_scale_,
        ln_bias_,
        encoder_attr_,
        (param.mask == nullptr) ? nullptr : param.mask->data<float>());
    CHECK_EQ(r, 0);
  } else {
    cast_in_guard_->Reserve(param.input->numel() * sizeof(float16));
    cast_out_guard_->Reserve(param.output->numel() * sizeof(float16));
    int r = xdnn::cast_v2<float, float16>(
        ctx.GetRawContext(),
        param.input->data<float>(),
        reinterpret_cast<float16*>(cast_in_guard_->addr_),
        param.input->numel());
    CHECK_EQ(r, 0);
    r = xdnn::transformer_encoder<float16, int16_t, int16_t>(
        ctx.GetRawContext(),
        reinterpret_cast<const float16*>(cast_in_guard_->addr_),
        w_list_,
        reinterpret_cast<float16*>(cast_out_guard_->addr_),
        io_max_,
        weight_max_,
        fc_bias_,
        ln_scale_,
        ln_bias_,
        encoder_attr_,
        (param.mask == nullptr) ? nullptr : param.mask->data<float>());
    CHECK_EQ(r, 0);
    r = xdnn::cast_v2<float16, float>(
        ctx.GetRawContext(),
        reinterpret_cast<float16*>(cast_out_guard_->addr_),
        param.output->mutable_data<float>(TARGET(kXPU)),
        param.output->numel());
    CHECK_EQ(r, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__encoder,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUEncoderCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("SeqLod",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("PadSeqLen",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
