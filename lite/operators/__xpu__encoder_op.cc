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

#include "lite/operators/__xpu__encoder_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUEncoderOp::CheckShape() const {
  if (param_.input->dims().size() == 3) {
    CHECK_EQ(param_.input->dims()[2], param_.hidden_dim);
  } else if (param_.input->dims().size() == 2) {
    CHECK_EQ(param_.input->dims()[1], param_.hidden_dim);
  } else {
    LOG(FATAL) << "Unsupported encoder input dim";
  }
  if (param_.do_slice) {
    CHECK(!param_.do_padding);
  }
  return true;
}

bool XPUEncoderOp::InferShapeImpl() const {
  auto input_shape = param_.input->dims();
  auto batch_size = input_shape[0];
  auto seq_len = input_shape[1];
  if (param_.seqLod && param_.seqLod->data<int>()) {
    batch_size = param_.input->lod()[0].size() - 1;
    seq_len = param_.padSeqLen->data<int>()[0];
  }
  if (param_.do_slice) {
    if (param_.has_slice_decrease_axis) {
      param_.output->Resize({batch_size, param_.hidden_dim});
    } else {
      param_.output->Resize({batch_size, 1, param_.hidden_dim});
    }
  } else {
    param_.output->Resize({batch_size, seq_len, param_.hidden_dim});
  }
  return true;
}

bool XPUEncoderOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  // Set Attrs
  param_.n_layers = op_desc.GetAttr<int>("n_layers");
  param_.hidden_dim = op_desc.GetAttr<int>("hidden_dim");
  param_.intermediate_size = op_desc.GetAttr<int>("intermediate_size");
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.head_dim = op_desc.GetAttr<int>("head_dim");
  param_.act_type = op_desc.GetAttr<int>("act_type");
  param_.alpha = op_desc.GetAttr<float>("alpha");
  param_.enable_qkv_fusion = op_desc.GetAttr<bool>("enable_qkv_fusion");
  param_.norm_before = op_desc.GetAttr<bool>("norm_before");
  param_.adaptive_seqlen = op_desc.GetAttr<bool>("adaptive_seqlen");
  param_.do_slice = op_desc.GetAttr<bool>("do_slice");
  if (op_desc.HasAttr("has_slice_decrease_axis")) {
    param_.has_slice_decrease_axis =
        op_desc.GetAttr<bool>("has_slice_decrease_axis");
  }
  param_.do_padding = op_desc.GetAttr<bool>("do_padding");
  param_.quant_type = op_desc.GetAttr<std::vector<std::string>>("quant_type");
  param_.weight_max = op_desc.GetAttr<std::vector<float>>("weight_max");
  param_.io_max = op_desc.GetAttr<std::vector<float>>("io_max");
  param_.precision = op_desc.GetAttr<std::vector<std::string>>("precision");

  // Attach tensors
  param_.input = const_cast<lite::Tensor*>(
      &scope->FindVar(op_desc.Input("Input").front())->Get<lite::Tensor>());
  param_.output = scope->FindVar(op_desc.Output("Output").front())
                      ->GetMutable<lite::Tensor>();

  param_.fc_weight.clear();
  for (auto& name : op_desc.Input("FCWeight")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.fc_weight.push_back(t);
  }

  param_.fc_bias.clear();
  for (auto& name : op_desc.Input("FCBias")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.fc_bias.push_back(t);
  }

  param_.ln_scale.clear();
  for (auto& name : op_desc.Input("LNScale")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.ln_scale.push_back(t);
  }

  param_.ln_bias.clear();
  for (auto& name : op_desc.Input("LNBias")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.ln_bias.push_back(t);
  }

  if (param_.adaptive_seqlen) {
    param_.seqLod = const_cast<lite::Tensor*>(
        &scope->FindVar(op_desc.Input("SeqLod").front())->Get<lite::Tensor>());
    param_.padSeqLen = const_cast<lite::Tensor*>(
        &scope->FindVar(op_desc.Input("PadSeqLen").front())
             ->Get<lite::Tensor>());
  }

  if (op_desc.HasInput("Mask") && op_desc.Input("Mask").size() > 0) {
    param_.mask = const_cast<lite::Tensor*>(
        &scope->FindVar(op_desc.Input("Mask").front())->Get<lite::Tensor>());
  } else {
    param_.mask = nullptr;
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__encoder, paddle::lite::operators::XPUEncoderOp);
