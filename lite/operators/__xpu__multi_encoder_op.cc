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

#include "lite/operators/__xpu__multi_encoder_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUMultiEncoderOp::CheckShape() const {
  CHECK_EQ(param_.input->dims().size(), 3UL);
  return true;
}

bool XPUMultiEncoderOp::InferShapeImpl() const {
  auto input_shape = param_.input->dims();
  auto batch_size = input_shape[0];
  auto seq_len = input_shape[1];
  auto head_num = input_shape[2];
  if (param_.SeqLod && param_.SeqLod->data<int>()) {
    batch_size = param_.SeqLod->numel() - 1;
    int seq_pad_len = 0;
    for (auto i = 1; i < param_.SeqLod->numel(); i++) {
      int cur_seqlen =
          param_.SeqLod->data<int>()[i] - param_.SeqLod->data<int>()[i - 1];
      seq_pad_len = seq_pad_len > cur_seqlen ? seq_pad_len : cur_seqlen;
    }
    seq_len = seq_pad_len;
  }
  if ((param_.slice_starts.size() > 0 && param_.slice_starts[0] == 0) &&
      (param_.slice_ends.size() > 0 && param_.slice_ends[0] == 1) &&
      (param_.slice_axes.size() > 0 && param_.slice_axes[0] == 1)) {
    param_.output->Resize({batch_size, 1, head_num});
  } else {
    param_.output->Resize({batch_size, seq_len, head_num});
  }
  return true;
}

bool XPUMultiEncoderOp::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
  param_.input = const_cast<lite::Tensor*>(
      &scope->FindVar(op_desc.Input("Input").front())->Get<lite::Tensor>());
  param_.fc_weight_max = const_cast<lite::Tensor*>(
      &scope->FindVar(op_desc.Input("FCWeightMax").front())
           ->Get<lite::Tensor>());
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

  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "SeqLod") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("SeqLod");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.SeqLod = &(arg_var->Get<lite::Tensor>());
      }
    }
  }
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Mask") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("Mask");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.mask = &(arg_var->Get<lite::Tensor>());
      }
    }
  }

  param_.n_layers = op_desc.GetAttr<int>("n_layers");
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.size_per_head = op_desc.GetAttr<int>("size_per_head");
  param_.act_type = op_desc.GetAttr<std::string>("act_type");
  param_.precision = op_desc.GetAttr<std::string>("precision");
  param_.enable_qkv_fusion = op_desc.GetAttr<bool>("enable_qkv_fusion");
  param_.norm_before = op_desc.GetAttr<bool>("norm_before");
  param_.adaptive_seqlen = op_desc.GetAttr<bool>("adaptive_seqlen");

  if (op_desc.HasAttr("slice_axes")) {
    param_.slice_axes = op_desc.GetAttr<std::vector<int>>("slice_axes");
  }
  if (op_desc.HasAttr("slice_starts")) {
    param_.slice_starts = op_desc.GetAttr<std::vector<int>>("slice_starts");
  }
  if (op_desc.HasAttr("slice_ends")) {
    param_.slice_ends = op_desc.GetAttr<std::vector<int>>("slice_ends");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__multi_encoder,
                 paddle::lite::operators::XPUMultiEncoderOp);
