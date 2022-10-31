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
  auto slice_decrease_axis = param_.slice_decrease_axis;
  if (param_.SeqLod && param_.SeqLod->data<int>()) {
    batch_size = param_.SeqLod->numel() - 1;
    seq_len = param_.PadSeqLen->data<int>()[0];
  }
  if ((param_.slice_starts.size() > 0 && param_.slice_starts[0] == 0) &&
      (param_.slice_ends.size() > 0 && param_.slice_ends[0] == 1) &&
      (param_.slice_axes.size() > 0 && param_.slice_axes[0] == 1)) {
    DDim out_dims(std::vector<int64_t>({batch_size, 1, head_num}));
    if (param_.slice_decrease_axis.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < slice_decrease_axis.size(); ++i) {
        CHECK_EQ(out_dims[slice_decrease_axis[i]], 1)
            << "xpu multiencoder with slice decrease dim should be 1";
        out_dims[slice_decrease_axis[i]] = 0;
      }
      for (size_t i = 0; i < out_dims.size(); ++i) {
        if (out_dims[i] != 0) {
          new_out_shape.push_back(out_dims[i]);
        }
      }
      if (new_out_shape.size() == 0) {
        new_out_shape.push_back(1);
      }
      DDim new_dims;
      new_dims.ConstructFrom(new_out_shape);
      out_dims = new_dims;
    }
    if (param_.norm_before) {
      param_.output->Resize({batch_size, 1, head_num});
    } else {
      param_.output->Resize(out_dims);
    }
  } else {
    param_.output->Resize({batch_size, seq_len, head_num});
  }
  return true;
}

bool XPUMultiEncoderOp::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
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
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "PadSeqLen") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("PadSeqLen");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.PadSeqLen = &(arg_var->Get<lite::Tensor>());
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
  param_.hidden_dim = op_desc.GetAttr<int>("hidden_dim");
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.size_per_head = op_desc.GetAttr<int>("size_per_head");
  param_.act_type = op_desc.GetAttr<std::string>("act_type");
  param_.precision = op_desc.GetAttr<std::string>("precision");
  param_.enable_qkv_fusion = op_desc.GetAttr<bool>("enable_qkv_fusion");
  param_.norm_before = op_desc.GetAttr<bool>("norm_before");
  param_.adaptive_seqlen = op_desc.GetAttr<bool>("adaptive_seqlen");
  param_.per_channel = op_desc.GetAttr<bool>("per_channel");
  if ((op_desc.HasAttr("enable_int8") &&
       op_desc.GetAttr<bool>("enable_int8")) ||
      (op_desc.HasAttr("enable_int16") &&
       op_desc.GetAttr<bool>("enable_int16"))) {
    param_.input_max = op_desc.GetAttr<std::vector<float>>("FCInputMax");
  }
  param_.weight_max.clear();
  for (const auto& weight_max_tensor :
       op_desc.GetAttr<std::vector<std::string>>("FCWeightMax")) {
    auto tensor = scope->FindMutableTensor(weight_max_tensor);
    CHECK(tensor != nullptr);
    param_.weight_max.push_back(tensor);
  }
  param_.quant_types =
      op_desc.GetAttr<std::vector<std::string>>("FCQuantTypes");

  if (op_desc.HasAttr("slice_axes")) {
    param_.slice_axes = op_desc.GetAttr<std::vector<int>>("slice_axes");
  }
  if (op_desc.HasAttr("slice_starts")) {
    param_.slice_starts = op_desc.GetAttr<std::vector<int>>("slice_starts");
  }
  if (op_desc.HasAttr("slice_ends")) {
    param_.slice_ends = op_desc.GetAttr<std::vector<int>>("slice_ends");
  }
  if (op_desc.HasAttr("slice_decrease_axis")) {
    param_.slice_decrease_axis =
        op_desc.GetAttr<std::vector<int>>("slice_decrease_axis");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__multi_encoder,
                 paddle::lite::operators::XPUMultiEncoderOp);
