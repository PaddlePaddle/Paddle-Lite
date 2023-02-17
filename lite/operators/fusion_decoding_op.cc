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

#include "lite/operators/fusion_decoding_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

// FusionDecoding
bool FusionDecodingOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Input); // [batch_size * beam_with, seqlen, hidden_dim]
  CHECK_OR_FALSE(param_.Memseqlen); // [batch_size]
  CHECK_OR_FALSE(param_.OutIds);
  CHECK_OR_FALSE(param_.ParentIds);
  CHECK_OR_FALSE(param_.SequenceLength);
  auto input_dims = param_.Input->dims();
  auto input_rank = input_dims.size();
  CHECK_OR_FALSE(input_rank == 3);
  auto memseqlen_dims = param_.Memseqlen->dims();
  auto memseqlen_rank = memseqlen_dims.size();
  CHECK_OR_FALSE(memseqlen_rank == 1);
  CHECK_OR_FALSE(memseqlen_dims[0] == input_dims[0]);
  return true;
}

bool FusionDecodingOp::InferShapeImpl() const {
  const int32_t max_out_len = param_.rel_len ?  param_.max_len + param_.Input->dims()[1] : param_.max_len;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> parent_ids_dims;
  std::vector<int64_t> sequence_length_dims;
  int batch_size;
  if (param_.decoding_strategy == "beam_search") {
    // TODO
    sequence_length_dims = {param_.Input->dims()[0]};
    batch_size = param_.Input->dims()[0] / param_.beam_size;
    output_dims = {max_out_len, batch_size, param_.beam_size};
    parent_ids_dims = output_dims;
  } else if (param_.decoding_strategy == "beam_search_v2") {
    sequence_length_dims = {param_.Input->dims()[0] * 2};
    batch_size = param_.Input->dims()[0] / param_.beam_size;
    output_dims = {max_out_len, batch_size, param_.beam_size * 2};
    parent_ids_dims = output_dims;
  } else if (param_.decoding_strategy == "topk_sampling" ||
             param_.decoding_strategy == "topp_sampling") {
    CHECK(false) << "\"topk_sampling\" or \"topp_sampling\" not supported! "; 
  } else {
    CHECK(false) << "Not supported decoding strategy. ";
  }
  param_.OutIds->Resize(output_dims);
  param_.ParentIds->Resize(parent_ids_dims);
  param_.SequenceLength->Resize(sequence_length_dims);
  return true;
}

bool FusionDecodingOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto input_ids = op_desc.Input("Input").front();
  auto mem_seq_len = op_desc.Input("MemSeqLen").front();
  auto output_ids = op_desc.Output("OutputIds").front();
  auto parent_ids = op_desc.Output("ParentIds").front();
  auto seq_len = op_desc.Output("SequenceLength").front();
  CHECK(scope->FindVar(input_ids));
  CHECK(scope->FindVar(mem_seq_len));
  CHECK(scope->FindVar(output_ids));
  CHECK(scope->FindVar(parent_ids));
  CHECK(scope->FindVar(seq_len));

  param_.Input = GetVar<lite::Tensor>(scope, input_ids);
  param_.Memseqlen = GetVar<lite::Tensor>(scope, mem_seq_len);

  param_.word_embedding = GetVar<lite::Tensor>(scope, op_desc.Input("WordEmbedding").front());

  param_.position_embedding = GetVar<lite::Tensor>(scope, op_desc.Input("PositionEncEmb").front());
  
  param_.self_ln_weight.clear();
  for(auto& name : op_desc.Input("SelfLayernormWeight@VECTOR")) {
    param_.self_ln_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }
  
  param_.self_ln_bias.clear();
  for(auto& name : op_desc.Input("SelfLayernormBias@VECTOR")) {
    param_.self_ln_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.self_q_weight.clear();
  for(auto& name : op_desc.Input("SelfQueryWeight@VECTOR")) {
    param_.self_q_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.self_q_bias.clear();
  for(auto& name : op_desc.Input("SelfQueryBias@VECTOR")) {
    param_.self_q_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.self_k_weight.clear();
  for(auto& name : op_desc.Input("SelfKeyWeight@VECTOR")) {
    param_.self_k_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.self_k_bias.clear();
  for(auto& name : op_desc.Input("SelfKeyBias@VECTOR")) {
    param_.self_k_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.self_v_weight.clear();
  for(auto& name : op_desc.Input("SelfValueWeight@VECTOR")) {
    param_.self_v_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.self_v_bias.clear();
  for(auto& name : op_desc.Input("SelfValueBias@VECTOR")) {
    param_.self_v_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.self_out_weight.clear();
  for(auto& name : op_desc.Input("SelfOutWeight@VECTOR")) {
    param_.self_out_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.self_out_bias.clear();
  for(auto& name : op_desc.Input("SelfOutBias@VECTOR")) {
    param_.self_out_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_ln_weight.clear();
  for(auto& name : op_desc.Input("CrossLayernormWeight@VECTOR")) {
    param_.cross_ln_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }
 
  param_.cross_ln_bias.clear();
  for(auto& name : op_desc.Input("CrossLayernormBias@VECTOR")) {
    param_.cross_ln_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_q_weight.clear();
  for(auto& name : op_desc.Input("CrossQueryWeight@VECTOR")) {
    param_.cross_q_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_q_bias.clear();
  for(auto& name : op_desc.Input("CrossQueryBias@VECTOR")) {
    param_.cross_q_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_k_weight.clear();
  for(auto& name : op_desc.Input("CrossKeyWeight@VECTOR")) {
    param_.cross_k_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_k_bias.clear();
  for(auto& name : op_desc.Input("CrossKeyBias@VECTOR")) {
    param_.cross_k_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_v_weight.clear();
  for(auto& name : op_desc.Input("CrossValueWeight@VECTOR")) {
    param_.cross_v_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_v_bias.clear();
  for(auto& name : op_desc.Input("CrossValueBias@VECTOR")) {
    param_.cross_v_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_out_weight.clear();
  for(auto& name : op_desc.Input("CrossOutWeight@VECTOR")) {
    param_.cross_out_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.cross_out_bias.clear();
  for(auto& name : op_desc.Input("CrossOutBias@VECTOR")) {
    param_.cross_out_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }
  
  param_.ffn_ln_weight.clear();
  for(auto& name : op_desc.Input("FFNLayernormWeight@VECTOR")) {
    param_.ffn_ln_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.ffn_ln_bias.clear();
  for(auto& name : op_desc.Input("FFNLayernormBias@VECTOR")) {
    param_.ffn_ln_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.ffn_inter_weight.clear();
  for(auto& name : op_desc.Input("FFNInterWeight@VECTOR")) {
    param_.ffn_inter_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.ffn_inter_bias.clear();
  for(auto& name : op_desc.Input("FFNInterBias@VECTOR")) {
    param_.ffn_inter_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.ffn_out_weight.clear();
  for(auto& name : op_desc.Input("FFNOutWeight@VECTOR")) {
    param_.ffn_out_weight.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.ffn_out_bias.clear();
  for(auto& name : op_desc.Input("FFNOutBias@VECTOR")) {
    param_.ffn_out_bias.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.decoder_ln_weight = GetVar<lite::Tensor>(scope, op_desc.Input("DecoderLayernormWeight").front());

  param_.decoder_ln_bias = GetVar<lite::Tensor>(scope, op_desc.Input("DecoderLayernormBias").front());
  
  param_.emb_weight = GetVar<lite::Tensor>(scope, op_desc.Input("EmbWeight").front());

  param_.emb_bias = GetVar<lite::Tensor>(scope, op_desc.Input("EmbBias").front());

  param_.OutIds = GetMutableVar<lite::Tensor>(scope, output_ids);
  param_.ParentIds = GetMutableVar<lite::Tensor>(scope, parent_ids);
  param_.SequenceLength = GetMutableVar<lite::Tensor>(scope, seq_len);

  param_.decoding_strategy = op_desc.GetAttr<std::string>("decoding_strategy");
  param_.beam_size = op_desc.GetAttr<int32_t>("beam_size");
  param_.topk = op_desc.GetAttr<int32_t>("topk");
  param_.topp = op_desc.GetAttr<float>("topp");
  param_.n_head = op_desc.GetAttr<int32_t>("n_head");
  param_.size_per_head = op_desc.GetAttr<int32_t>("size_per_head");
  param_.num_layer = op_desc.GetAttr<int32_t>("num_layer");
  param_.bos_id = op_desc.GetAttr<int32_t>("bos_id");
  param_.eos_id = op_desc.GetAttr<int32_t>("eos_id");
  param_.max_len = op_desc.GetAttr<int64_t>("max_len");
  param_.beam_search_diversity_rate = op_desc.GetAttr<float>("beam_search_diversity_rate");
  param_.alpha = op_desc.GetAttr<float>("alpha");
  param_.rel_len = op_desc.GetAttr<bool>("rel_len");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fusion_decoding, paddle::lite::operators::FusionDecodingOp);
