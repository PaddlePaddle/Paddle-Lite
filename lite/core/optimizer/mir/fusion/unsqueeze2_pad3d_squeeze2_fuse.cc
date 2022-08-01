// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

//
// Created by chenyaohuang on 2021/12/17.
//

#include "lite/core/optimizer/mir/fusion/unsqueeze2_pad3d_squeeze2_fuse.h"

#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void Unsqueeze2Pad3dSqueeze2Fuser::BuildPattern() {
  auto paddings_teller = [](const Node* node) -> bool {
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    return paddings.size() == 6 && paddings[4] == 0 && paddings[5] == 0;
  };

  // create input nodes.
  auto* unsqu_input = VarNode("unsqu_input")
                          ->assert_is_op_input(unsqueeze2_type_, "X")
                          ->AsInput();

  // create op nodes
  auto* unsque = OpNode("unsqueeze2", unsqueeze2_type_)
                     ->assert_is_op(unsqueeze2_type_)
                     ->AsIntermediate();
  auto* p3d = OpNode("pad3d", pad3d_type_)
                  ->assert_is_op(pad3d_type_)
                  ->assert_node_satisfied(paddings_teller)
                  ->AsIntermediate();
  auto* sque = OpNode("squeeze2", squeeze2_type_)
                   ->assert_is_op(squeeze2_type_)
                   ->AsIntermediate();

  // create intermediate nodes
  auto* unsqu_shape = VarNode("unsqu_shape")
                          ->assert_is_op_output(unsqueeze2_type_, "XShape")
                          ->AsIntermediate();
  auto* unsque_out = VarNode("unsque_out")
                         ->assert_is_op_output(unsqueeze2_type_, "Out")
                         ->assert_is_op_input(pad3d_type_, "X")
                         ->AsIntermediate();
  auto* pad3d_out = VarNode("pad3d_out")
                        ->assert_is_op_output(pad3d_type_, "Out")
                        ->assert_is_op_input(squeeze2_type_, "X")
                        ->AsIntermediate();
  // create output node
  auto* sque_out = VarNode("sque_out")
                       ->assert_is_op_output(squeeze2_type_, "Out")
                       ->AsOutput();
  auto* sque_shape = VarNode("sque_shape")
                         ->assert_is_op_output(squeeze2_type_, "XShape")
                         ->AsIntermediate();
  std::vector<PMNode*> unsque_output{unsque_out, unsqu_shape};
  std::vector<PMNode*> sque_output{sque_out, sque_shape};
  *unsqu_input >> *unsque >> unsque_output;
  *unsque_out >> *p3d >> *pad3d_out;
  *pad3d_out >> *sque >> sque_output;
}

void Unsqueeze2Pad3dSqueeze2Fuser::InsertNewNode(SSAGraph* graph,
                                                 const key2nodes_t& matched) {
  auto pad3d_instruct = matched.at("pad3d")->stmt();
  auto pad3d_op_desc = pad3d_instruct->mutable_op_info();
  auto padding =
      pad3d_instruct->op_info()->GetAttr<std::vector<int>>("paddings");
  if (padding.size() == 6 && padding[4] == 0 && padding[5] == 0) {
    pad3d_op_desc->mutable_inputs()->clear();
    pad3d_op_desc->mutable_outputs()->clear();
    auto data_format =
        pad3d_instruct->op_info()->GetAttr<std::string>("data_format");
    if (data_format == "NCDHW") {
      pad3d_op_desc->SetAttr("data_format", std::string("NCHW"));
    } else if (data_format == "NDHWC") {
      pad3d_op_desc->SetAttr("data_format", std::string("NHWC"));
    }
    pad3d_op_desc->SetAttr(
        "paddings",
        std::vector<int>{padding[0], padding[1], padding[2], padding[3]});
    pad3d_op_desc->SetType("pad2d");
    pad3d_op_desc->SetInput("X", {matched.at("unsqu_input")->arg()->name});
    pad3d_op_desc->SetOutput("Out", {matched.at("sque_out")->arg()->name});
    {
      std::string attr_name = "value";
      std::string attr_name_ = "pad_value";
      auto attr_type = pad3d_op_desc->GetAttrType(attr_name);
      switch (attr_type) {
        case OpDescAPI::AttrType::INT:
          pad3d_op_desc->SetAttr(attr_name_,
                                 pad3d_op_desc->GetAttr<int>(attr_name));
          break;
        case OpDescAPI::AttrType::FLOAT:
          pad3d_op_desc->SetAttr(attr_name_,
                                 pad3d_op_desc->GetAttr<float>(attr_name));
          break;
        case OpDescAPI::AttrType::BOOLEAN:
          pad3d_op_desc->SetAttr(attr_name_,
                                 pad3d_op_desc->GetAttr<bool>(attr_name));
          break;
        case OpDescAPI::AttrType::STRING:
          pad3d_op_desc->SetAttr(
              attr_name_, pad3d_op_desc->GetAttr<std::string>(attr_name));
          break;
        case OpDescAPI::AttrType::FLOATS: {
          pad3d_op_desc->SetAttr(
              attr_name_,
              pad3d_op_desc->GetAttr<std::vector<float>>(attr_name));
        } break;
        case OpDescAPI::AttrType::INTS: {
          pad3d_op_desc->SetAttr(
              attr_name, pad3d_op_desc->GetAttr<std::vector<int>>(attr_name));
        } break;
        case OpDescAPI::AttrType::STRINGS: {
          pad3d_op_desc->SetAttr(
              attr_name,
              pad3d_op_desc->GetAttr<std::vector<std::string>>(attr_name));
        } break;
        default:
          LOG(FATAL) << ":Unknow type(" << static_cast<int>(attr_type) << ")";
          break;
      }
    }
    cpp::OpDesc* op_desc = pad3d_op_desc;
    auto pad2d_op = LiteOpRegistry::Global().Create("pad2d");
    auto* scope = pad3d_instruct->op()->scope();
    pad2d_op->Attach(*op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(
        pad2d_op, pad3d_instruct->op()->valid_places());
    IR_NODE_LINK_TO(matched.at("unsqu_input"), new_op_node);
    IR_OP_VAR_LINK(new_op_node, matched.at("sque_out"));
  }
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
