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

#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/core/mir/pass.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace mir {

static void UpdateInputTo(cpp::OpDesc* desc,
                          const std::string& from,
                          const std::string& to) {
  for (auto& item : *desc->mutable_inputs()) {
    for (auto& input : item.second) {
      if (input == from) {
        input = to;
      }
    }
  }
  if (desc->Type() != "subgraph") return;
  auto input_names =
      desc->GetAttr<std::vector<std::string>>("input_data_names");
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (input_names[i] == from) {
      input_names[i] = to;
    }
  }
  desc->SetAttr<std::vector<std::string>>("input_data_names", input_names);
}

static void UpdateOutputTo(cpp::OpDesc* desc,
                           const std::string& from,
                           const std::string& to) {
  for (auto& item : *desc->mutable_outputs()) {
    for (auto& output : item.second) {
      if (output == from) {
        output = to;
      }
    }
  }
  if (desc->Type() != "subgraph") return;
  auto output_names =
      desc->GetAttr<std::vector<std::string>>("output_data_names");
  for (size_t i = 0; i < output_names.size(); ++i) {
    if (output_names[i] == from) {
      output_names[i] = to;
    }
  }
  desc->SetAttr<std::vector<std::string>>("output_data_names", output_names);
}

/*
 * The pass changes the node's target to mlu which follows a mlu subgraph op
 * */
class MLUPostprocessPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  void GetSubgraphOpArgType(Node* inst_node,
                            const Type** arg_type,
                            SSAGraph* graph);

  void ModifyInputOutputDataType(SSAGraph* graph);

  void ModifyLayout(SSAGraph* graph);

  bool NeedInsert(Node* node, const Type* inst_type);

  void InsertBefore(SSAGraph* graph,
                    Node* head_node,
                    Node* inst_node,
                    const Type* type,
                    bool use_mlu_cast);

  void InsertAfter(SSAGraph* graph,
                   Node* tail_node,
                   Node* inst_node,
                   const Type* type,
                   bool use_mlu_cast);

  Node* InsertCastBefore(const std::string& op_type,
                         const std::string& cast_arg_name,
                         SSAGraph* graph,
                         Node* cur_node,
                         Node* inst_node,
                         const Type* cast_type);

  Node* InsertCastAfter(const std::string& op_type,
                        const std::string& cast_arg_name,
                        SSAGraph* graph,
                        Node* cur_node,
                        Node* inst_node,
                        const Type* cast_type);

  void RecreateOp(Node* inst_node, SSAGraph* graph);

  void GatherAndModifyFirstConvNodes(SSAGraph* graph);

  bool IsFirstConvNode(Node* arg_node);

  bool IsFirstConvInSubgraph(Node* arg_node, Node* inst);

  void AdjustSubgraph(Node* subgraph_node, const Type* op_type);

 private:
  std::set<std::string> first_conv_nodes_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
