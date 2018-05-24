/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "framework/program/program-optimize/program_optimize.h"
#include "framework/program/program-optimize/fusion_op_register.h"

namespace paddle_mobile {

namespace framework {

std::shared_ptr<ProgramDesc> ProgramOptimize::Optimize() {}

std::shared_ptr<ProgramDesc> ProgramOptimize::FushionOptimize(
    std::shared_ptr<ProgramDesc> ori_des) {
  ProgramDesc *optimize_program = new ProgramDesc(*ori_des);

  for (int i = 0; i < optimize_program->Blocks().size(); ++i) {
    std::unordered_map<std::string, std::shared_ptr<Node>> output_nodes;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Node>>>
        type_map;

    std::shared_ptr<Node> begin_node;
    auto block = optimize_program->Block(i);
    //        DLOG << " ops size: " << block->Ops().size();
    for (int j = 0; j < block->Ops().size(); ++j) {
      auto op = block->Ops()[j];
      auto op_type = op->Type();
      if (op_input_output_key.find(op->Type()) == op_input_output_key.end()) {
        LOG(kLOG_ERROR) << "return null ";
        return nullptr;
      }

      std::shared_ptr<Node> node = std::make_shared<Node>(op);

      //
      type_map[op->Type()].push_back(node);

      if (j == 0) {
        begin_node = node;
      }

      auto input_keys = op_input_output_key.at(op->Type()).first;
      for (auto input_key : input_keys) {
        auto op_inputs = op->Input(input_key);
        for (int l = 0; l < op_inputs.size(); ++l) {
          std::string input_key = op_inputs[l];
          if (output_nodes.find(input_key) != output_nodes.end()) {
            auto input_node = output_nodes[input_key];
            *input_node > node;
          }
        }
      }

      auto output_keys = op_input_output_key.at(op_type).second;
      for (auto output_key : output_keys) {
        auto op_outputs = op->Output(output_key);
        for (int k = 0; k < op_outputs.size(); ++k) {
          output_nodes[op_outputs[k]] = node;
        }
      }
    }

    for (auto &registed : FusionOpRegister::Instance()->Matchers()) {
      std::string fusion_type = registed.first;
      std::shared_ptr<FusionOpMatcher> matcher = registed.second;
      //      DLOG << " registed node \n " << matcher->BeginNode();

      auto match_vector = type_map[matcher->BeginType()];

      for (auto &match_node : match_vector) {
        auto depth = matcher->BeginNode().Depth();
        auto sub_node = match_node->To(depth);
        //        DLOG << " sub node: " << *sub_node;
        if (*sub_node == matcher->BeginNode()) {
          //          DLOG << " match success " << " fusion node: \n" <<
          //          matcher->BeginNode() << "\nsub node: \n" << *sub_node;
          //          DLOG << "match node\n"<< *match_node;
          matcher->FolderNodes(*match_node);
          //          DLOG << " after match node\n"<< *match_node;
          //          match_node->Description();

          //          DLOG << "begin node: \n" << *begin_node;
        }
      }
    }

    //    DLOG << "node: \n" << *begin_node;
    block->ops_ = begin_node->OpDescs();
  }
  std::shared_ptr<ProgramDesc> shared_optimzie(optimize_program);
  return shared_optimzie;
}
}  // namespace framework
}  // namespace paddle_mobile
