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

std::shared_ptr<ProgramDesc> ProgramOptimize::FushionOptimize(
    std::shared_ptr<ProgramDesc> ori_des, bool add_split) {

//  ProgramDesc *optimize_program = new ProgramDesc(*ori_des);
  std::shared_ptr<ProgramDesc> optimize_program = std::make_shared<ProgramDesc>(*ori_des);
  current_block_ = optimize_program->Blocks().size();

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
          matcher->FolderNodes(match_node.get());
          //          DLOG << " after match node\n"<< *match_node;
          //          match_node->Description();

          //          DLOG << "begin node: \n" << *begin_node;
        }
      }
    }

    //    DLOG << "node: \n" << *begin_node;


    std::vector<std::shared_ptr<framework::OpDesc>> op_descs;
    GenerateOps(&op_descs, begin_node.get());
    block->ops_ = op_descs;
  }

  for (int m = 0; m < new_blocks_.size(); ++m) {
    std::shared_ptr<BlockDesc> new_block = new_blocks_[m];
    new_block->index_ = m + ori_des->blocks_.size();
    optimize_program->blocks_.push_back(new_block);
  }
  return optimize_program;
}


void ProgramOptimize::GenerateOps(std::vector<std::shared_ptr<framework::OpDesc>> *op_desc,
                                  Node *input_node,
                                  Node *current_node,
                                  bool adding_thread,
                                  int thread_num,
                                  std::shared_ptr<BlockDesc> new_block) {
  if (current_node->outputs_.size() > 1) {
    adding_thread = false;
  }

  bool can_add_split = false;
  // 如果当前节点有多个输出 并且 只有当前节点对应的 op_desc_ 输出数为 1 时支持
  if (current_node->outputs_.size() > 1 &&
      op_input_output_key[current_node->op_desc_->type_].second.size() == 1) {
    can_add_split = true;

    // 遍历当前节点的 output 节点
    for (const auto &output : current_node->outputs_) {
      // 不支持 output 有多个 output 的情况
      if (output->outputs_.size() > 1) {
        DLOG << "don't support multi output of output";
        can_add_split = false;
        break;
      }

      //与节点关联的 OpDesc
      std::shared_ptr<framework::OpDesc> &op_desc = output->op_desc_;

      //获取这个 op 的 inputs key 和 outputs key
      auto inputs_and_outputs = op_input_output_key[op_desc->type_];

      //判断现在 是否存在这个 op
      //判断这个 output 和 input key 的 size 等于 1
      if (op_input_output_key.find(op_desc->type_) !=
          op_input_output_key.end() &&
          inputs_and_outputs.first.size() == 1 &&
          inputs_and_outputs.second.size() == 1) {
        auto inputs_of_output = op_desc->Input(inputs_and_outputs.first[0]);
        auto outputs_of_output = op_desc->Output(inputs_and_outputs.second[0]);

        // 判断一下, 如果输入和输出没有同名, 是支持的
        for (int i = 0; i < inputs_of_output.size(); ++i) {
          std::string input_of_output = inputs_of_output[i];
          for (int j = 0; j < outputs_of_output.size(); ++j) {
            std::string output_of_output = outputs_of_output[j];
            if (input_of_output == output_of_output) {
              DLOG << "output的 output 包含 input" << input_of_output;
              can_add_split = false;
              break;
            }
          }
        }
      } else {  // 如果模型中包含没有的 op, 则不支持添加 split
        DLOG << "找不到 这个 op 类型: " << output->op_desc_->type_;
        can_add_split = false;
      }
    }
  }

  if (current_node->inputs_.size() > 1 && input_node != current_node->inputs_.back()) {
    return;
  } else if (current_node->inputs_.size() > 1 && input_node == current_node->inputs_.back()) {
    new_block.reset();
    adding_thread = false;
    op_desc->push_back(current_node->op_desc_);
  } else {
    if (new_block.get() && adding_thread) {
      new_block->ops_.push_back(current_node->op_desc_);
    } else {
      op_desc->push_back(current_node->op_desc_);
    }
  }
  if (adding_thread) {
    Attribute attr;
    attr.Set<int>(thread_num);
    current_node->op_desc_->attrs_["thread"] = attr;
  }



  if (can_add_split) {
    new_block = std::make_shared<BlockDesc>();
    new_block->multi_thread_ = true;
    new_block->index_ = current_block_;
    new_blocks_.push_back(new_block);

    adding_thread = true;
    std::shared_ptr<OpDesc> split_op_desc =
            std::make_shared<OpDesc>();
    split_op_desc->type_ = G_OP_TYPE_SPLIT;
    auto outputs = current_node->op_desc_->Output(
            op_input_output_key[current_node->op_desc_->Type()].second[0]);
    split_op_desc->inputs_ = {
            {op_input_output_key[G_OP_TYPE_SPLIT].first[0], outputs}};
    auto &split_outputs =
            split_op_desc->outputs_[op_input_output_key[G_OP_TYPE_SPLIT].second[0]];
    for (const auto &output : current_node->outputs_) {
      split_outputs.push_back(outputs[0]);
    }

    Attribute attr;
    attr.Set<int>(current_block_);
    split_op_desc->attrs_["block_id"] = attr;

    op_desc->push_back(split_op_desc);
    current_block_++;
  }

  for (int i = 0; i < current_node->outputs_.size(); ++i) {
    auto &output = current_node->outputs_[i];
    if (can_add_split) {
      GenerateOps(op_desc, current_node, output.get(), adding_thread, i, new_block);
    } else {
      GenerateOps(op_desc, current_node, output.get(), adding_thread, thread_num, new_block);
    }
  }
}

void ProgramOptimize::GenerateOps(std::vector<std::shared_ptr<framework::OpDesc>> *op_descs,
                                  Node *begin_node) {


  //std::vector<std::shared_ptr<framework::OpDesc>> *op_desc,
  //             Node *input_node, Node *current_node, bool adding_thread, int thread_num
  this->GenerateOps(op_descs, begin_node, begin_node, false, -1, nullptr);
}

}  // namespace framework
}  // namespace paddle_mobile
