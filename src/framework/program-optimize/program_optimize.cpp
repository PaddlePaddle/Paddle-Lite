/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "program_optimize.h"

namespace paddle_mobile {

namespace framework {

std::shared_ptr<ProgramDesc> ProgramOptimize::Optimize() {}

std::shared_ptr<ProgramDesc> ProgramOptimize::FushionOptimize(
    std::shared_ptr<ProgramDesc> ori_des) {
  for (int i = 0; i < ori_des->Blocks().size(); ++i) {
    std::unordered_map<std::string, std::shared_ptr<Node>> output_nodes;
    std::shared_ptr<Node> begin_node;
    auto block = ori_des->Block(i);
    //        DLOG << " ops size: " << block->Ops().size();
    for (int j = 0; j < block->Ops().size(); ++j) {
      auto op = block->Ops()[j];
      auto op_type = op->Type();
      //            DLOG << "op type: " << op_type << " index: " << j;
      if (op_input_output_key.find(op->Type()) == op_input_output_key.end()) {
        return NULL;
      }

      std::shared_ptr<Node> node = std::make_shared<Node>(op);
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

    DLOG << "node: \n" << *begin_node;
  }
  return ori_des;
}
}  // namespace framework
}  // namespace paddle_mobile
