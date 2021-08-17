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

#include <memory>
#include <set>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/xpu_pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

namespace fusion {

class XPUGraphDedup {
 public:
  template <typename T>
  bool VectorIdentical(std::vector<T>& vec0, std::vector<T>& vec1) {  // NOLINT
    if (vec0.size() != vec1.size()) {
      return false;
    }
    std::sort(vec0.begin(), vec0.end());
    std::sort(vec1.begin(), vec1.end());
    return vec0 == vec1;
  }

  template <typename T>
  bool VectorStrictIdentical(std::vector<T>& vec0,    // NOLINT
                             std::vector<T>& vec1) {  // NOLINT
    return vec0 == vec1;
  }

  bool NodeIdentical(const mir::Node& node0, const mir::Node& node1) {
    CHECK(node0.IsStmt());
    CHECK(node1.IsStmt());

    auto* op_info0 = node0.stmt()->op_info();
    auto* op_info1 = node1.stmt()->op_info();

    // 1. op type
    // XXX(miaotianxiang): skip |feed| and |fetch|
    if (op_info0->Type() != op_info1->Type() || op_info0->Type() == "feed" ||
        op_info0->Type() == "fetch" || op_info1->Type() == "feed" ||
        op_info1->Type() == "fetch") {
      return false;
    }
    // 2. input
    auto input_argname0 = op_info0->input_argnames();
    auto input_argname1 = op_info1->input_argnames();
    if (!VectorIdentical(input_argname0, input_argname1)) {
      return false;
    }
    for (auto& argname : input_argname0) {
      auto input0 = op_info0->Input(argname);
      auto input1 = op_info1->Input(argname);
      if (!VectorStrictIdentical(input0, input1)) {
        return false;
      }
    }
    // 3. output
    auto output_argname0 = op_info0->output_argnames();
    auto output_argname1 = op_info1->output_argnames();
    if (!VectorIdentical(output_argname0, output_argname1)) {
      return false;
    }

    // 3. attribute
    auto attr_type0 = op_info0->attr_types();
    auto attr_type1 = op_info1->attr_types();
    if (attr_type0 != attr_type1) {
      return false;
    }
    for (auto pair : attr_type0) {
      const std::string& attr_name = pair.first;
      switch (pair.second /* attr_type */) {
#define ATTR_COMPARE(attr_type, cpp_type)         \
  case cpp::OpDesc::AttrType::attr_type:          \
    if (op_info0->GetAttr<cpp_type>(attr_name) != \
        op_info1->GetAttr<cpp_type>(attr_name))   \
      return false;                               \
    break

        ATTR_COMPARE(INT, int32_t);
        ATTR_COMPARE(FLOAT, float);
        ATTR_COMPARE(STRING, std::string);
        ATTR_COMPARE(INTS, std::vector<int32_t>);
        ATTR_COMPARE(FLOATS, std::vector<float>);
        ATTR_COMPARE(STRINGS, std::vector<std::string>);
        ATTR_COMPARE(BOOLEAN, bool);
        ATTR_COMPARE(BLOCK, int16_t);
        ATTR_COMPARE(LONG, int64_t);
        ATTR_COMPARE(LONGS, std::vector<int64_t>);
#undef ATTR_COMPARE

        default:
          return false;
          break;
      }
    }

    VLOG(3) << "XPUGraphDedup Remove [" << op_info1->Type() << "]";
    return true;
  }

  void Dedup(SSAGraph* graph,
             mir::Node& to_keep,      // NOLINT
             mir::Node& to_remove) {  // NOLINT
    CHECK(to_keep.IsStmt());
    CHECK(to_remove.IsStmt());

    std::set<const Node*> remove_set = {&to_remove};
    for (auto& argname : to_keep.stmt()->op_info()->output_argnames()) {
      auto output0 = to_keep.stmt()->op_info()->Output(argname);
      auto output1 = to_remove.stmt()->op_info()->Output(argname);
      CHECK(output0.size() == output1.size());
      for (size_t i = 0; i < output0.size(); ++i) {
        auto& keep_name = output0[i];
        auto& remove_name = output1[i];
        auto* keep_node = graph->RetrieveArgument(keep_name);
        auto* remove_node = graph->RetrieveArgument(remove_name);
        remove_set.insert(remove_node);
        VLOG(3) << "XPUGraphDedup Remove [" << remove_name << "]";
        for (auto* stmt_node : remove_node->outlinks) {
          auto new_op_info = *stmt_node->stmt()->op_info();
          new_op_info.UpdateAllInputs(remove_name, keep_name);
          stmt_node->stmt()->ResetOp(new_op_info, graph->valid_places());
          DirectedLink(keep_node, stmt_node);
        }
      }
    }
    GraphSafeRemoveNodes(graph, remove_set);
  }

  bool FindAndDedup(SSAGraph* graph) {
    for (auto* node : graph->NodeTopologicalOrder()) {
      if (node->IsStmt()) continue;
      CHECK(node->IsArg());

      auto& arg_outlinks = node->outlinks;
      for (auto it0 = arg_outlinks.begin(); it0 != arg_outlinks.end(); ++it0) {
        auto it1 = it0;
        for (++it1; it1 != arg_outlinks.end(); ++it1) {
          if (NodeIdentical(**it0, **it1)) {
            Dedup(graph, **it0, **it1);
            return true;
          }
        }
      }
    }
    return false;
  }

  void operator()(SSAGraph* graph) {
    while (FindAndDedup(graph)) {
      graph->CheckValid();
    }
  }
};

}  // namespace fusion

class XPUGraphDedupPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    fusion::XPUGraphDedup()(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__graph_dedup_pass, paddle::lite::mir::XPUGraphDedupPass)
    .BindTargets({TARGET(kXPU)});
