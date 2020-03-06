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

#include "lite/core/mir/graph_visualize_pass.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include "lite/core/mir/pass_registry.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

using inference::analysis::Dot;

void GraphVisualizePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  Visualize(graph.get());
}

std::string Visualize(mir::SSAGraph* graph) {
  inference::analysis::Dot dot;
  int op_idx = 0;
  std::set<std::string> exists_var_names;
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt()) continue;
    auto op_info = node->AsStmt().op_info();
    auto op_type = op_info->Type();
    std::string op_unique_name =
        string_format("%s%d", op_type.c_str(), op_idx++);
    dot.AddNode(op_unique_name,
                {Dot::Attr("shape", "box"),
                 Dot::Attr("style", "filled"),
                 Dot::Attr("color", "black"),
                 Dot::Attr("fillcolor", "yellow")});
    for (auto& x : node->inlinks) {
      auto var_name = x->AsArg().name;
      if (!exists_var_names.count(var_name)) {
        dot.AddNode(var_name, {});
        exists_var_names.insert(var_name);
      }
      dot.AddEdge(var_name, op_unique_name, {});
    }
    for (auto& x : node->outlinks) {
      auto var_name = x->AsArg().name;
      if (!exists_var_names.count(var_name)) {
        dot.AddNode(var_name, {});
        exists_var_names.insert(var_name);
      }
      dot.AddEdge(op_unique_name, var_name, {});
    }
    // Display all of the attributes of the Op
    VLOG(3) << "* " << op_unique_name;
    const auto& attr_names = op_info->AttrNames();
    for (auto& attr_name : attr_names) {
      using AttrType = cpp::OpDesc::AttrType;
      auto attr_type = op_info->GetAttrType(attr_name);
      std::ostringstream os;
      os << "- " << attr_name;
      switch (attr_type) {
        case AttrType::INT:
          os << ":int:" << std::to_string(op_info->GetAttr<int>(attr_name));
          break;
        case AttrType::FLOAT:
          os << ":float:" << std::to_string(op_info->GetAttr<float>(attr_name));
          break;
        case AttrType::BOOLEAN:
          os << ":int:" << std::to_string(op_info->GetAttr<bool>(attr_name));
          break;
        case AttrType::STRING:
          os << ":string: \"" << op_info->GetAttr<std::string>(attr_name)
             << "\"";
          break;
        case AttrType::FLOATS: {
          auto vals = op_info->GetAttr<std::vector<float>>(attr_name);
          os << ":floats: {" + Join(vals, ",") << "}";
        } break;
        case AttrType::INTS: {
          auto vals = op_info->GetAttr<std::vector<int>>(attr_name);
          os << ":ints: {" + Join(vals, ",") + "}";
        } break;
        case AttrType::STRINGS: {
          auto vals = op_info->GetAttr<std::vector<std::string>>(attr_name);
          os << ":strings: {" + Join(vals, ",") << "}";
        } break;
        default:
          os << ":Unsupported attribute type(" << static_cast<int>(attr_type)
             << ")";
          break;
      }
      VLOG(3) << os.str();
    }
  }

  auto res = dot.Build();
  // If we use VLOG here, we can not type all graph out.
  // So we change VLOG to std::cout.
  std::cout << "dot:\n" << res << std::endl;
  return res;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(graph_visualize_pass, paddle::lite::mir::GraphVisualizePass)
    .BindTargets({TARGET(kAny)});
