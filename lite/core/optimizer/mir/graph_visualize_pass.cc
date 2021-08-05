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

#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

void GraphVisualizePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  VLOG(5) << "\n" << Visualize(graph.get());
}

std::string Visualize(mir::SSAGraph* graph) {
  std::ostringstream os;
  Dot dot;
  auto string_trunc = [](const std::string& str) -> std::string {
    const int max_disp_size = 100;
    if (str.length() > max_disp_size)
      return str.substr(0, max_disp_size) + "...";
    return str;
  };
  auto attr_repr = [&](const OpInfo* op_info,
                       const std::string& attr_name) -> std::string {
    std::ostringstream os;
    using AttrType = cpp::OpDesc::AttrType;
    auto attr_type = op_info->GetAttrType(attr_name);
    switch (attr_type) {
      case AttrType::INT:
        os << ":int:"
           << paddle::lite::to_string(op_info->GetAttr<int>(attr_name));
        break;
      case AttrType::FLOAT:
        os << ":float:"
           << paddle::lite::to_string(op_info->GetAttr<float>(attr_name));
        break;
      case AttrType::BOOLEAN:
        os << ":int:"
           << paddle::lite::to_string(op_info->GetAttr<bool>(attr_name));
        break;
      case AttrType::STRING:
        os << ":string: \""
           << string_trunc(op_info->GetAttr<std::string>(attr_name)) << "\"";
        break;
      case AttrType::FLOATS: {
        std::vector<float> vals =
            op_info->GetAttr<std::vector<float>>(attr_name);
        os << ":floats: {" + Join(vals, ",") << "}";
      } break;
      case AttrType::INTS: {
        std::vector<int> vals = op_info->GetAttr<std::vector<int>>(attr_name);
        os << ":ints: {" + Join(vals, ",") + "}";
      } break;
      case AttrType::STRINGS: {
        std::vector<std::string> vals =
            op_info->GetAttr<std::vector<std::string>>(attr_name);
        os << ":strings: {" + string_trunc(Join(vals, ",")) << "}";
      } break;
      default:
        os << ":Unknow type(" << static_cast<int>(attr_type) << ")";
        break;
    }
    return os.str();
  };
  int op_idx = 0;
  std::set<std::string> exists_var_names;
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt()) continue;
    auto op_info = node->AsStmt().op_info();
    auto op_type = op_info->Type();
    std::string op_name;
    if (node->AsStmt().need_sync_) {
      std::ostringstream oss;
      for (size_t i = 0; i < node->AsStmt().sync_streams_.size(); ++i) {
        oss << std::to_string(node->AsStmt().sync_streams_[i]);
        if (i != node->AsStmt().sync_streams_.size() - 1) {
          oss << ",";
        }
      }
      op_name = string_format("%s%d, stream=%d, sync_streams={%s}",
                              op_type.c_str(),
                              op_idx++,
                              node->AsStmt().stream_id_,
                              oss.str().c_str());
    } else {
      op_name = string_format("%s%d", op_type.c_str(), op_idx++);
    }
    // Add its input&output variables as the Dot nodes
    dot.AddNode(op_name,
                {Dot::Attr("shape", "box"),
                 Dot::Attr("style", "filled"),
                 Dot::Attr("color", "black"),
                 Dot::Attr("fillcolor", "yellow")});
    for (auto& x : node->inlinks) {
      std::string var_name;
      if (x->AsArg().lane != -1) {
        var_name = string_format(
            "%s, lane=%d", x->AsArg().name.c_str(), x->AsArg().lane);
      } else {
        var_name = x->AsArg().name;
      }
      if (!exists_var_names.count(var_name)) {
        dot.AddNode(var_name, {});
        exists_var_names.insert(var_name);
      }
      std::vector<Dot::Attr> attrs;
      std::string arg_name;
      if (op_info->GetInputArgname(var_name, &arg_name)) {
        attrs.emplace_back("label", arg_name);
      } else {
        VLOG(5) << "Can not find the input argument for var " << var_name
                << " in " << op_type;
      }
      dot.AddEdge(var_name, op_name, attrs);
    }
    for (auto& x : node->outlinks) {
      std::string var_name;
      if (x->AsArg().lane != -1) {
        var_name = string_format(
            "%s, lane=%d", x->AsArg().name.c_str(), x->AsArg().lane);
      } else {
        var_name = x->AsArg().name;
      }
      if (!exists_var_names.count(var_name)) {
        dot.AddNode(var_name, {});
        exists_var_names.insert(var_name);
      }
      std::vector<Dot::Attr> attrs;
      std::string arg_name;
      if (op_info->GetOutputArgname(var_name, &arg_name)) {
        attrs.emplace_back("label", arg_name);
      } else {
        VLOG(5) << "Can not find the output argument for var " << var_name
                << " in " << op_type;
      }
      dot.AddEdge(op_name, var_name, attrs);
    }
    // Output its all of attributes(name and values)
    os << "* " << op_name << "\n";
    const auto& attr_names = op_info->AttrNames();
    for (auto& attr_name : attr_names) {
      os << " - " << attr_name << attr_repr(op_info, attr_name) << "\n";
    }
  }
  os << dot.Build();
  return os.str();
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(graph_visualize_pass, paddle::lite::mir::GraphVisualizePass)
    .BindTargets({TARGET(kAny)});
