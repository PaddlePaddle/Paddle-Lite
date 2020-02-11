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

  int id = 0;
  std::set<std::string> exists_args;
  for (auto& node : graph->mutable_nodes()) {
    std::string key;
    if (node.IsArg()) {
      // key = node.AsArg().name + " lane: " + node.AsArg().lane;
      key = string_format("%s, lane=%d",
                          node.AsArg().name.c_str(),
                          static_cast<int>(node.AsArg().lane));
    } else {
      key = string_format("%s%d, sync=%d, stream=%d",
                          node.AsStmt().op_type().c_str(),
                          id++,
                          node.AsStmt().need_sync_, node.AsStmt().stream_id_);
    }
    if (node.IsStmt()) {
      dot.AddNode(key,
                  {Dot::Attr("shape", "box"),
                   Dot::Attr("style", "filled"),
                   Dot::Attr("color", "black"),
                   Dot::Attr("fillcolor", "yellow")});
      for (auto& x : node.inlinks) {
        auto name =
            x->AsArg().name + ", lane: " + std::to_string(x->AsArg().lane);
        if (!exists_args.count(name)) {
          dot.AddNode(name, {});
        }
        dot.AddEdge(name, key, {});
        exists_args.insert(name);
      }
      for (auto& x : node.outlinks) {
        auto name =
            x->AsArg().name + ", lane: " + std::to_string(x->AsArg().lane);
        if (!exists_args.count(name)) {
          dot.AddNode(name, {});
        }
        dot.AddEdge(key, name, {});
        exists_args.insert(name);
      }
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

REGISTER_MIR_PASS(graph_visualze, paddle::lite::mir::GraphVisualizePass)
    .BindTargets({TARGET(kAny)});
