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

#include "lite/core/mir/static_kernel_pick_pass.h"
#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

bool KernelScoreCmp(const std::pair<float, std::unique_ptr<KernelBase>>& a,
                    const std::pair<float, std::unique_ptr<KernelBase>>& b) {
  return a.first > b.first;
}

void StaticKernelPickPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  kernel_pick_factors_.ConsiderTarget();
  kernel_pick_factors_.ConsiderPrecision();
  kernel_pick_factors_.ConsiderDataLayout();
  CHECK(kernel_pick_factors_.any_factor_considered())
      << "kernel_pick_factors should be specified first";
  CHECK(graph) << "graph not valid";

  // sort kernels by the factors.
  VLOG(4) << "graph->mutable_nodes().size():" << graph->mutable_nodes().size();
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsStmt()) continue;
    auto& instruct = node.AsStmt();

    std::unordered_map<std::string, PrecisionType> in_types;
    std::unordered_map<std::string, PrecisionType> out_types;
    for (std::list<Node*>::iterator i = node.inlinks.begin();
         i != node.inlinks.end();
         ++i) {
      if ((*i)->arg()->type)
        in_types[(*i)->arg()->name] = (*i)->arg()->type->precision();
    }
    for (std::list<Node*>::iterator i = node.outlinks.begin();
         i != node.outlinks.end();
         ++i) {
      if ((*i)->arg()->type)
        out_types[(*i)->arg()->name] = (*i)->arg()->type->precision();
    }
    // Get candidate kernels
    std::vector<std::pair<float, std::unique_ptr<KernelBase>>> scored;
    CHECK(!instruct.kernels().empty()) << "No kernels found for "
                                       << instruct.op_type();
    VLOG(4) << "instruct.kernels().size():" << instruct.kernels().size();
    for (auto&& kernel : instruct.kernels()) {
      float score = KernelGrade(instruct,
                                *kernel,
                                graph->valid_places(),
                                in_types,
                                out_types,
                                instruct.op_info()->input_names(),
                                instruct.op_info()->output_names());
      VLOG(4) << "kernel->summary():" << kernel->summary()
              << " score:" << score;
      scored.emplace_back(score, std::move(kernel));
    }
    std::sort(scored.begin(), scored.end(), KernelScoreCmp);
    instruct.kernels().clear();

    if (!instruct.op_info()->HasAttr("enable_int8")) {
      // Move kernel back
      // Just keep a single best kernel.
      // TODO(Superjomn) reconsider this.
      instruct.kernels().emplace_back(std::move(scored.front().second));
      VLOG(2) << "pick " << instruct.kernels().front()->name() << "\n\n";

    } else {
      bool out_type_int8 = true;
      // Only if all ops linked to this op output has enable_int8 attr,
      // then the op output type is int8, or fp32.
      for (auto* out_n : node.outlinks) {
        CHECK(out_n->IsArg());
        for (auto* tmp_op : out_n->outlinks) {
          CHECK(tmp_op->IsStmt());
          if (!tmp_op->AsStmt().op_info()->HasAttr("enable_int8")) {
            out_type_int8 = false;
            break;
          }
        }
        if (!out_type_int8) break;
      }
      // If the out_type_int8 is true, it turns out that the output type of this
      // op can be int8.
      // So we need to specify output scale for this op.
      if (out_type_int8) {
        auto out_node = node.outlinks.front();
        CHECK(out_node->IsArg());
        auto one_adj_op_node = out_node->outlinks.front();
        CHECK(one_adj_op_node->IsStmt());
        auto& one_adj_instruct = one_adj_op_node->AsStmt();
        CHECK(one_adj_instruct.op_info()->HasAttr("enable_int8"));
        CHECK(one_adj_instruct.op_info()->HasAttr("input_scale"));

        instruct.mutable_op_info()->SetAttr(
            "output_scale",
            one_adj_instruct.op_info()->GetAttr<float>("input_scale"));

        auto update_desc = *instruct.mutable_op_info();
        instruct.ResetOp(update_desc, graph->valid_places());
        scored.clear();
        for (auto&& kernel : instruct.kernels()) {
          float score = KernelGrade(instruct,
                                    *kernel,
                                    graph->valid_places(),
                                    in_types,
                                    out_types,
                                    instruct.op_info()->input_names(),
                                    instruct.op_info()->output_names());
          scored.emplace_back(score, std::move(kernel));
        }
        std::sort(scored.begin(), scored.end(), KernelScoreCmp);
        instruct.kernels().clear();
      }
      // If the out_type_int8 is true, we should pick the kernel with the
      // int8 input and int8 output.
      // If the out_type_int8 is false, we should pick the kernel with the
      // int8 input and fp32 output.
      auto output_arguments = instruct.op_info()->OutputArgumentNames();
      for (auto& candidate : scored) {
        bool all_output_type_match = true;
        auto expect_output_type =
            out_type_int8 ? PRECISION(kInt8) : PRECISION(kFloat);

        for (auto& arg_name : output_arguments) {
          const Type* out_arg_ty =
              candidate.second->GetOutputDeclType(arg_name);
          if (out_arg_ty->precision() != expect_output_type) {
            all_output_type_match = false;
          }
        }

        if (all_output_type_match) {
          instruct.kernels().emplace_back(std::move(candidate.second));
          VLOG(2) << "instruct.kernels.emplace_back "
                  << instruct.kernels().front()->name();
          break;
        }
      }
      CHECK(!instruct.kernels().empty()) << "No kernels found for "
                                         << instruct.op_type();
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(static_kernel_pick_pass,
                  paddle::lite::mir::StaticKernelPickPass)
    .BindTargets({TARGET(kAny)});
