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

#include "lite/core/mir/mlu_postprocess_pass.h"
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {

Node* MLUPostprocessPass::InsertCastBefore(const std::string& op_type,
                                           const std::string& cast_arg_name,
                                           SSAGraph* graph,
                                           Node* cur_node,
                                           Node* inst_node,
                                           const Type* cast_type) {
  // create the arg node
  auto* cast_arg = graph->NewArgumentNode(cast_arg_name);
  cast_arg->AsArg().type = cast_type;
  inst_node->AsStmt().op()->scope()->Var(cast_arg_name);

  // create the stmt node
  auto* cast_inst = graph->NewInstructNode();
  // create op
  auto cast_op = LiteOpRegistry::Global().Create(op_type);
  CHECK(cast_op) << "create op [" << op_type << "] failed";
  cpp::OpDesc op_desc;
  op_desc.SetType(op_type);
  if (op_type == "cast") {
    op_desc.SetAttr<int>("in_dtype", 5);   // FP32
    op_desc.SetAttr<int>("out_dtype", 4);  // FP16
    op_desc.SetInput("X", {cur_node->AsArg().name});
    op_desc.SetOutput("Out", {cast_arg_name});
  } else if (op_type == "transpose") {
    // NCHW -> NHWC
    op_desc.SetAttr<std::vector<int>>("axis", {0, 2, 3, 1});
    op_desc.SetInput("X", {cur_node->AsArg().name});
    op_desc.SetOutput("Out", {cast_arg_name});
  } else if (op_type == "io_copy") {
    op_desc.SetInput("Input", {cur_node->AsArg().name});
    op_desc.SetOutput("Out", {cast_arg_name});
  } else {
    CHECK(0) << "Unsupport cast type";
  }
  cast_op->Attach(op_desc, inst_node->AsStmt().op()->scope());
  // create kernels
  auto kernels = cast_op->CreateKernels(graph->valid_places());
  std::vector<std::unique_ptr<KernelBase>> selected_kernels;
  bool is_found = false;
  for (auto& kernel : kernels) {
    if (op_type == "cast") {
      const Type* in_arg_ty = kernel->GetInputDeclType("X");
      if (PrecisionCompatibleTo(*in_arg_ty, *cur_node->AsArg().type)) {
        is_found = true;
      }
    } else if (op_type == "transpose") {
      is_found = true;
    } else if (op_type == "io_copy") {
      const Type* in_arg_ty = kernel->GetInputDeclType("Input");
      const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
      if (TargetCompatibleTo(*in_arg_ty, *cur_node->AsArg().type) &&
          TargetCompatibleTo(*out_arg_ty, *cast_type)) {
        is_found = true;
      }
    } else {
      CHECK(0) << "Unsupport cast type";
    }
    if (is_found) {
      selected_kernels.emplace_back(std::move(kernel));
      // we pick the kernel
      cast_inst->AsStmt(op_type, std::move(selected_kernels), cast_op);
      auto& stmt = cast_inst->AsStmt();
      stmt.picked_kernel().SetContext(
          ContextScheduler::Global().NewContext(stmt.picked_kernel().target()));
      break;
    }
  }
  CHECK(is_found) << "Can't find a Cast kernel for Cast op: "
                  << cur_node->AsArg().name << "->" << op_type;
  // modify links
  DirectedLink(cur_node, cast_inst);
  DirectedLink(cast_inst, cast_arg);
  return cast_arg;
}

Node* MLUPostprocessPass::InsertCastAfter(const std::string& op_type,
                                          const std::string& cast_arg_name,
                                          SSAGraph* graph,
                                          Node* cur_node,
                                          Node* inst_node,
                                          const Type* cast_type) {
  // create the arg node
  auto* cast_arg = graph->NewArgumentNode(cast_arg_name);
  cast_arg->AsArg().type = cast_type;
  auto* var = inst_node->AsStmt().op()->scope()->Var(cast_arg_name);
  // for CastAfter manully set the tensor's type
  var->GetMutable<::paddle::lite::Tensor>();

  // create the stmt node
  auto* cast_inst = graph->NewInstructNode();
  // create op
  auto cast_op = LiteOpRegistry::Global().Create(op_type);
  CHECK(cast_op) << "create op [" << op_type << "] failed";
  cpp::OpDesc op_desc;
  op_desc.SetType(op_type);
  if (op_type == "cast") {
    op_desc.SetAttr<int>("in_dtype", 4);   // FP32
    op_desc.SetAttr<int>("out_dtype", 5);  // FP16
    op_desc.SetInput("X", {cast_arg_name});
    op_desc.SetOutput("Out", {cur_node->AsArg().name});
  } else if (op_type == "transpose") {
    // NHWC -> NCHW
    op_desc.SetAttr<std::vector<int>>("axis", {0, 3, 1, 2});
    op_desc.SetInput("X", {cast_arg_name});
    op_desc.SetOutput("Out", {cur_node->AsArg().name});
  } else if (op_type == "io_copy") {
    op_desc.SetInput("Input", {cast_arg_name});
    op_desc.SetOutput("Out", {cur_node->AsArg().name});
  } else {
    CHECK(0) << "Unsupport cast type";
  }

  cast_op->Attach(op_desc, inst_node->AsStmt().op()->scope());

  // create kernels
  auto kernels = cast_op->CreateKernels(graph->valid_places());
  std::vector<std::unique_ptr<KernelBase>> selected_kernels;
  bool is_found = false;
  for (auto& kernel : kernels) {
    if (op_type == "cast") {
      const Type* in_arg_ty = kernel->GetInputDeclType("X");
      if (PrecisionCompatibleTo(*in_arg_ty, *cast_type)) {
        is_found = true;
      }
    } else if (op_type == "transpose") {
      is_found = true;
    } else if (op_type == "io_copy") {
      const Type* in_arg_ty = kernel->GetInputDeclType("Input");
      const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
      if (TargetCompatibleTo(*in_arg_ty, *cast_type) &&
          TargetCompatibleTo(*out_arg_ty, *cur_node->AsArg().type)) {
        is_found = true;
      }
    } else {
      CHECK(0) << "Unsupport cast type";
    }
    if (is_found) {
      selected_kernels.emplace_back(std::move(kernel));
      // we pick the kernel
      cast_inst->AsStmt(op_type, std::move(selected_kernels), cast_op);
      auto& stmt = cast_inst->AsStmt();
      stmt.picked_kernel().SetContext(
          ContextScheduler::Global().NewContext(stmt.picked_kernel().target()));
      break;
    }
  }
  CHECK(is_found) << "Can't find a Cast kernel for Cast op: "
                  << cur_node->AsArg().name << "->" << op_type;
  // modify links
  DirectedLink(cast_arg, cast_inst);
  DirectedLink(cast_inst, cur_node);
  return cast_arg;
}

void MLUPostprocessPass::InsertBefore(SSAGraph* graph,
                                      Node* head_node,
                                      Node* inst_node,
                                      const Type* inst_type) {
  const auto* head_type = head_node->AsArg().type;

  // break original link
  RemoveDirectedLink(head_node, inst_node);

  auto* cur_node = head_node;
  const auto name_prefix =
      head_node->AsArg().name + string_format("_%p", inst_node) + "/trans_";
  bool is_first_conv_head =
      std::find(first_conv_nodes_.begin(),
                first_conv_nodes_.end(),
                head_node->AsArg().name) != first_conv_nodes_.end();

  // layout cast node
  if (head_type->layout() != inst_type->layout()) {
    cur_node = InsertCastBefore(
        "transpose",
        name_prefix + "transpose",
        graph,
        cur_node,
        inst_node,
        LiteType::GetTensorTy(
            head_type->target(), head_type->precision(), inst_type->layout()));
  }

  // precision cast node
  if (head_type->precision() != inst_type->precision() && !is_first_conv_head) {
    cur_node = InsertCastBefore(
        "cast",
        name_prefix + "cast",
        graph,
        cur_node,
        inst_node,
        LiteType::GetTensorTy(
            head_type->target(), inst_type->precision(), inst_type->layout()));
  }

  // io copy
  cur_node = InsertCastBefore(
      "io_copy",
      name_prefix + "io_copy",
      graph,
      cur_node,
      inst_node,
      LiteType::GetTensorTy(
          inst_type->target(), inst_type->precision(), inst_type->layout()));

  // connect cur_node to inst_node
  DirectedLink(cur_node, inst_node);

  // reset opdesc and update kernel information
  UpdateInputTo(inst_node->AsStmt().op()->mutable_op_info(),
                head_node->AsArg().name,
                cur_node->AsArg().name);
  // for subgraph op, modify the BlockDesc
  auto* sub_block_desc = dynamic_cast<paddle::lite::operators::SubgraphOp*>(
                             inst_node->AsStmt().op().get())
                             ->GetSubBlock();
  for (size_t i = 0; i < sub_block_desc->OpsSize(); ++i) {
    auto* sub_block_op_desc = sub_block_desc->GetOp<cpp::OpDesc>(i);
    UpdateInputTo(
        sub_block_op_desc, head_node->AsArg().name, cur_node->AsArg().name);
  }

  // recreate the op
  RecreateOp(inst_node, graph);

  graph->CheckValid();
}

void MLUPostprocessPass::GetSubgraphOpArgType(Node* inst_node,
                                              const Type** arg_type,
                                              SSAGraph* graph) {
  CHECK(inst_node->IsStmt());
  constexpr auto subgraph_target = TARGET(kMLU);
  constexpr auto subgraph_layout = DATALAYOUT(kNHWC);

  // get subgraph's valid precision
  const auto& places = graph->valid_places();
  std::set<::paddle::lite_api::PrecisionType> prec_set;
  for (const auto& place : places) {
    if (place.target == TARGET(kMLU)) {
      prec_set.insert(place.precision);
    }
  }

  // get subgraph op's type info
  size_t kernel_size = inst_node->AsStmt().kernels().size();
  CHECK_GT(kernel_size, 0);
  VLOG(4) << "subgraph kernel size: " << kernel_size;

  for (size_t i = 0; i < kernel_size; ++i) {
    auto* kernel = inst_node->AsStmt().kernels()[i].get();
    VLOG(4) << i << "th kernel: " << TargetToStr(kernel->target()) << ", "
            << PrecisionToStr(kernel->precision()) << ", "
            << DataLayoutToStr(kernel->layout());
  }

  for (size_t i = 0; i < kernel_size; ++i) {
    auto* kernel = inst_node->AsStmt().kernels()[i].get();
    CHECK(kernel->target() == subgraph_target);
    CHECK(kernel->layout() == subgraph_layout);
    if (prec_set.count(kernel->precision()) == 1) {
      const auto subgraph_precision = kernel->precision();
      CHECK(subgraph_precision == PRECISION(kFloat) ||
            subgraph_precision == PRECISION(kFP16))
          << "Mlu node has unsupport precision";
      VLOG(4) << "picked kernel precision: "
              << PrecisionToStr(subgraph_precision);
      *arg_type = LiteType::GetTensorTy(
          subgraph_target, subgraph_precision, subgraph_layout);
      break;
    }
  }
}

bool MLUPostprocessPass::NeedInsert(Node* node, const Type* inst_type) {
  CHECK(node->IsArg());

  // some op, for example batch_norm, has output nodes useless
  if (node->outlinks.size() == 0) {
    return false;
  }

  // check if node is weight or persistent
  bool is_persist = node->AsArg().is_weight || node->AsArg().is_persist;
  if (is_persist) {
    VLOG(4) << "Persistent arg name: " << node->AsArg().name
            << " is_weight: " << node->AsArg().is_weight
            << " is_persist: " << node->AsArg().is_persist;
    return false;
  }

  const auto target = node->AsArg().type->target();
  const auto precision = node->AsArg().type->precision();
  const auto layout = node->AsArg().type->layout();
  VLOG(4) << "arg name: " << node->AsArg().name
          << " type: " << TargetToStr(target) << ", "
          << PrecisionToStr(precision) << ", " << DataLayoutToStr(layout);

  // do not insert nodes if previous node is on mlu already
  if (target == inst_type->target()) {
    CHECK(layout == inst_type->layout()) << "Mlu node has wrong layout";
    return false;
  }

  return true;
}

void MLUPostprocessPass::InsertAfter(SSAGraph* graph,
                                     Node* tail_node,
                                     Node* inst_node,
                                     const Type* inst_type) {
  const auto* tail_type = tail_node->AsArg().type;

  // break original link
  RemoveDirectedLink(inst_node, tail_node);

  auto* cur_node = tail_node;
  const auto name_prefix =
      tail_node->AsArg().name + string_format("_%p", inst_node) + "/trans_";

  // layout cast node
  if (tail_type->layout() != inst_type->layout()) {
    cur_node = InsertCastAfter(
        "transpose",
        name_prefix + "transpose",
        graph,
        cur_node,
        inst_node,
        LiteType::GetTensorTy(
            tail_type->target(), tail_type->precision(), inst_type->layout()));
  }

  // precision cast node
  if (tail_type->precision() != inst_type->precision()) {
    cur_node = InsertCastAfter(
        "cast",
        name_prefix + "cast",
        graph,
        cur_node,
        inst_node,
        LiteType::GetTensorTy(
            tail_type->target(), inst_type->precision(), inst_type->layout()));
  }

  // io copy
  cur_node = InsertCastAfter(
      "io_copy",
      name_prefix + "io_copy",
      graph,
      cur_node,
      inst_node,
      LiteType::GetTensorTy(
          inst_type->target(), inst_type->precision(), inst_type->layout()));

  // connect cur_node to inst_node
  DirectedLink(inst_node, cur_node);

  // reset opdesc and update kernel information
  UpdateOutputTo(inst_node->AsStmt().op()->mutable_op_info(),
                 tail_node->AsArg().name,
                 cur_node->AsArg().name);
  // for subgraph op, modify the BlockDesc
  auto* sub_block_desc = dynamic_cast<paddle::lite::operators::SubgraphOp*>(
                             inst_node->AsStmt().op().get())
                             ->GetSubBlock();
  for (size_t i = 0; i < sub_block_desc->OpsSize(); ++i) {
    auto* sub_block_op_desc = sub_block_desc->GetOp<cpp::OpDesc>(i);
    UpdateOutputTo(
        sub_block_op_desc, tail_node->AsArg().name, cur_node->AsArg().name);
  }

  // recreate the op
  RecreateOp(inst_node, graph);

  graph->CheckValid();
}

void MLUPostprocessPass::RecreateOp(Node* inst_node, SSAGraph* graph) {
  auto original_selected_kernel =
      std::move(inst_node->AsStmt().kernels().front());
  auto updated_op_info = *inst_node->AsStmt().mutable_op_info();

  inst_node->AsStmt().ResetOp(updated_op_info, graph->valid_places());
  inst_node->AsStmt().kernels().clear();
  inst_node->AsStmt().kernels().emplace_back(
      std::move(original_selected_kernel));
  for (auto& kernel : inst_node->AsStmt().kernels()) {
    VLOG(4) << "kernel info: " << kernel->name();
    inst_node->AsStmt().op()->AttachKernel(kernel.get());
  }
}

bool MLUPostprocessPass::IsFirstConvInSubgraph(Node* arg_node, Node* inst) {
  auto* block_desc =
      static_cast<operators::SubgraphOp*>(inst->AsStmt().op().get())
          ->GetSubBlock();
  for (int op_idx = 0; op_idx < block_desc->OpsSize(); op_idx++) {
    auto op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    if (op_desc->Type() == "conv2d") {
      for (auto& names : op_desc->inputs()) {
        if (std::find(names.second.begin(),
                      names.second.end(),
                      arg_node->AsArg().name) != names.second.end()) {
          return true;
        }
      }
    }
  }
  return false;
}

bool MLUPostprocessPass::IsFirstConvNode(Node* arg_node) {
  CHECK(arg_node->IsArg());
  for (auto& inst : arg_node->outlinks) {
    if (inst->AsStmt().op_type() == "subgraph") {
      return IsFirstConvInSubgraph(arg_node, inst);
    }
  }
  return false;
}

void MLUPostprocessPass::GatherFirstConvNodes(SSAGraph* graph) {
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsStmt()) continue;
    if (node.AsStmt().op_type() == "feed") {
      for (auto& out : node.outlinks) {
        if (IsFirstConvNode(out)) {
          first_conv_nodes_.insert(out->AsArg().name);
        }
      }
    }
  }
}

void MLUPostprocessPass::ModifyLayout(SSAGraph* graph) {
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsStmt()) continue;
    if (node.AsStmt().op_type() == "feed") {
      for (auto& out : node.outlinks) {
        bool change = true;
        for (auto& inst : out->outlinks) {
          if (inst->AsStmt().op_type() != "subgraph") {
            change = false;
            break;
          }
        }
        if (change) {
          const auto* old_type = out->AsArg().type;
          out->AsArg().type =
              LiteType::GetTensorTy(old_type->target(),
                                    old_type->precision(),
                                    ::paddle::lite_api::DataLayoutType::kNHWC,
                                    old_type->device());
        }
      }
    }
    if (node.AsStmt().op_type() == "fetch") {
      for (auto& inp : node.inlinks) {
        bool change = true;
        for (auto& inst : inp->inlinks) {
          if (inst->AsStmt().op_type() != "subgraph") {
            change = false;
            break;
          }
        }
        if (change) {
          const auto* old_type = inp->AsArg().type;
          inp->AsArg().type =
              LiteType::GetTensorTy(old_type->target(),
                                    old_type->precision(),
                                    ::paddle::lite_api::DataLayoutType::kNHWC,
                                    old_type->device());
        }
      }
    }
  }
}

void MLUPostprocessPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // currently for non-persistent input and output args, mlu subgraph op
  // only support float16/float32 data type

  // in two situations as folllows:
  // 1: feed->arg_in->subgraph->... 2: ...->subgraph->arg_out->fetch;
  // arg_in and arg_out are assumed to be NHWC which user should be aware of.
  // Thus here we change these args' layout to NHWC
  ModifyLayout(graph.get());

  if (lite::DeviceInfo::Global().UseFirstConv()) {
    GatherFirstConvNodes(graph.get());
  }

  // insert io_copy, layout and precision cast of subgraph's inputs and outputs
  for (auto& node : graph->mutable_nodes()) {
    if (node.IsStmt() && node.AsStmt().op_type() == "subgraph") {
      const Type* subgraph_arg_type = nullptr;
      GetSubgraphOpArgType(&node, &subgraph_arg_type, graph.get());

      auto links_tmp = node.inlinks;
      for (auto p_in : links_tmp) {
        if (NeedInsert(p_in, subgraph_arg_type)) {
          InsertBefore(graph.get(), p_in, &node, subgraph_arg_type);
        }
      }
      links_tmp.assign(node.outlinks.begin(), node.outlinks.end());
      for (auto p_out : links_tmp) {
        if (NeedInsert(p_out, subgraph_arg_type)) {
          InsertAfter(graph.get(), p_out, &node, subgraph_arg_type);
        }
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(mlu_postprocess_pass, paddle::lite::mir::MLUPostprocessPass)
    .BindTargets({TARGET(kMLU)});
