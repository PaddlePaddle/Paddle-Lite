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
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/subgraph/subgraph_detector.h"
#include "lite/operators/subgraph_op.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace mir {

static LITE_THREAD_LOCAL int g_stream_id = 0;

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

  VLOG(4) << "insert cast before subgraph";
  VLOG(4) << "curent node type: " << cur_node->AsArg().type->name()
          << " cast to node type: " << cast_type->name();

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
  } else if (op_type == "layout") {
    // NCHW -> NHWC
    op_desc.SetInput("Input", {cur_node->AsArg().name});
    op_desc.SetOutput("Out", {cast_arg_name});
  } else if (op_type == "io_copy") {
    op_desc.SetInput("Input", {cur_node->AsArg().name});
    op_desc.SetOutput("Out", {cast_arg_name});
  } else {
    CHECK(0) << "Unsupport cast type";
  }
  cast_op->Attach(op_desc, inst_node->AsStmt().op()->scope());

  auto v_places = graph->valid_places();
  // create kernels
  auto kernels = cast_op->CreateKernels(v_places);
  std::vector<std::unique_ptr<KernelBase>> selected_kernels;
  bool is_found = false;
  for (auto& kernel : kernels) {
    if (op_type == "cast") {
      const Type* in_arg_ty = kernel->GetInputDeclType("X");
      if (PrecisionCompatibleTo(*in_arg_ty, *cur_node->AsArg().type) &&
          DataLayoutCompatible(*in_arg_ty, *cur_node->AsArg().type)) {
        is_found = true;
      }
    } else if (op_type == "layout") {
      const Type* in_arg_ty = kernel->GetInputDeclType("Input");
      const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
      if (DataLayoutCompatible(*in_arg_ty, *cur_node->AsArg().type) &&
          DataLayoutCompatible(*out_arg_ty, *cast_type) &&
          //  for first conv
          PrecisionCompatibleTo(*in_arg_ty, *cur_node->AsArg().type)) {
        is_found = true;
      }
    } else if (op_type == "io_copy") {
      const Type* in_arg_ty = kernel->GetInputDeclType("Input");
      const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
      if (TargetCompatibleTo(*in_arg_ty, *cur_node->AsArg().type) &&
          TargetCompatibleTo(*out_arg_ty, *cast_type) &&
          PrecisionCompatible(*in_arg_ty, *cur_node->AsArg().type) &&
          PrecisionCompatible(*out_arg_ty, *cast_type)) {
        is_found = true;
      }
    } else {
      CHECK(0) << "Unsupport cast type";
    }
    if (is_found) {
      VLOG(4) << "insert kernel: " << kernel->name();
      selected_kernels.emplace_back(std::move(kernel));
      // we pick the kernel
      cast_inst->AsStmt(op_type, std::move(selected_kernels), cast_op);
      auto& stmt = cast_inst->AsStmt();
      stmt.picked_kernel().SetContext(ContextScheduler::Global().NewContext(
          stmt.picked_kernel().target(), g_stream_id));
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
  var->GetMutable<paddle::lite::Tensor>();
  VLOG(4) << "insert cast after subgraph";
  VLOG(4) << "curent node type: " << cur_node->AsArg().type->name()
          << " cast to node type: " << cast_type->name();

  // create the stmt node
  auto* cast_inst = graph->NewInstructNode();
  // create op
  auto cast_op = LiteOpRegistry::Global().Create(op_type);
  CHECK(cast_op) << "create op [" << op_type << "] failed";
  cpp::OpDesc op_desc;
  op_desc.SetType(op_type);
  if (op_type == "cast") {
    op_desc.SetAttr<int>("in_dtype", 4);   // FP16
    op_desc.SetAttr<int>("out_dtype", 5);  // FP32
    op_desc.SetInput("X", {cast_arg_name});
    op_desc.SetOutput("Out", {cur_node->AsArg().name});
  } else if (op_type == "layout") {
    // NHWC -> NCHW
    op_desc.SetInput("Input", {cast_arg_name});
    op_desc.SetOutput("Out", {cur_node->AsArg().name});
  } else if (op_type == "io_copy") {
    op_desc.SetInput("Input", {cast_arg_name});
    op_desc.SetOutput("Out", {cur_node->AsArg().name});
  } else {
    CHECK(0) << "Unsupport cast type";
  }

  cast_op->Attach(op_desc, inst_node->AsStmt().op()->scope());

  auto v_places = graph->valid_places();
  // create kernels
  auto kernels = cast_op->CreateKernels(v_places);
  std::vector<std::unique_ptr<KernelBase>> selected_kernels;
  bool is_found = false;
  for (auto& kernel : kernels) {
    if (op_type == "cast") {
      const Type* in_arg_ty = kernel->GetInputDeclType("X");
      if (PrecisionCompatibleTo(*in_arg_ty, *cast_type)) {
        is_found = true;
      }
    } else if (op_type == "layout") {
      const Type* in_arg_ty = kernel->GetInputDeclType("Input");
      const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
      if (DataLayoutCompatible(*in_arg_ty, *cast_type) &&
          DataLayoutCompatible(*out_arg_ty, *cur_node->AsArg().type) &&
          PrecisionCompatibleTo(*in_arg_ty, *cast_type)) {
        is_found = true;
      }
    } else if (op_type == "io_copy") {
      const Type* in_arg_ty = kernel->GetInputDeclType("Input");
      const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
      if (TargetCompatibleTo(*in_arg_ty, *cast_type) &&
          TargetCompatibleTo(*out_arg_ty, *cur_node->AsArg().type) &&
          PrecisionCompatible(*in_arg_ty, *cur_node->AsArg().type) &&
          PrecisionCompatible(*out_arg_ty, *cast_type)) {
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
      stmt.picked_kernel().SetContext(ContextScheduler::Global().NewContext(
          stmt.picked_kernel().target(), g_stream_id));
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
                                      const Type* inst_type,
                                      bool use_mlu_cast) {
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

  // precision cast node
  if (!use_mlu_cast) {
    if (head_type->precision() != inst_type->precision() &&
        !is_first_conv_head) {
      cur_node = InsertCastBefore("cast",
                                  name_prefix + "cast",
                                  graph,
                                  cur_node,
                                  inst_node,
                                  LiteType::GetTensorTy(head_type->target(),
                                                        inst_type->precision(),
                                                        head_type->layout()));
    }

    // layout cast node
    if (head_type->layout() != inst_type->layout()) {
      cur_node = InsertCastBefore("layout",
                                  name_prefix + "layout",
                                  graph,
                                  cur_node,
                                  inst_node,
                                  LiteType::GetTensorTy(head_type->target(),
                                                        inst_type->precision(),
                                                        inst_type->layout()));
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
  } else {
    // io copy
    cur_node = InsertCastBefore(
        "io_copy",
        name_prefix + "io_copy",
        graph,
        cur_node,
        inst_node,
        LiteType::GetTensorTy(
            inst_type->target(), head_type->precision(), head_type->layout()));
  }

  // connect cur_node to inst_node
  DirectedLink(cur_node, inst_node);

  // reset opdesc and update kernel information
  UpdateInputTo(inst_node->AsStmt().op()->mutable_op_info(),
                head_node->AsArg().name,
                cur_node->AsArg().name);
  // for subgraph op, modify the BlockDesc
  auto sub_program_desc = dynamic_cast<paddle::lite::operators::SubgraphOp*>(
                              inst_node->AsStmt().op().get())
                              ->GetProgramDesc();
  CHECK(sub_program_desc);
  int sub_block_idx =
      inst_node->AsStmt().op()->op_info()->GetAttr<int32_t>("sub_block");
  auto* sub_block_desc =
      sub_program_desc->GetBlock<cpp::BlockDesc>(sub_block_idx);
  for (size_t sub_op_idx = 0; sub_op_idx < sub_block_desc->OpsSize();
       ++sub_op_idx) {
    auto* sub_op_desc = const_cast<cpp::OpDesc*>(
        sub_block_desc->GetOp<cpp::OpDesc>(sub_op_idx));
    UpdateInputTo(sub_op_desc, head_node->AsArg().name, cur_node->AsArg().name);
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
  std::set<paddle::lite_api::PrecisionType> prec_set;
  for (const auto& place : places) {
    if (place.target == TARGET(kMLU)) {
      prec_set.insert(place.precision);
    }
  }

  // get subgraph op's type info
  size_t kernel_size = inst_node->AsStmt().kernels().size();
  CHECK_GT(kernel_size, 0u);
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
      *arg_type = LiteType::GetTensorTy(
          subgraph_target, subgraph_precision, subgraph_layout);
      VLOG(4) << "picked subgraph kernel type: " << (*arg_type)->name();
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
                                     const Type* inst_type,
                                     bool use_mlu_cast) {
  const auto* tail_type = tail_node->AsArg().type;

  // break original link
  RemoveDirectedLink(inst_node, tail_node);

  auto* cur_node = tail_node;
  const auto name_prefix =
      tail_node->AsArg().name + string_format("_%p", inst_node) + "/trans_";

  // precision cast node
  if (!use_mlu_cast) {
    if (tail_type->precision() != inst_type->precision()) {
      cur_node = InsertCastAfter("cast",
                                 name_prefix + "cast",
                                 graph,
                                 cur_node,
                                 inst_node,
                                 LiteType::GetTensorTy(tail_type->target(),
                                                       inst_type->precision(),
                                                       tail_type->layout()));
    }

    // layout cast node
    if (tail_type->layout() != inst_type->layout()) {
      cur_node = InsertCastAfter("layout",
                                 name_prefix + "layout",
                                 graph,
                                 cur_node,
                                 inst_node,
                                 LiteType::GetTensorTy(tail_type->target(),
                                                       inst_type->precision(),
                                                       inst_type->layout()));
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
  } else {
    cur_node = InsertCastAfter(
        "io_copy",
        name_prefix + "io_copy",
        graph,
        cur_node,
        inst_node,
        LiteType::GetTensorTy(
            inst_type->target(), tail_type->precision(), tail_type->layout()));
  }

  // connect cur_node to inst_node
  DirectedLink(inst_node, cur_node);

  // reset opdesc and update kernel information
  UpdateOutputTo(inst_node->AsStmt().op()->mutable_op_info(),
                 tail_node->AsArg().name,
                 cur_node->AsArg().name);
  // for subgraph op, modify the BlockDesc
  auto sub_program_desc = dynamic_cast<paddle::lite::operators::SubgraphOp*>(
                              inst_node->AsStmt().op().get())
                              ->GetProgramDesc();
  CHECK(sub_program_desc);
  int sub_block_idx =
      inst_node->AsStmt().op()->op_info()->GetAttr<int32_t>("sub_block");
  auto* sub_block_desc =
      sub_program_desc->GetBlock<cpp::BlockDesc>(sub_block_idx);
  for (size_t sub_op_idx = 0; sub_op_idx < sub_block_desc->OpsSize();
       ++sub_op_idx) {
    auto* sub_op_desc = const_cast<cpp::OpDesc*>(
        sub_block_desc->GetOp<cpp::OpDesc>(sub_op_idx));
    UpdateOutputTo(
        sub_op_desc, tail_node->AsArg().name, cur_node->AsArg().name);
    /* graph like this
     *        subgraph_op_0
     *          /       \
     *         /         \
     * subgraph_op_1   host_op
     */
    UpdateInputTo(sub_op_desc, tail_node->AsArg().name, cur_node->AsArg().name);
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

bool MLUPostprocessPass::IsFirstConvInSubgraph(Node* arg_node,
                                               Node* inst_node) {
  auto sub_program_desc = dynamic_cast<paddle::lite::operators::SubgraphOp*>(
                              inst_node->AsStmt().op().get())
                              ->GetProgramDesc();
  CHECK(sub_program_desc);
  int sub_block_idx =
      inst_node->AsStmt().op()->op_info()->GetAttr<int32_t>("sub_block");
  auto* sub_block_desc =
      sub_program_desc->GetBlock<cpp::BlockDesc>(sub_block_idx);
  for (size_t sub_op_idx = 0; sub_op_idx < sub_block_desc->OpsSize();
       sub_op_idx++) {
    auto sub_op_desc = sub_block_desc->GetOp<cpp::OpDesc>(sub_op_idx);
    CHECK(sub_op_desc);
    if (sub_op_desc->Type() == "conv2d") {
      for (auto& names : sub_op_desc->inputs()) {
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

void MLUPostprocessPass::GatherAndModifyFirstConvNodes(SSAGraph* graph) {
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsStmt()) continue;
    if (node.AsStmt().op_type() == "feed") {
      for (auto& out : node.outlinks) {
        if (IsFirstConvNode(out)) {
          first_conv_nodes_.insert(out->AsArg().name);
          // modify first conv nodes' type
          const auto* old_type = out->AsArg().type;
          out->AsArg().type =
              LiteType::GetTensorTy(old_type->target(),
                                    paddle::lite_api::PrecisionType::kInt8,
                                    old_type->layout(),
                                    old_type->device());
        }
      }
    }
  }
}

void MLUPostprocessPass::ModifyInputOutputDataType(SSAGraph* graph) {
  for (auto& node : graph->mutable_nodes()) {
    if (node.IsStmt() && node.AsStmt().op_type() == "subgraph") {
      const Type* subgraph_arg_type = nullptr;
      GetSubgraphOpArgType(&node, &subgraph_arg_type, graph);
      for (auto& in_node : node.inlinks) {
        const auto* in_node_type = in_node->AsArg().type;
        VLOG(4) << "MLU subgraph input type: " << in_node->AsArg().name
                << *in_node_type;
        if (in_node->AsArg().is_weight || in_node->AsArg().is_persist) {
          CHECK(in_node_type->target() == TARGET(kHost) &&
                in_node_type->precision() == PRECISION(kAny) &&
                in_node_type->layout() == DATALAYOUT(kNCHW))
              << "MLU subgraph unexpected persistent input type!";
          in_node->AsArg().type = LiteType::GetTensorTy(
              TARGET(kMLU), PRECISION(kAny), DATALAYOUT(kNHWC));
        } else {
          CHECK((in_node_type->target() == TARGET(kHost) ||
                 in_node_type->target() == TARGET(kX86)) &&
                in_node_type->precision() == PRECISION(kFloat) &&
                in_node_type->layout() == DATALAYOUT(kNCHW))
              << "MLU subgraph unexpected common input type!";
        }
      }
      for (auto& out_node : node.outlinks) {
        const auto* out_node_type = out_node->AsArg().type;
        auto& out_arg = out_node->AsArg();
        VLOG(4) << "MLU subgraph output type: " << out_node->AsArg().name
                << *out_node_type;
        if (out_node->AsArg().is_weight || out_node->AsArg().is_persist) {
          CHECK(out_node_type->target() == TARGET(kHost) &&
                out_node_type->precision() == PRECISION(kAny) &&
                out_node_type->layout() == DATALAYOUT(kNCHW))
              << "MLU subgraph unexpected persistent input type!";
          out_node->AsArg().type = LiteType::GetTensorTy(
              TARGET(kMLU), PRECISION(kAny), DATALAYOUT(kNHWC));
        } else if (out_node_type->precision() == PRECISION(kAny) &&
                   out_node->outlinks.empty()) {
          out_arg.is_persist = true;
          out_arg.type = LiteType::GetTensorTy(
              TARGET(kMLU), PRECISION(kAny), DATALAYOUT(kNHWC));
        } else {
          CHECK(out_node_type->precision() == PRECISION(kFloat))
              << "MLU subgraph unexpected common output type!";
          if (out_node->outlinks.empty()) {
            out_arg.type = LiteType::GetTensorTy(TARGET(kHost),
                                                 subgraph_arg_type->precision(),
                                                 DATALAYOUT(kNHWC));
            VLOG(4) << "unused output node type: " << out_arg.name
                    << out_node_type->name();
          } else {
            out_arg.type = LiteType::GetTensorTy(
                TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
            VLOG(4) << "output node type: " << out_arg.name
                    << out_node_type->name();
          }
        }
        const auto target = out_node->AsArg().type->target();
        const auto precision = out_node->AsArg().type->precision();
        const auto layout = out_node->AsArg().type->layout();
        VLOG(4) << "arg name: " << out_node->AsArg().name
                << " type: " << TargetToStr(target) << ", "
                << PrecisionToStr(precision) << ", " << DataLayoutToStr(layout);
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
                                    paddle::lite_api::DataLayoutType::kNHWC,
                                    old_type->device());
          // modify inst feed to NHWC, while set_mlu_input_layout(kNHWC)
          // invoked, to keep consistent with actual data layout
          auto place = node.AsStmt().place();
          place.layout = DATALAYOUT(kNHWC);
          std::vector<Place> valid_places = {place};
          auto updated_op_info = *node.AsStmt().op_info();
          node.AsStmt().ResetOp(updated_op_info, valid_places, nullptr);
          auto kernel = &(node.AsStmt().picked_kernel());
          VLOG(4) << "kernel info: " << kernel->name();
          node.AsStmt().op()->AttachKernel(kernel);
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
                                    paddle::lite_api::DataLayoutType::kNHWC,
                                    old_type->device());
        }
      }
    }
  }
}

std::pair<bool, std::string> CheckInputAndInsert(Scope* scope,
                                                 cpp::BlockDesc* block_desc,
                                                 const std::string& input_name,
                                                 const Type* tensor_type,
                                                 const Type* subgraph_type) {
  auto cur_node = input_name;
  bool do_insert = false;
  if (!DataLayoutCompatible(*tensor_type, *subgraph_type)) {
    auto layout_op = block_desc->AddOp<cpp::OpDesc>();
    auto layout_arg_name = string_format("%s/layout", cur_node.c_str());
    scope->Var(layout_arg_name);
    VLOG(4) << "insert layout for subgraph input, arg tensor name: "
            << layout_arg_name;
    layout_op->SetType("layout");
    layout_op->SetInput("Input", {cur_node});
    layout_op->SetOutput("Out", {layout_arg_name});
    cur_node = layout_arg_name;
    do_insert = true;
  }

  if (!PrecisionCompatible(*tensor_type, *subgraph_type) &&
      tensor_type->precision() != PRECISION(kInt8) &&
      tensor_type->precision() != PRECISION(kInt32)) {
    auto cast_op = block_desc->AddOp<cpp::OpDesc>();
    auto cast_arg_name = string_format("%s/cast", cur_node.c_str());
    scope->Var(cast_arg_name);
    VLOG(4) << "insert cast for subgraph input, arg tensor name: "
            << cast_arg_name;
    cast_op->SetType("cast");
    cast_op->SetAttr<int>("in_dtype", 5);   // FP32
    cast_op->SetAttr<int>("out_dtype", 4);  // FP16
    cast_op->SetInput("X", {cur_node});
    cast_op->SetOutput("Out", {cast_arg_name});
    cur_node = cast_arg_name;
    do_insert = true;
  }

  return std::make_pair(do_insert, cur_node);
}

std::pair<bool, std::string> CheckOutputAndInsert(
    Scope* scope,
    cpp::BlockDesc* block_desc,
    const std::string& output_name,
    const Type* tensor_type,
    const Type* subgraph_type) {
  auto cur_node = output_name;
  bool do_insert = false;
  cpp::OpDesc *layout_op = nullptr, *cast_op = nullptr;
  size_t cast_idx = 0;

  // subgraph -> cast -> layout -> output
  if (!PrecisionCompatible(*tensor_type, *subgraph_type)) {
    cast_op = block_desc->AddOp<cpp::OpDesc>();
    cast_idx = block_desc->OpsSize() - 1;
    CHECK_EQ(cast_op, block_desc->GetOp<cpp::OpDesc>(cast_idx));
    cast_op->SetType("cast");
    cast_op->SetAttr<int>("in_dtype", 4);   // FP16
    cast_op->SetAttr<int>("out_dtype", 5);  // FP32
    do_insert = true;
  }

  if (!DataLayoutCompatible(*tensor_type, *subgraph_type)) {
    auto layout_arg_name = string_format("%s/layout", cur_node.c_str());
    scope->Var(layout_arg_name);
    VLOG(4) << "insert layout for subgraph output, arg tensor name: "
            << layout_arg_name;
    layout_op = block_desc->AddOp<cpp::OpDesc>();
    layout_op->SetType("layout");
    layout_op->SetInput("Input", {layout_arg_name});
    layout_op->SetOutput("Out", {cur_node});
    cur_node = layout_arg_name;
    do_insert = true;
  }

  if (cast_op) {
    cast_op = block_desc->GetOp<cpp::OpDesc>(cast_idx);
    auto cast_arg_name = string_format("%s/cast", cur_node.c_str());
    scope->Var(cast_arg_name);
    VLOG(4) << "insert cast for subgraph output, arg tensor name: "
            << cast_arg_name;
    cast_op->SetInput("X", {cast_arg_name});
    cast_op->SetOutput("Out", {cur_node});
    cur_node = cast_arg_name;
  }

  return std::make_pair(do_insert, cur_node);
}

// insert cast op on mlu, to avoid cast on cpu
void MLUPostprocessPass::AdjustSubgraph(Node* subgraph_node,
                                        const Type* subgraph_type) {
  CHECK_EQ(subgraph_node->AsStmt().op()->Type(), "subgraph");
  auto subgraph_op =
      dynamic_cast<operators::SubgraphOp*>(subgraph_node->AsStmt().op().get());
  CHECK(subgraph_op);
  auto sub_program_desc = subgraph_op->GetProgramDesc();
  CHECK(sub_program_desc);
  int sub_block_idx = subgraph_op->op_info()->GetAttr<int32_t>("sub_block");
  auto* sub_block_desc = const_cast<cpp::BlockDesc*>(
      sub_program_desc->GetBlock<cpp::BlockDesc>(sub_block_idx));

  // create a new block desc to keep op sequence correct
  cpp::BlockDesc new_block_desc;
  new_block_desc.ClearOps();
  new_block_desc.ClearVars();
  new_block_desc.SetIdx(sub_block_desc->Idx());
  new_block_desc.SetParentIdx(sub_block_desc->ParentIdx());
  new_block_desc.SetForwardBlockIdx(sub_block_desc->ForwardBlockIdx());

  // find all IO that is not weight or persist
  std::list<std::string> i_names, o_names;
  std::map<std::string, std::string> node_replace;

  // Insert cast op for iotensor which is not weight or persist
  for (auto& input : subgraph_node->inlinks) {
    auto input_name = input->AsArg().name;
    if (!(input->AsArg().is_weight || input->AsArg().is_persist)) {
      i_names.emplace_back(input_name);
      auto ret = CheckInputAndInsert(subgraph_op->scope(),
                                     &new_block_desc,
                                     input_name,
                                     input->AsArg().type,
                                     subgraph_type);
      if (ret.first) {
        node_replace[input_name] = ret.second;
      }
    }
  }
  for (auto& output : subgraph_node->outlinks) {
    auto output_name = output->AsArg().name;
    if (!(output->AsArg().is_weight || output->AsArg().is_persist)) {
      o_names.emplace_back(output_name);
      auto ret = CheckOutputAndInsert(subgraph_op->scope(),
                                      sub_block_desc,
                                      output_name,
                                      output->AsArg().type,
                                      subgraph_type);
      if (ret.first) {
        node_replace[output_name] = ret.second;
      }
    }
  }

  // update input and output
  for (size_t sub_op_idx = 0; sub_op_idx < sub_block_desc->OpsSize();
       ++sub_op_idx) {
    auto sub_op_desc = sub_block_desc->GetOp<cpp::OpDesc>(sub_op_idx);
    auto new_op_desc = new_block_desc.AddOp<cpp::OpDesc>();
    *new_op_desc = *sub_op_desc;

    if (sub_op_desc->Type() != "layout" && sub_op_desc->Type() != "cast") {
      auto op_input_args = new_op_desc->InputArgumentNames();
      for (auto& input_arg : op_input_args) {
        auto op_input = new_op_desc->Input(input_arg);
        for (auto& it : i_names) {
          auto index = std::find(op_input.begin(), op_input.end(), it);
          if (index != op_input.end() &&
              node_replace.find(it) != node_replace.end()) {
            index = op_input.erase(index);
            op_input.emplace(index, node_replace.at(it));
            VLOG(4) << new_op_desc->Type() << "] change input from " << it
                    << " to " << node_replace.at(it);
          }
        }
        new_op_desc->SetInput(input_arg, op_input);
      }

      auto op_output_args = new_op_desc->OutputArgumentNames();
      for (auto& output_arg : op_output_args) {
        auto op_output = new_op_desc->Output(output_arg);
        for (auto& it : o_names) {
          auto index = std::find(op_output.begin(), op_output.end(), it);
          if (index != op_output.end() &&
              node_replace.find(it) != node_replace.end()) {
            index = op_output.erase(index);
            op_output.emplace(index, node_replace.at(it));
            VLOG(4) << new_op_desc->Type() << "] change output from " << it
                    << " to " << node_replace.at(it);
          }
        }
        new_op_desc->SetOutput(output_arg, op_output);
      }
    }
  }

  *sub_block_desc = new_block_desc;
}

void ModifyValidPlaces(SSAGraph* graph, bool use_mlu_cast) {
  // remove invalid places, since only support X86, host, MLU
  auto v_places = graph->valid_places();
  for (auto it = v_places.begin(); it != v_places.end();) {
    if (it->target != TARGET(kMLU) && it->target != TARGET(kHost) &&
        it->target != TARGET(kX86)) {
      it = v_places.erase(it);
    } else {
      ++it;
    }
  }

  if (use_mlu_cast) {
    // insert mlu float place for float io copy, no effect to subgraph type
    v_places.emplace_back(TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC));
  }

  graph->SetValidPlaces(v_places);
  VLOG(4) << "valid places after modified:";
  for (auto& p : v_places) {
    VLOG(4) << p.DebugString();
  }
}

void MLUPostprocessPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
// currently for non-persistent input and output args, mlu subgraph op
// only support float16/float32 data type

// in two situations as folllows:
// 1: feed->arg_in->subgraph->... 2: ...->subgraph->arg_out->fetch;
// arg_in and arg_out are assumed to be NHWC which user should be aware of.
// Thus here we change these args' layout to NHWC
#ifdef LITE_WITH_MLU
  ModifyInputOutputDataType(graph.get());

  if (lite::TargetWrapperMlu::InputLayout() == DATALAYOUT(kNHWC)) {
    ModifyLayout(graph.get());
  }

  if (lite::TargetWrapperMlu::UseFirstConv()) {
    GatherAndModifyFirstConvNodes(graph.get());
  }
#endif

  g_stream_id = static_cast<int>(reinterpret_cast<int64_t>(graph.get()));
  bool disable_mlu_cast = GetBoolFromEnv("LITE_DISABLE_MLU_CAST");
  ModifyValidPlaces(graph.get(), !disable_mlu_cast);
  // insert io_copy, layout and precision cast of subgraph's inputs and outputs
  for (auto& node : graph->mutable_nodes()) {
    if (node.IsStmt() && node.AsStmt().op_type() == "subgraph") {
      const Type* subgraph_arg_type = nullptr;
      GetSubgraphOpArgType(&node, &subgraph_arg_type, graph.get());
      if (!disable_mlu_cast) {
        AdjustSubgraph(&node, subgraph_arg_type);
      }

      auto links_tmp = node.inlinks;
      for (auto p_in : links_tmp) {
        if (NeedInsert(p_in, subgraph_arg_type)) {
          InsertBefore(
              graph.get(), p_in, &node, subgraph_arg_type, !disable_mlu_cast);
        }
      }
      links_tmp.assign(node.outlinks.begin(), node.outlinks.end());
      for (auto p_out : links_tmp) {
        if (NeedInsert(p_out, subgraph_arg_type)) {
          InsertAfter(
              graph.get(), p_out, &node, subgraph_arg_type, !disable_mlu_cast);
        }
      }
    }
  }
  // std::vector<std::vector<Node*>> subgraphs({graph->NodeTopologicalOrder()});
  // SubgraphVisualizer(graph.get(), subgraphs)();
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(mlu_postprocess_pass, paddle::lite::mir::MLUPostprocessPass)
    .BindTargets({TARGET(kMLU)});
