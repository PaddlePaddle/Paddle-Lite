  if (op_type == "cast") {
    op_desc.SetAttr<int>("in_dtype", 5);   // FP32
    op_desc.SetAttr<int>("out_dtype", 4);  // FP16
    op_desc.SetInput("X", {cur_node->AsArg().name});
    op_desc.SetOutput("Out", {cast_arg_name});
  }

    if (op_type == "cast") {
      const Type* in_arg_ty = kernel->GetInputDeclType("X");
      if (PrecisionCompatibleTo(*in_arg_ty, *cur_node->AsArg().type) &&
          DataLayoutCompatible(*in_arg_ty, *cur_node->AsArg().type)) {
        is_found = true;
      }
    }

  if (op_type == "cast") {
    op_desc.SetAttr<int>("in_dtype", 4);   // FP16
    op_desc.SetAttr<int>("out_dtype", 5);  // FP32
    op_desc.SetInput("X", {cast_arg_name});
    op_desc.SetOutput("Out", {cur_node->AsArg().name});
  }

    if (op_type == "cast") {
      const Type* in_arg_ty = kernel->GetInputDeclType("X");
      if (PrecisionCompatibleTo(*in_arg_ty, *cast_type)) {
        is_found = true;
      }
    }

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

bool MLUPostprocessPass::IsFirstConvInSubgraph(Node* arg_node,
                                               Node* inst_node) {
  auto sub_program_desc = static_cast<paddle::lite::operators::SubgraphOp*>(
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


// insert cast op on mlu, to avoid cast on cpu
void MLUPostprocessPass::AdjustSubgraph(Node* subgraph_node,
                                        const Type* subgraph_type) {
  CHECK_EQ(subgraph_node->AsStmt().op()->Type(), "subgraph");
  auto subgraph_op =
      static_cast<operators::SubgraphOp*>(subgraph_node->AsStmt().op().get());
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

  if (use_mlu_cast) {
    // insert mlu float place for float io copy, no effect to subgraph type
    v_places.emplace_back(TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC));
  }


  if (lite::TargetWrapperMlu::UseFirstConv()) {
    GatherAndModifyFirstConvNodes(graph.get());
  }
