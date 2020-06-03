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

#include "lite/core/mir/int64_to_int32_pass.h"

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

void Int64ToInt32Pass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::list<Node*> nodes;
  for (auto& node : graph->StmtTopologicalOrder()) {
    nodes.push_back(node);
  }

  for (auto& node : nodes) {
    if (!node->IsStmt() || node->AsStmt().op_type() == "while") continue;
    if (!node->IsStmt() || node->AsStmt().op_type() == "feed") continue;
    if (!node->IsStmt() || node->AsStmt().op_type() == "fetch") continue;
    auto inlinks = node->inlinks;
    ChangeInt64ToInt32IfNeeded(node);
  }
}

/*
    some op decide data type beside input or output tensor from op_param:
        3. fillconstant
        4. FillConstantBatchSiz
        5. uniformrandom

    int64 input or output from arm kernels
        1. argmax:
        2. beam_search
        3. gather
        4. lookup_table
        5. read_from_arry
        6. topk
        7. write_to_arry
        8. feed
        9. compare
        10. ctc

    may support int64
        1. cast
        2. concat
*/

void Int64ToInt32Pass::ChangeInt64ToInt32IfNeeded(Node* inst_node) {
  CHECK(inst_node->IsStmt());
  auto& inst = inst_node->AsStmt();
  std::string op_type = inst.op_info()->Type();
  // TODO(zhaoying): support more op
  if (op_type == "cast") {
    auto in_dtype = inst.op_info()->GetAttr<int>("in_dtype");
    auto out_dtype = inst.op_info()->GetAttr<int>("out_dtype");
    VLOG(6) << "in_dtype : " << in_dtype;
    VLOG(6) << "out_dtype : " << out_dtype;
    // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
    // SIZE_T = 19;UINT8 = 20;INT8 = 21;
    cpp::OpDesc* cast_opdesc = const_cast<OpInfo*>(inst.op_info());
    cast_opdesc->SetAttr<int>("out_dtype", 2);
    cast_opdesc->SetAttr<int>("in_dtype", 2);
  }
  if (op_type == "fill_constant") {
    CHECK(0) << "int64_to_int32 pass do not expect fill_constant op for now";
  } else if (op_type == "uniform_random") {
    CHECK(0) << "int64_to_int32 pass do not expect uniform_random op for now";
    //   auto dtype = opdesc.GetAttr<int>("dtype");
    //   if (dtype == static_cast<int32_t>(lite::core::FluidType::INT64)) {
    //     opdesc.SetAttr<int>("dtype",static_cast<int32_t>(lite::core::FluidType::INT32);
    //   }
  } else if (op_type == "fill_constant_batch_size_like") {
    CHECK(0) << "int64_to_int32 pass do not expect "
                "fill_constant_batch_size_like op for now";
  }
  for (auto* in : inst_node->inlinks) {
    CHECK(in->IsRoleSet());
    CHECK(in->IsArg());
    CHECK(in->AsArg().type);
    auto in_arg_name = in->AsArg().name;
    std::string tmp;
    CHECK(inst.op_info()->GetInputArgname(in_arg_name, &tmp));
    auto rt_precision = in->AsArg().type->precision();
    // ================== DEBUG INFO ===================
    VLOG(6) << "op :" << op_type;
    VLOG(6) << "arg name :" << in_arg_name;
    VLOG(6) << "arg :" << tmp;
    VLOG(6) << "runtime precision :" << PrecisionToStr(rt_precision);
    // ================== DEBUG END  ===================

    if (rt_precision == PRECISION(kInt64)) {
      VLOG(6) << "change precison from int64 to int32";
      in->AsArg().type =
          const_cast<Type*>(Type::GetTensorTy(in->AsArg().type->target(),
                                              PRECISION(kInt32),
                                              in->AsArg().type->layout()));
    }
  }

  for (auto* out : inst_node->outlinks) {
    CHECK(out->IsRoleSet());
    CHECK(out->IsArg());
    CHECK(out->AsArg().type);
    auto out_arg_name = out->AsArg().name;
    std::string tmp;
    CHECK(inst.op_info()->GetOutputArgname(out_arg_name, &tmp));
    auto rt_precision = out->AsArg().type->precision();
    // ================== DEBUG INFO ===================
    VLOG(6) << "op :" << op_type;
    VLOG(6) << "arg name :" << out_arg_name;
    VLOG(6) << "arg :" << tmp;
    VLOG(6) << "runtime precision :" << PrecisionToStr(rt_precision);
    // ================== DEBUG END  ===================
    if (rt_precision == PRECISION(kInt64)) {
      VLOG(6) << "change precison from int64 to int32";
      out->AsArg().type =
          const_cast<Type*>(Type::GetTensorTy(out->AsArg().type->target(),
                                              PRECISION(kInt32),
                                              out->AsArg().type->layout()));
    }
  }
}
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(int64_to_int32_pass, paddle::lite::mir::Int64ToInt32Pass)
    .BindTargets({TARGET(kMLU)});
