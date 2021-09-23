// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <set>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "lite/core/framework.pb.h"
#include "lite/core/model/general/program_desc.h"
#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/pb/program_desc.h"
#include "lite/model_parser/ssa/program_desc.h"

#include "lite/core/optimizer/mir/dot.h"
#include "lite/model_parser/model_parser.h"

namespace paddle {
namespace lite {
namespace general {
namespace {

class Op {
 public:
  Op(const std::string& type,
     const std::map<std::string, std::vector<std::string>>& inputs,
     const std::map<std::string, std::vector<std::string>>& outputs,
     std::vector<Op>&& sub_block = {})
      : type_{type},
        inputs_{inputs},
        outputs_{outputs},
        sub_block_{std::move(sub_block)} {}
  const std::string& type() const { return type_; }
  const std::map<std::string, std::vector<std::string>>& inputs() const {
    return inputs_;
  }
  const std::map<std::string, std::vector<std::string>>& outputs() const {
    return outputs_;
  }
  const std::vector<Op>& sub_block() const { return sub_block_; }

 private:
  std::string type_;
  std::map<std::string, std::vector<std::string>> inputs_;
  std::map<std::string, std::vector<std::string>> outputs_;
  std::vector<Op> sub_block_;
};

std::set<std::string> VarSet(
    const std::map<std::string, std::vector<std::string>>& var_map) {
  std::set<std::string> set;
  for (const auto& pair : var_map) {
    for (const auto& var : pair.second) {
      set.emplace(var);
    }
  }
  return set;
}

class ProgramDescGenerator {
 public:
  explicit ProgramDescGenerator(std::vector<Op>&& ops) {
    InitBlock(std::move(ops));
  }
  const general::ProgramDesc& general_program() { return program_desc_; }

 protected:
  int InitBlock(const std::vector<Op>& ops,
                const std::set<std::string>& inputs = {},
                const std::set<std::string>& outputs = {}) {
    auto* block = program_desc_.AddBlock<general::BlockDesc>();
    int ret{block_idx_};
    block->SetIdx(block_idx_++);
    std::set<std::string> scope_vars;
    std::set<std::string> param_vars;
    std::merge(inputs.cbegin(),
               inputs.cend(),
               outputs.cbegin(),
               outputs.cend(),
               std::inserter(param_vars, param_vars.begin()));
    for (const auto& op : ops) {
      auto* op_desc = block->AddOp<general::OpDesc>();
      op_desc->SetType(op.type());
      *(op_desc->mutable_inputs()) = op.inputs();
      *(op_desc->mutable_outputs()) = op.outputs();
      auto in_set = VarSet(op.inputs());
      auto out_set = VarSet(op.outputs());
      for (const auto& var : in_set) {
        scope_vars.emplace(var);
      }
      for (const auto& var : out_set) {
        scope_vars.emplace(var);
      }
      if (!op.sub_block().empty()) {
        op_desc->SetAttr("sub_block",
                         InitBlock(op.sub_block(), in_set, out_set));
      }
    }
    for (const auto& var : scope_vars) {
      if (param_vars.find(var) == param_vars.end()) {
        AddVar(block, var);
      }
    }
    return ret;
  }

 private:
  void AddVar(general::BlockDesc* block_desc, const std::string& name) {
    CHECK(block_desc);
    auto* var = block_desc->AddVar<general::VarDesc>();
    var->SetName(name);
    var->SetType(lite::VarDescAPI::Type::LOD_TENSOR);
  }
  general::ProgramDesc program_desc_;
  int block_idx_{0};
};

void Visualize(const general::ProgramDesc& program) {
  paddle::lite::mir::Dot dot;
  using Attr = paddle::lite::mir::Dot::Attr;
  const std::vector<Attr> op_attrs{Attr("style", "filled"),
                                   Attr("fillcolor", "yellow")};
  const std::vector<Attr> var_attrs{Attr("style", "filled"),
                                    Attr("fillcolor", "gray"),
                                    Attr("shape", "record")};
  const std::vector<Attr> edge_attrs{};
  std::set<std::string> vars{};
  for (size_t block_idx = 0; block_idx < program.BlocksSize(); block_idx++) {
    dot.Clear();
    vars.clear();
    const lite::cpp::BlockDesc* block =
        program.GetBlock<lite::cpp::BlockDesc>(block_idx);
    for (size_t op_idx = 0; op_idx < block->OpsSize(); op_idx++) {
      const lite::cpp::OpDesc* op = block->GetOp<lite::cpp::OpDesc>(op_idx);
      for (auto& in : op->input_vars()) vars.insert(in);
      for (auto& out : op->output_vars()) vars.insert(out);
    }
    for (auto& ele : vars) {
      dot.AddNode(ele, var_attrs);
    }
    for (size_t op_idx = 0; op_idx < block->OpsSize(); op_idx++) {
      const lite::cpp::OpDesc* op = block->GetOp<lite::cpp::OpDesc>(op_idx);
      const std::string op_indx_str = std::to_string(op_idx);
      dot.AddNode(op_indx_str, op_attrs, op->Type());
      for (auto& var_name : op->input_vars()) {
        dot.AddEdge(var_name, op_indx_str, edge_attrs);
      }
      for (auto& var_name : op->output_vars()) {
        dot.AddEdge(op_indx_str, var_name, edge_attrs);
      }
    }
    std::string graph = dot.Build();
    std::cout << graph << "\n";
  }
}

// Here are some directed graphs used to represent loops to acyclics.
// Unless otherwise specified, the single connection direction is downward,
// and the double connection represents a ring.
/*
 *  Block 2:
 *
 *      tmp_0
 *        |
 *   Reshape[2,0]
 */
std::vector<Op> BlockOps_2() {
  std::vector<Op> ops;
  ops.emplace_back(Op{"Reshape_0", {{"X", {"tmp_0"}}}, {{"Y", {"tmp_0"}}}});
  return ops;
}

/*
 *
 *  Block 1:
 *
 *       --------------------------  tmp_0  ---------------------
 *      / /            /              | |         /|\            \
 *  Reshape[1,0]  Operator[1,1]  Block Op[1,2]     |       Operator[1,3]
 *                            \                    |          /
 *                           tmp_1 ---------       |       tmp_2
 *                                          \      |     /    \ \
 *                                           Operator[1,5]   Reshape[1,4]
 *                                                 |
 *                                              var_out
 */
std::vector<Op> BlockOps_1() {
  std::vector<Op> ops;
  ops.emplace_back(Op{"Reshape_0", {{"X", {"tmp_0"}}}, {{"Y", {"tmp_0"}}}});
  ops.emplace_back(Op{"Operator_1", {{"X", {"tmp_0"}}}, {{"Y", {"tmp_1"}}}});
  ops.emplace_back(Op{
      "fake_block_op", {{"X", {"tmp_0"}}}, {{"Out", {"tmp_0"}}}, BlockOps_2()});
  ops.emplace_back(Op{"Operator_3", {{"X", {"tmp_0"}}}, {{"Y", {"tmp_2"}}}});
  ops.emplace_back(Op{"Operator_4", {{"X", {"tmp_2"}}}, {{"Y", {"tmp_2"}}}});
  ops.emplace_back(Op{"Operator_5",
                      {{"X0", {"tmp_1"}}, {"X1", {"tmp_2"}}},
                      {{"Y", {"var_out", "tmp_0"}}}});
  return ops;
}

/*
 *  The meaning of square brackets:
 *  Operator [block_id, exe_order]
 *
 *  Block 0:
 *
 *     Feed[0,0]
 *        |
 *      tmp_0
 *        |
 *  Block Op[0,1]
 *        |
 *     var_out
 *        |
 *    Fetch[0,2]
 */
std::vector<Op> BlockOps_0() {
  std::vector<Op> ops;
  ops.emplace_back(Op{"Feed_0", {{"X", {"feed"}}}, {{"Y", {"tmp_0"}}}});
  ops.emplace_back(Op{"fake_block_op",
                      {{"X", {"tmp_0"}}},
                      {{"Out", {"var_out", "tmp_0"}}},
                      BlockOps_1()});
  ops.emplace_back(Op{"Fetch_2", {{"X", {"var_out"}}}, {{"Y", {"fetch"}}}});
  return ops;
}

void PrintGeneralProgram(const general::ProgramDesc& general_prog) {
  paddle::framework::proto::ProgramDesc pb_proto_desc;
  lite::pb::ProgramDesc pb_desc(&pb_proto_desc);
  TransformProgramDescCppToAny(general_prog, &pb_desc);
  std::string s;
  if (google::protobuf::TextFormat::PrintToString(pb_proto_desc, &s)) {
    std::cout << "Test Program:\n" << s << std::endl;
  } else {
    LOG(FATAL) << "The format of protobuf desc is wrong.";
  }
}

}  // namespace

/*  The flat directed acyclic graph generated by the above three blocks.
 *
 *                        Feed[0,0]
 *                            |
 *                         tmp_0{0}
 *                            |
 *                       Reshape[1,0]
 *                            |
 *            ----------  tmp_0{1}  ----------------------------
 *            |               |                                 |
 *       Operator[1,1]        |                                 |
 *            |               |                                 |
 *   ----  tmp_1{0}     ---------------- tmp_0{2}               |
 *  |         |        /      |           /   \                 |
 *  |    WB[tmp_0{1}][2,1]    Reshape[2,0]   Operator[1,3]      |
 *  |                                             |             |
 *  |                                          tmp_2{0}         |
 *  |                                             |             |
 *  |                                        Reshape[1,4]       |
 *  |                                             |             |
 *   ---------------------------------------    tmp_2{1}        |
 *                                           \    |             |
 *                                           Operator[1,5]      |
 *                                          /     |             |
 *                                ----------    tmp_0{3}        |
 *                               |                |             |
 *                           var_out{0}    WB[tmp_0{1}][1,6]  --
 *                               |
 *                           Fetch[0,2]
 */

// TEST(SSAProgramTest, test) {
//   ProgramDescGenerator program_gen(BlockOps_0());
//   general::ProgramDesc cpp_desc{program_gen.general_program()};
//   ssa::PlainProgramDesc plain_program(cpp_desc);
//   ssa::ProgramDescConverter converter(plain_program);
//   PrintGeneralProgram(converter.general_program());
// }

std::vector<Op> test_2_BlockOps_1() {
  std::vector<Op> ops;
  ops.emplace_back(Op{"lod_array_length",
                      {{"X", {"array_0.out"}}},
                      {{"Out", {"array_length_0.tmp_0"}}}});
  ops.emplace_back(
      Op{"arg_max", {{"X", {"tmp_7"}}}, {{"Out", {"argmax_0.tmp_0"}}}});
  ops.emplace_back(
      Op{"write_to_array",
         {{"X", {"argmax_0.tmp_0"}}, {"I", {"array_length_0.tmp_0"}}},
         {{"Out", {"array_0.out"}}}});

  return ops;
}

std::vector<Op> test_2_BlockOps_0() {
  std::vector<Op> ops;
  // ops.emplace_back(Op{"feed", {}, {{"Out", {"array_0.out"}}}});
  ops.emplace_back(Op{"conditional_block",
                      {{"Input", {"array_0.out", "tmp_7"}}},
                      {{"Out", {"array_0.out"}}},
                      test_2_BlockOps_1()});
  return ops;
}

TEST(SSAProgramTest_2, test) {
  ProgramDescGenerator program_gen(test_2_BlockOps_0());
  general::ProgramDesc cpp_desc{program_gen.general_program()};
  Visualize(cpp_desc);
  for (size_t block_idx = 0; block_idx < cpp_desc.BlocksSize(); block_idx++) {
    const lite::general::BlockDesc* block =
        cpp_desc.GetBlock<lite::general::BlockDesc>(block_idx);
    for (size_t var_idx = 0; var_idx < block->VarsSize(); var_idx++) {
      const lite::general::VarDesc* var =
          block->GetVar<lite::general::VarDesc>(var_idx);
      if (var->Name() == "array_0.out")
        const_cast<lite::general::VarDesc*>(var)->SetType(
            lite::VarDescAPI::Type::LOD_TENSOR_ARRAY);
    }
  }
  ssa::PlainProgramDesc plain_program(cpp_desc);
  ssa::ProgramDescConverter converter(plain_program);
  const general::ProgramDesc program = converter.general_program();
  // PrintGeneralProgram(program);
  Visualize(program);
}

}  // namespace general
}  // namespace lite
}  // namespace paddle
