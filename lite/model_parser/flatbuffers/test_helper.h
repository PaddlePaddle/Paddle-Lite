// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <string>
#include <vector>
#include "lite/model_parser/base/io.h"
#include "lite/model_parser/flatbuffers/program_desc.h"

namespace paddle {
namespace lite {
namespace fbs {
namespace test {
#ifdef LITE_WITH_FLATBUFFERS_DESC
inline lite::model_parser::Buffer GenerateProgramCache() {
  /* --------- Set Program --------- */
  ProgramDesc program;
  program.SetVersion(1000600);

  /* --------- Set Block A --------- */
  BlockDesc block_a(program.AddBlock<proto::BlockDescT>());

  VarDesc var_a2(block_a.AddVar<proto::VarDescT>());
  var_a2.SetType(paddle::lite::VarDataType::LOD_TENSOR);
  var_a2.SetName("var_a2");
  var_a2.SetShape({2, 2, 1});

  VarDesc var_a0(block_a.AddVar<proto::VarDescT>());
  var_a0.SetType(paddle::lite::VarDataType::LOD_TENSOR);
  var_a0.SetName("var_a0");
  var_a0.SetShape({1, 2});

  OpDesc op_a0(block_a.AddOp<proto::OpDescT>());
  op_a0.SetType("Type");
  op_a0.SetInput("X", {"var_a0"});
  op_a0.SetOutput("Y0", {"var_a0", "var_a1"});
  op_a0.SetOutput("Y1", {"var_a2"});
  op_a0.SetAttr<std::string>("Attr5", "attr_5");
  op_a0.SetAttr<std::vector<std::string>>("Attr2", {"attr_2"});
  op_a0.SetAttr<float>("Attr1", 0.98f);
  op_a0.SetAttr<int32_t>("Attr0", 16);

  /* --------- Set Block B --------- */
  BlockDesc block_b(program.AddBlock<proto::BlockDescT>());

  VarDesc var_b0(block_b.AddVar<proto::VarDescT>());
  var_b0.SetType(paddle::lite::VarDataType::LOD_TENSOR);
  var_b0.SetName("var_b0");
  var_b0.SetShape({-1, 1});

  OpDesc op_b0(block_b.AddOp<proto::OpDescT>());
  op_b0.SetType("Type0");
  op_b0.SetInput("X", {"var_b0"});
  op_b0.SetOutput("Y1", {"var_b0"});
  op_b0.SetAttr<std::string>("Attr5", "attr_5");

  OpDesc op_b1(block_b.AddOp<proto::OpDescT>());
  op_b1.SetType("Type1");
  op_b1.SetInput("X", {"var_b0"});
  op_b1.SetOutput("Y1", {"var_b0"});
  op_b1.SetAttr<std::string>("Attr5", "attr_5");
  op_b1.SetAttr<std::vector<std::string>>("Attr2", {"attr_2"});
  op_b1.SetAttr<bool>("Attr1", true);

  /* --------- Cache Program ---------- */
  return program.data();
}

inline void CheckProgramCache(ProgramDesc* program) {
  CHECK_EQ(program->Version(), 1000600);
  CHECK_EQ(program->BlocksSize(), 2u);

  /* --------- Check Block A --------- */
  BlockDesc block_a(program->GetBlock<proto::BlockDescT>(0));
  CHECK_EQ(block_a.OpsSize(), 1u);
  CHECK_EQ(block_a.VarsSize(), 2u);

  auto var_a2 = VarDesc(block_a.GetVar<proto::VarDescT>(0));
  CHECK(var_a2.GetShape() == std::vector<int64_t>({2, 2, 1}));

  auto op_a0 = OpDesc(block_a.GetOp<proto::OpDescT>(0));
  CHECK_EQ(op_a0.Type(), std::string("Type"));
  CHECK(op_a0.Input("X") == std::vector<std::string>({"var_a0"}));
  CHECK(op_a0.Output("Y0") == std::vector<std::string>({"var_a0", "var_a1"}));
  CHECK(op_a0.Output("Y1") == std::vector<std::string>({"var_a2"}));
  CHECK_EQ(op_a0.GetAttr<float>("Attr1"), 0.98f);
  CHECK_EQ(op_a0.GetAttr<int32_t>("Attr0"), 16);
  CHECK_EQ(op_a0.GetAttr<std::string>("Attr5"), std::string("attr_5"));
  CHECK(op_a0.GetAttr<std::vector<std::string>>("Attr2") ==
        std::vector<std::string>({"attr_2"}));

  /* --------- Check Block B --------- */
  BlockDesc block_b(program->GetBlock<proto::BlockDescT>(1));
  CHECK_EQ(block_b.OpsSize(), 2u);
  CHECK_EQ(block_b.VarsSize(), 1u);

  auto op_b0 = OpDesc(block_b.GetOp<proto::OpDescT>(1));
  CHECK_EQ(op_b0.GetAttr<bool>("Attr1"), true);
  CHECK_EQ(op_b0.HasAttr("Attr4"), false);
}

inline void CheckProgramCache(const ProgramDescView& program) {
  CHECK_EQ(program.Version(), 1000600);
  CHECK_EQ(program.BlocksSize(), 2u);

  /* --------- Check Block A --------- */
  const auto& block_a = *program.GetBlock<BlockDescView>(0);
  CHECK_EQ(block_a.OpsSize(), 1u);
  CHECK_EQ(block_a.VarsSize(), 2u);

  const auto& var_a2 = *block_a.GetVar<VarDescView>(0);
  CHECK(var_a2.GetShape() == std::vector<int64_t>({2, 2, 1}));

  const auto& op_a0 = *block_a.GetOp<OpDescView>(0);
  CHECK_EQ(op_a0.Type(), std::string("Type"));
  CHECK(op_a0.Input("X") == std::vector<std::string>({"var_a0"}));
  CHECK(op_a0.Output("Y0") == std::vector<std::string>({"var_a0", "var_a1"}));
  CHECK(op_a0.Output("Y1") == std::vector<std::string>({"var_a2"}));
  CHECK_EQ(op_a0.GetAttr<float>("Attr1"), 0.98f);
  CHECK_EQ(op_a0.GetAttr<int32_t>("Attr0"), 16);
  CHECK_EQ(op_a0.GetAttr<std::string>("Attr5"), std::string("attr_5"));
  CHECK(static_cast<std::vector<std::string>>(
            op_a0.GetAttr<std::vector<std::string>>("Attr2")) ==
        std::vector<std::string>({"attr_2"}));

  /* --------- Check Block B --------- */
  const auto& block_b = *program.GetBlock<BlockDescView>(1);
  CHECK_EQ(block_b.OpsSize(), 2u);
  CHECK_EQ(block_b.VarsSize(), 1u);

  const auto& op_b0 = *block_b.GetOp<OpDescView>(1);
  CHECK_EQ(op_b0.GetAttr<bool>("Attr1"), true);
  CHECK_EQ(op_b0.HasAttr("Attr4"), false);
}
#endif  // LITE_WITH_FLATBUFFERS_DESC

}  // namespace test
}  // namespace fbs
}  // namespace lite
}  // namespace paddle
