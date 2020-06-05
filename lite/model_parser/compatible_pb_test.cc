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

#include "lite/model_parser/compatible_pb.h"
#include <gtest/gtest.h>
#include "lite/model_parser/cpp/block_desc.h"
#include "lite/model_parser/cpp/op_desc.h"
#include "lite/model_parser/cpp/program_desc.h"
#include "lite/model_parser/cpp/var_desc.h"
#include "lite/model_parser/naive_buffer/block_desc.h"
#include "lite/model_parser/naive_buffer/op_desc.h"
#include "lite/model_parser/naive_buffer/program_desc.h"
#include "lite/model_parser/naive_buffer/var_desc.h"
#include "lite/model_parser/pb/block_desc.h"
#include "lite/model_parser/pb/op_desc.h"
#include "lite/model_parser/pb/program_desc.h"
#include "lite/model_parser/pb/var_desc.h"

namespace paddle {
namespace lite {

/// For VarDesc test
template <typename VarDescType>
void SetVarDesc(VarDescType* desc) {
  desc->SetName("X");
  desc->SetPersistable(true);
  desc->SetType(VarDescAPI::Type::LOD_TENSOR);
  desc->SetShape({1, 3, 224, 224});
  desc->SetDataType(VarDescAPI::VarDataType::FP32);
}

template <typename VarDescType>
void SetVarDesc1(VarDescType* desc) {
  desc->SetName("Y");
  desc->SetPersistable(false);
  desc->SetType(VarDescAPI::Type::SELECTED_ROWS);
  desc->SetShape({1, 3, 224, 224});
  desc->SetDataType(VarDescAPI::VarDataType::FP32);
}

template <typename VarDescType>
void CheckVarDesc(const VarDescType& desc) {
  ASSERT_EQ(desc.Name(), "X");
  ASSERT_TRUE(desc.Persistable());
  ASSERT_EQ(desc.GetType(), VarDescAPI::Type::LOD_TENSOR);
}

template <typename VarDescType>
void CheckVarDesc1(const VarDescType& desc) {
  ASSERT_EQ(desc.Name(), "Y");
  ASSERT_FALSE(desc.Persistable());
  ASSERT_EQ(desc.GetType(), VarDescAPI::Type::SELECTED_ROWS);
}

template <typename VarDescType>
void TestVarX(VarDescType* desc) {
  SetVarDesc<VarDescType>(desc);
  CheckVarDesc<VarDescType>(*desc);
}

TEST(VarDesc, Basic) {
  // pb VarDesc
  framework::proto::VarDesc pb_proto_desc;
  pb::VarDesc pb_desc(&pb_proto_desc);
  TestVarX<pb::VarDesc>(&pb_desc);

  // cpp VarDesc
  cpp::VarDesc cpp_desc;
  TestVarX<cpp::VarDesc>(&cpp_desc);

  // naive buffer OpDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::VarDesc nb_proto_desc(&table);
  naive_buffer::VarDesc nb_desc(&nb_proto_desc);
  TestVarX<naive_buffer::VarDesc>(&nb_desc);
}

template <typename VarDescType>
void TestVarCppToAny(VarDescType* any_desc) {
  cpp::VarDesc desc;
  SetVarDesc1<cpp::VarDesc>(&desc);
  TransformVarDescCppToAny(desc, any_desc);
  CheckVarDesc1<VarDescType>(*any_desc);
}

TEST(VarDesc, CppToAny) {
  // pb VarDesc
  framework::proto::VarDesc pb_proto_desc;
  pb::VarDesc pb_desc(&pb_proto_desc);
  TestVarCppToAny<pb::VarDesc>(&pb_desc);

  // naive buffer VarDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::VarDesc nb_proto_desc(&table);
  naive_buffer::VarDesc nb_desc(&nb_proto_desc);
  TestVarCppToAny<naive_buffer::VarDesc>(&nb_desc);
}

template <typename VarDescType>
void TestVarAnyToCpp(VarDescType* desc) {
  SetVarDesc1<VarDescType>(desc);
  cpp::VarDesc cpp_desc;
  TransformVarDescAnyToCpp(*desc, &cpp_desc);
  CheckVarDesc1<cpp::VarDesc>(cpp_desc);
}

TEST(VarDesc, AnyToCpp) {
  // pb VarDesc
  framework::proto::VarDesc pb_proto_desc;
  pb::VarDesc pb_desc(&pb_proto_desc);
  TestVarAnyToCpp<pb::VarDesc>(&pb_desc);

  // naive buffer VarDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::VarDesc nb_proto_desc(&table);
  naive_buffer::VarDesc nb_desc(&nb_proto_desc);
  TestVarAnyToCpp<naive_buffer::VarDesc>(&nb_desc);
}

/// For OpDesc test
template <typename OpDescType>
void SetOpDesc(OpDescType* desc) {
  desc->SetInput("X", {"a", "b"});
  desc->SetOutput("Y", {"c", "d"});
  desc->template SetAttr<int32_t>("aint", 100);
}

template <typename OpDescType>
void SetOpDesc1(OpDescType* desc) {
  desc->SetInput("X", {"m", "n", "k"});
  desc->SetOutput("Y", {"w"});
  desc->template SetAttr<float>("afloat", 0.005);
}

template <typename OpDescType>
void CheckOpDesc(const OpDescType& desc) {
  auto X = desc.Input("X");
  ASSERT_EQ(X.size(), 2UL);
  ASSERT_EQ(X[0], "a");
  ASSERT_EQ(X[1], "b");

  auto Y = desc.Output("Y");
  ASSERT_EQ(Y.size(), 2UL);
  ASSERT_EQ(Y[0], "c");
  ASSERT_EQ(Y[1], "d");

  ASSERT_TRUE(desc.HasAttr("aint"));
  ASSERT_FALSE(desc.HasAttr("afloat"));
  ASSERT_EQ(desc.template GetAttr<int32_t>("aint"), 100);
}

template <typename OpDescType>
void CheckOpDesc1(const OpDescType& desc) {
  auto X = desc.Input("X");
  ASSERT_EQ(X.size(), 3UL);
  ASSERT_EQ(X[0], "m");
  ASSERT_EQ(X[1], "n");
  ASSERT_EQ(X[2], "k");

  auto Y = desc.Output("Y");
  ASSERT_EQ(Y.size(), 1UL);
  ASSERT_EQ(Y[0], "w");

  ASSERT_TRUE(desc.HasAttr("afloat"));
  ASSERT_FALSE(desc.HasAttr("aint"));
  EXPECT_NEAR(desc.template GetAttr<float>("afloat"), 0.005, 1e-5);
}

template <typename OpDescType>
void TestOpX(OpDescType* desc) {
  SetOpDesc<OpDescType>(desc);
  CheckOpDesc<OpDescType>(*desc);
}

TEST(OpDesc, Basic) {
  // pb OpDesc
  framework::proto::OpDesc pb_proto_desc;
  pb::OpDesc pb_desc(&pb_proto_desc);
  TestOpX<pb::OpDesc>(&pb_desc);

  // cpp OpDesc
  cpp::OpDesc cpp_desc;
  TestOpX<cpp::OpDesc>(&cpp_desc);

  // naive buffer OpDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::OpDesc nb_proto_desc(&table);
  naive_buffer::OpDesc nb_desc(&nb_proto_desc);
  TestOpX<naive_buffer::OpDesc>(&nb_desc);
}

template <typename OpDescType>
void TestOpCppToAny(OpDescType* any_desc) {
  cpp::OpDesc desc;
  SetOpDesc1<cpp::OpDesc>(&desc);
  TransformOpDescCppToAny(desc, any_desc);
  CheckOpDesc1<OpDescType>(*any_desc);
}

TEST(OpDesc, CppToAny) {
  // pb OpDesc
  framework::proto::OpDesc pb_proto_desc;
  pb::OpDesc pb_desc(&pb_proto_desc);
  TestOpCppToAny<pb::OpDesc>(&pb_desc);

  // naive buffer OpDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::OpDesc nb_proto_desc(&table);
  naive_buffer::OpDesc nb_desc(&nb_proto_desc);
  TestOpCppToAny<naive_buffer::OpDesc>(&nb_desc);
}

template <typename OpDescType>
void TestOpAnyToCpp(OpDescType* desc) {
  SetOpDesc1<OpDescType>(desc);
  cpp::OpDesc cpp_desc;
  TransformOpDescAnyToCpp(*desc, &cpp_desc);
  CheckOpDesc1<cpp::OpDesc>(cpp_desc);
}

TEST(OpDesc, AnyToCpp) {
  // pb OpDesc
  framework::proto::OpDesc pb_proto_desc;
  pb::OpDesc pb_desc(&pb_proto_desc);
  TestOpAnyToCpp<pb::OpDesc>(&pb_desc);

  // naive buffer OpDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::OpDesc nb_proto_desc(&table);
  naive_buffer::OpDesc nb_desc(&nb_proto_desc);
  TestOpAnyToCpp<naive_buffer::OpDesc>(&nb_desc);
}

template <typename T>
void SetBlockDesc(T* desc);

/// For BlockDesc test
#define SET_BLOCK_DESC(NT, PNT)                            \
  template <>                                              \
  void SetBlockDesc<NT::BlockDesc>(NT::BlockDesc * desc) { \
    desc->ClearVars();                                     \
    desc->ClearOps();                                      \
                                                           \
    desc->SetIdx(1);                                       \
    desc->SetParentIdx(-1);                                \
    desc->SetForwardBlockIdx(2);                           \
                                                           \
    NT::VarDesc var1(desc->AddVar<PNT::VarDesc>());        \
    SetVarDesc<NT::VarDesc>(&var1);                        \
    NT::VarDesc var2(desc->AddVar<PNT::VarDesc>());        \
    SetVarDesc1<NT::VarDesc>(&var2);                       \
                                                           \
    NT::OpDesc op1(desc->AddOp<PNT::OpDesc>());            \
    SetOpDesc<NT::OpDesc>(&op1);                           \
    NT::OpDesc op2(desc->AddOp<PNT::OpDesc>());            \
    SetOpDesc1<NT::OpDesc>(&op2);                          \
  }

template <>
void SetBlockDesc<cpp::BlockDesc>(cpp::BlockDesc* desc) {
  desc->ClearVars();
  desc->ClearOps();

  desc->SetIdx(1);
  desc->SetParentIdx(-1);
  desc->SetForwardBlockIdx(2);

  SetVarDesc<cpp::VarDesc>(desc->AddVar<cpp::VarDesc>());
  SetVarDesc1<cpp::VarDesc>(desc->AddVar<cpp::VarDesc>());

  SetOpDesc<cpp::OpDesc>(desc->AddOp<cpp::OpDesc>());
  SetOpDesc1<cpp::OpDesc>(desc->AddOp<cpp::OpDesc>());
}

SET_BLOCK_DESC(naive_buffer, naive_buffer::proto);
SET_BLOCK_DESC(pb, framework::proto);

template <typename T>
void CheckBlockDesc(const T& desc);

#define CHECK_BLOCK_DESC(NT, PNT)                                      \
  template <>                                                          \
  void CheckBlockDesc<NT::BlockDesc>(const NT::BlockDesc& some_desc) { \
    auto desc = some_desc;                                             \
    ASSERT_EQ(desc.Idx(), 1);                                          \
    ASSERT_EQ(desc.ParentIdx(), -1);                                   \
    ASSERT_EQ(desc.ForwardBlockIdx(), 2);                              \
                                                                       \
    ASSERT_EQ(desc.VarsSize(), 2UL);                                   \
    NT::VarDesc var1(desc.GetVar<PNT::VarDesc>(0));                    \
    CheckVarDesc<NT::VarDesc>(var1);                                   \
    NT::VarDesc var2(desc.GetVar<PNT::VarDesc>(1));                    \
    CheckVarDesc1<NT::VarDesc>(var2);                                  \
                                                                       \
    ASSERT_EQ(desc.OpsSize(), 2UL);                                    \
    NT::OpDesc op1(desc.GetOp<PNT::OpDesc>(0));                        \
    CheckOpDesc<NT::OpDesc>(op1);                                      \
    NT::OpDesc op2(desc.GetOp<PNT::OpDesc>(1));                        \
    CheckOpDesc1<NT::OpDesc>(op2);                                     \
  }

CHECK_BLOCK_DESC(naive_buffer, naive_buffer::proto);
CHECK_BLOCK_DESC(pb, framework::proto);

template <>
void CheckBlockDesc<cpp::BlockDesc>(const cpp::BlockDesc& some_desc) {
  auto desc = some_desc;
  ASSERT_EQ(desc.Idx(), 1);
  ASSERT_EQ(desc.ParentIdx(), -1);
  ASSERT_EQ(desc.ForwardBlockIdx(), 2);

  ASSERT_EQ(desc.VarsSize(), 2UL);
  CheckVarDesc<cpp::VarDesc>(*desc.GetVar<cpp::VarDesc>(0));
  CheckVarDesc1<cpp::VarDesc>(*desc.GetVar<cpp::VarDesc>(1));

  ASSERT_EQ(desc.OpsSize(), 2UL);
  CheckOpDesc<cpp::OpDesc>(*desc.GetOp<cpp::OpDesc>(0));
  CheckOpDesc1<cpp::OpDesc>(*desc.GetOp<cpp::OpDesc>(1));
}

template <typename BlockDescType>
void TestBlockX(BlockDescType* desc) {
  SetBlockDesc<BlockDescType>(desc);
  CheckBlockDesc<BlockDescType>(*desc);
}

TEST(BlockDesc, Basic) {
  // pb BlockDesc
  framework::proto::BlockDesc pb_proto_desc;
  pb::BlockDesc pb_desc(&pb_proto_desc);
  TestBlockX<pb::BlockDesc>(&pb_desc);

  // cpp OpDesc
  cpp::BlockDesc cpp_desc;
  TestBlockX<cpp::BlockDesc>(&cpp_desc);

  // naive buffer OpDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::BlockDesc nb_proto_desc(&table);
  naive_buffer::BlockDesc nb_desc(&nb_proto_desc);
  TestBlockX<naive_buffer::BlockDesc>(&nb_desc);
}

template <typename BlockDescType>
void TestBlockCppToAny(BlockDescType* any_desc) {
  cpp::BlockDesc desc;
  SetBlockDesc<cpp::BlockDesc>(&desc);
  TransformBlockDescCppToAny(desc, any_desc);
  CheckBlockDesc<BlockDescType>(*any_desc);
}

TEST(BlockDesc, CppToAny) {
  // pb BlockDesc
  framework::proto::BlockDesc pb_proto_desc;
  pb::BlockDesc pb_desc(&pb_proto_desc);
  TestBlockCppToAny<pb::BlockDesc>(&pb_desc);

  // naive buffer BlockDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::BlockDesc nb_proto_desc(&table);
  naive_buffer::BlockDesc nb_desc(&nb_proto_desc);
  TestBlockCppToAny<naive_buffer::BlockDesc>(&nb_desc);
}

template <typename BlockDescType>
void TestBlockAnyToCpp(BlockDescType* desc) {
  SetBlockDesc<BlockDescType>(desc);
  cpp::BlockDesc cpp_desc;
  TransformBlockDescAnyToCpp(*desc, &cpp_desc);
  CheckBlockDesc<cpp::BlockDesc>(cpp_desc);
}

TEST(BlockDesc, AnyToCpp) {
  // pb OpDesc
  framework::proto::BlockDesc pb_proto_desc;
  pb::BlockDesc pb_desc(&pb_proto_desc);
  TestBlockAnyToCpp<pb::BlockDesc>(&pb_desc);

  // naive buffer OpDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::BlockDesc nb_proto_desc(&table);
  naive_buffer::BlockDesc nb_desc(&nb_proto_desc);
  TestBlockAnyToCpp<naive_buffer::BlockDesc>(&nb_desc);
}

/// For ProgramDesc test
template <typename ProgramDescType>
void TestProgramCppToAny(ProgramDescType* any_desc) {
  cpp::ProgramDesc desc;
  TransformProgramDescCppToAny(desc, any_desc);
}

TEST(ProgramDesc, CppToAny) {
  // pb ProgramDesc
  framework::proto::ProgramDesc pb_proto_desc;
  pb::ProgramDesc pb_desc(&pb_proto_desc);
  TestProgramCppToAny<pb::ProgramDesc>(&pb_desc);

  // naive buffer ProgramDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::ProgramDesc nb_proto_desc(&table);
  naive_buffer::ProgramDesc nb_desc(&nb_proto_desc);
  TestProgramCppToAny<naive_buffer::ProgramDesc>(&nb_desc);
}

template <typename ProgramDescType>
void TestProgramAnyToCpp(ProgramDescType* desc) {
  cpp::ProgramDesc cpp_desc;
  TransformProgramDescAnyToCpp(*desc, &cpp_desc);
}

TEST(ProgramDesc, AnyToCpp) {
  // pb OpDesc
  framework::proto::ProgramDesc pb_proto_desc;
  pb::ProgramDesc pb_desc(&pb_proto_desc);
  TestProgramAnyToCpp<pb::ProgramDesc>(&pb_desc);

  // naive buffer OpDesc
  naive_buffer::BinaryTable table;
  naive_buffer::proto::ProgramDesc nb_proto_desc(&table);
  naive_buffer::ProgramDesc nb_desc(&nb_proto_desc);
  TestProgramAnyToCpp<naive_buffer::ProgramDesc>(&nb_desc);
}

}  // namespace lite
}  // namespace paddle
