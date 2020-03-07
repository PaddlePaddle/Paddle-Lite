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

#include <gtest/gtest.h>
#include "lite/model_parser/naive_buffer/block_desc.h"
#include "lite/model_parser/naive_buffer/combined_params_desc.h"
#include "lite/model_parser/naive_buffer/op_desc.h"
#include "lite/model_parser/naive_buffer/param_desc.h"
#include "lite/model_parser/naive_buffer/program_desc.h"
#include "lite/model_parser/naive_buffer/var_desc.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

TEST(NaiveBufferWrapper, OpDesc) {
  BinaryTable table0;
  proto::OpDesc pt_desc0(&table0);
  OpDesc nb_desc0(&pt_desc0);

  // Set OpDesc
  nb_desc0.SetType("mul");
  nb_desc0.SetInput("X", {"a"});
  nb_desc0.SetInput("Y", {"b"});
  nb_desc0.SetOutput("Out", {"c"});
  nb_desc0.SetAttr<int32_t>("x_num_col_dims", 0);
  nb_desc0.SetAttr<int32_t>("y_num_col_dims", 1);

  // Save model
  pt_desc0.Save();
  table0.SaveToFile("1.bf");

  // Load model
  BinaryTable table1;
  table1.LoadFromFile("1.bf");
  proto::OpDesc pt_desc1(&table1);
  pt_desc1.Load();
  OpDesc nb_desc1(&pt_desc1);

  ASSERT_EQ(nb_desc1.Type(), "mul");
  auto x = nb_desc1.Input("X");
  ASSERT_EQ(x.size(), 1);
  ASSERT_EQ(x[0], "a");
  auto y = nb_desc1.Input("Y");
  ASSERT_EQ(y.size(), 1);
  ASSERT_EQ(y[0], "b");
  auto out = nb_desc1.Output("Out");
  ASSERT_EQ(out.size(), 1);
  ASSERT_EQ(out[0], "c");
  ASSERT_TRUE(nb_desc1.HasAttr("x_num_col_dims"));
  ASSERT_EQ(nb_desc1.GetAttr<int32_t>("x_num_col_dims"), 0);
  ASSERT_EQ(nb_desc1.GetAttrType("x_num_col_dims"), OpDescAPI::AttrType::INT);
  ASSERT_TRUE(nb_desc1.HasAttr("y_num_col_dims"));
  ASSERT_EQ(nb_desc1.GetAttr<int32_t>("y_num_col_dims"), 1);
  ASSERT_EQ(nb_desc1.GetAttrType("y_num_col_dims"), OpDescAPI::AttrType::INT);
}

TEST(NaiveBufferWrapper, VarDesc) {
  BinaryTable table0;
  proto::VarDesc pt_desc0(&table0);
  VarDesc nb_desc0(&pt_desc0);

  // Set VarDesc
  nb_desc0.SetName("a");
  nb_desc0.SetPersistable(true);
  nb_desc0.SetType(VarDescAPI::VarDataType::LOD_TENSOR);

  // Save model
  pt_desc0.Save();
  table0.SaveToFile("2.bf");

  // Load model
  BinaryTable table1;
  table1.LoadFromFile("2.bf");
  proto::VarDesc pt_desc1(&table1);
  pt_desc1.Load();
  VarDesc nb_desc1(&pt_desc1);

  ASSERT_EQ(nb_desc1.Name(), "a");
  ASSERT_EQ(nb_desc1.GetType(), VarDescAPI::VarDataType::LOD_TENSOR);
  ASSERT_TRUE(nb_desc1.Persistable());
}

TEST(NaiveBufferWrapper, ParamDesc) {
  BinaryTable table0;
  proto::ParamDesc pt_desc0(&table0);
  ParamDesc nb_desc0(&pt_desc0);

  // Set ParamDesc
  nb_desc0.SetName("fc_w.0");
  nb_desc0.SetModelVersion(0);
  nb_desc0.SetTensorVersion(1);
  std::vector<std::vector<uint64_t>> lod({{1, 2, 3}, {4, 5}});
  nb_desc0.SetLoDLevel(2);
  nb_desc0.SetLoD(lod);
  std::vector<int64_t> dim({1, 2, 5});
  nb_desc0.SetDim(dim);
  nb_desc0.SetDataType(VarDescAPI::VarDataType::FP32);
  std::vector<float> data;
  for (int i = 0; i < 10; ++i) {
    data.push_back(i / 10.0);
  }
  nb_desc0.SetData(data);

  // Save model
  pt_desc0.Save();
  table0.SaveToFile("3.bf");

  // Load model
  BinaryTable table1;
  table1.LoadFromFile("3.bf");
  proto::ParamDesc pt_desc1(&table1);
  pt_desc1.Load();
  ParamDesc nb_desc1(&pt_desc1);

  ASSERT_EQ(nb_desc1.Name(), "fc_w.0");
  ASSERT_EQ(nb_desc1.ModelVersion(), 0);
  ASSERT_EQ(nb_desc1.TensorVersion(), 1);
  ASSERT_EQ(nb_desc1.LoDLevel(), 2);
  ASSERT_EQ(nb_desc1.LoD(), lod);
  ASSERT_EQ(nb_desc1.Dim(), dim);
  auto data1 = nb_desc1.Data<float>();
  ASSERT_EQ(data1.size(), data.size());
  for (size_t i = 0; i < data1.size(); ++i) {
    EXPECT_NEAR(data1[i], data[i], 1e-6);
  }
}

TEST(NaiveBufferWrapper, CombinedParamsDesc) {
  BinaryTable table0;
  proto::CombinedParamsDesc pt_desc0(&table0);
  CombinedParamsDesc nb_desc0(&pt_desc0);

  // Set ParamDesc
  ParamDesc param_desc0_0(nb_desc0.AddParam());
  param_desc0_0.SetName("fc_w.0");
  param_desc0_0.SetModelVersion(0);
  param_desc0_0.SetTensorVersion(1);
  std::vector<std::vector<uint64_t>> param_desc0_0_lod({{1, 2, 3}, {4, 5}});
  param_desc0_0.SetLoDLevel(2);
  param_desc0_0.SetLoD(param_desc0_0_lod);
  std::vector<int64_t> param_desc0_0_dim({1, 2, 5});
  param_desc0_0.SetDim(param_desc0_0_dim);
  param_desc0_0.SetDataType(VarDescAPI::VarDataType::FP32);
  std::vector<float> param_desc0_0_data;
  for (int i = 0; i < 10; ++i) {
    param_desc0_0_data.push_back(i / 10.0);
  }
  param_desc0_0.SetData(param_desc0_0_data);

  ParamDesc param_desc0_1(nb_desc0.AddParam());
  param_desc0_1.SetName("fc_b.0");
  param_desc0_1.SetModelVersion(0);
  param_desc0_1.SetTensorVersion(1);
  std::vector<std::vector<uint64_t>> param_desc0_1_lod({{1}, {2, 3}, {4, 5}});
  param_desc0_1.SetLoDLevel(3);
  param_desc0_1.SetLoD(param_desc0_1_lod);
  std::vector<int64_t> param_desc0_1_dim({1, 2, 2, 5});
  param_desc0_1.SetDim(param_desc0_1_dim);
  param_desc0_1.SetDataType(VarDescAPI::VarDataType::FP32);
  std::vector<float> param_desc0_1_data;
  for (int i = 0; i < 20; ++i) {
    param_desc0_1_data.push_back((i - 10) / 10.0);
  }
  param_desc0_1.SetData(param_desc0_1_data);

  // Save model
  pt_desc0.Save();
  table0.SaveToFile("4.bf");

  // Load model
  BinaryTable table1;
  table1.LoadFromFile("4.bf");
  proto::CombinedParamsDesc pt_desc1(&table1);
  pt_desc1.Load();
  CombinedParamsDesc nb_desc1(&pt_desc1);

  ASSERT_EQ(nb_desc1.ParamsSize(), 2);

  ParamDesc param_desc1_0(nb_desc1.GetParam(0));
  ASSERT_EQ(param_desc1_0.Name(), "fc_w.0");
  ASSERT_EQ(param_desc1_0.ModelVersion(), 0);
  ASSERT_EQ(param_desc1_0.TensorVersion(), 1);
  ASSERT_EQ(param_desc1_0.LoDLevel(), 2);
  ASSERT_EQ(param_desc1_0.LoD(), param_desc0_0_lod);
  ASSERT_EQ(param_desc1_0.Dim(), param_desc0_0_dim);
  auto param_desc1_0_data = param_desc1_0.Data<float>();
  ASSERT_EQ(param_desc1_0_data.size(), param_desc0_0_data.size());
  for (size_t i = 0; i < param_desc1_0_data.size(); ++i) {
    EXPECT_NEAR(param_desc1_0_data[i], param_desc0_0_data[i], 1e-6);
  }

  ParamDesc param_desc1_1(nb_desc1.GetParam(1));
  ASSERT_EQ(param_desc1_1.Name(), "fc_b.0");
  ASSERT_EQ(param_desc1_1.ModelVersion(), 0);
  ASSERT_EQ(param_desc1_1.TensorVersion(), 1);
  ASSERT_EQ(param_desc1_1.LoDLevel(), 3);
  ASSERT_EQ(param_desc1_1.LoD(), param_desc0_1_lod);
  ASSERT_EQ(param_desc1_1.Dim(), param_desc0_1_dim);
  auto param_desc1_1_data = param_desc1_1.Data<float>();
  ASSERT_EQ(param_desc1_1_data.size(), param_desc0_1_data.size());
  for (size_t i = 0; i < param_desc1_1_data.size(); ++i) {
    EXPECT_NEAR(param_desc1_1_data[i], param_desc0_1_data[i], 1e-6);
  }
}

TEST(NaiveBufferWrapper, BlockDesc) {
  BinaryTable table0;
  proto::BlockDesc pt_desc0(&table0);
  BlockDesc nb_desc0(&pt_desc0);

  // Set BlockDesc
  nb_desc0.SetIdx(1);
  nb_desc0.SetParentIdx(2);
  nb_desc0.SetForwardBlockIdx(3);
  VarDesc var0_0(nb_desc0.AddVar<proto::VarDesc>());
  var0_0.SetName("a");
  var0_0.SetPersistable(true);
  var0_0.SetType(VarDescAPI::VarDataType::LOD_TENSOR);
  VarDesc var0_1(nb_desc0.AddVar<proto::VarDesc>());
  var0_1.SetName("b");
  var0_1.SetPersistable(false);
  var0_1.SetType(VarDescAPI::VarDataType::READER);
  OpDesc op0_0(nb_desc0.AddOp<proto::OpDesc>());
  op0_0.SetType("mul");
  op0_0.SetInput("X", {"a"});
  op0_0.SetInput("Y", {"b"});
  op0_0.SetOutput("Out", {"c"});
  op0_0.SetAttr<int32_t>("x_num_col_dims", 0);
  op0_0.SetAttr<int32_t>("y_num_col_dims", 1);

  // Save model
  pt_desc0.Save();
  table0.SaveToFile("5.bf");

  // Load model
  BinaryTable table1;
  table1.LoadFromFile("5.bf");
  proto::BlockDesc pt_desc1(&table1);
  pt_desc1.Load();
  BlockDesc nb_desc1(&pt_desc1);

  ASSERT_EQ(nb_desc1.Idx(), 1);
  ASSERT_EQ(nb_desc1.ParentIdx(), 2);
  ASSERT_EQ(nb_desc1.ForwardBlockIdx(), 3);

  ASSERT_EQ(nb_desc1.VarsSize(), 2);
  VarDesc var1_0(nb_desc1.GetVar<proto::VarDesc>(0));
  ASSERT_EQ(var1_0.Name(), "a");
  ASSERT_EQ(var1_0.GetType(), VarDescAPI::VarDataType::LOD_TENSOR);
  ASSERT_TRUE(var1_0.Persistable());
  VarDesc var1_1(nb_desc1.GetVar<proto::VarDesc>(1));
  ASSERT_EQ(var1_1.Name(), "b");
  ASSERT_EQ(var1_1.GetType(), VarDescAPI::VarDataType::READER);
  ASSERT_FALSE(var1_1.Persistable());

  ASSERT_EQ(nb_desc1.OpsSize(), 1);
  OpDesc op1_0(nb_desc1.GetOp<proto::OpDesc>(0));
  ASSERT_EQ(op1_0.Type(), "mul");
  auto x = op1_0.Input("X");
  ASSERT_EQ(x.size(), 1);
  ASSERT_EQ(x[0], "a");
  auto y = op1_0.Input("Y");
  ASSERT_EQ(y.size(), 1);
  ASSERT_EQ(y[0], "b");
  auto out = op1_0.Output("Out");
  ASSERT_EQ(out.size(), 1);
  ASSERT_EQ(out[0], "c");
  ASSERT_TRUE(op1_0.HasAttr("x_num_col_dims"));
  ASSERT_EQ(op1_0.GetAttr<int32_t>("x_num_col_dims"), 0);
  ASSERT_EQ(op1_0.GetAttrType("x_num_col_dims"), OpDescAPI::AttrType::INT);
  ASSERT_TRUE(op1_0.HasAttr("y_num_col_dims"));
  ASSERT_EQ(op1_0.GetAttr<int32_t>("y_num_col_dims"), 1);
  ASSERT_EQ(op1_0.GetAttrType("y_num_col_dims"), OpDescAPI::AttrType::INT);
}

TEST(NaiveBufferWrapper, ProgramDesc) {
  BinaryTable table0;
  proto::ProgramDesc pt_desc0(&table0);
  ProgramDesc nb_desc0(&pt_desc0);

  // Set ProgramDesc
  nb_desc0.SetVersion(1);
  for (int i = 0; i < 3; ++i) {
    nb_desc0.AddBlock<proto::BlockDesc>();
  }

  // Save model
  pt_desc0.Save();
  table0.SaveToFile("6.bf");

  // Load model
  BinaryTable table1;
  table1.LoadFromFile("6.bf");
  proto::ProgramDesc pt_desc1(&table1);
  pt_desc1.Load();
  ProgramDesc nb_desc1(&pt_desc1);

  ASSERT_EQ(nb_desc1.Version(), 1);
  ASSERT_EQ(nb_desc1.BlocksSize(), 3);
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
