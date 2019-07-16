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
#include "lite/model_parser/naive_buffer/op_desc.h"
#include "lite/model_parser/naive_buffer/param_desc.h"
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

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
