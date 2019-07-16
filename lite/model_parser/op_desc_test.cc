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

#include "lite/model_parser/cpp/op_desc.h"
#include <gtest/gtest.h>
#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/naive_buffer/op_desc.h"
#include "lite/model_parser/pb/op_desc.h"

namespace paddle {
namespace lite {

using pt_desc_type = naive_buffer::proto::OpDesc;
using pt_table_type = naive_buffer::BinaryTable;

template <typename OpDescType>
void TestX(OpDescType* desc) {
  desc->SetInput("X", {"a", "b"});
  auto X = desc->Input("X");
  ASSERT_EQ(X.size(), 2UL);
  ASSERT_EQ(X[0], "a");
  ASSERT_EQ(X[1], "b");

  desc->SetOutput("Y", {"c", "d"});
  auto Y = desc->Output("Y");
  ASSERT_EQ(Y.size(), 2UL);
  ASSERT_EQ(Y[0], "c");
  ASSERT_EQ(Y[1], "d");

  desc->template SetAttr<int32_t>("aint", 100);
  ASSERT_TRUE(desc->HasAttr("aint"));
  ASSERT_FALSE(desc->HasAttr("afloat"));
  ASSERT_EQ(desc->template GetAttr<int32_t>("aint"), 100);
}

TEST(OpDesc, Basic) {
  // pb OpDesc
  pb::OpDesc pb_desc;
  TestX<pb::OpDesc>(&pb_desc);

  // cpp OpDesc
  cpp::OpDesc cpp_desc;
  TestX<cpp::OpDesc>(&cpp_desc);

  // naive buffer OpDesc
  pt_table_type table;
  pt_desc_type pt_desc(&table);
  naive_buffer::OpDesc nb_desc(&pt_desc);
  TestX<naive_buffer::OpDesc>(&nb_desc);
}

template <typename OpDescType>
void TestCppToAny(OpDescType* any_desc) {
  cpp::OpDesc desc;

  desc.SetInput("X", {"a", "b"});
  desc.SetOutput("Y", {"c", "d"});
  desc.template SetAttr<int32_t>("aint", 100);

  TransformOpDescCppToAny(desc, any_desc);
  {
    auto& desc = *any_desc;
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
}

TEST(OpDesc, CppToAny) {
  // pb OpDesc
  pb::OpDesc pb_desc;
  TestCppToAny<pb::OpDesc>(&pb_desc);

  // naive buffer OpDesc
  pt_table_type table;
  pt_desc_type pt_desc(&table);
  naive_buffer::OpDesc nb_desc(&pt_desc);
  TestCppToAny<naive_buffer::OpDesc>(&nb_desc);
}

template <typename OpDescType>
void TestAnyToCpp(OpDescType* desc) {
  desc->SetInput("X", {"a", "b"});
  desc->SetOutput("Y", {"c", "d"});
  desc->template SetAttr<int32_t>("aint", 100);

  cpp::OpDesc cpp_desc;

  TransformOpDescAnyToCpp(*desc, &cpp_desc);
  {
    auto& desc = cpp_desc;
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
}

TEST(OpDesc, AnyToCpp) {
  // pb OpDesc
  pb::OpDesc pb_desc;
  TestAnyToCpp<pb::OpDesc>(&pb_desc);

  // naive buffer OpDesc
  pt_table_type table;
  pt_desc_type pt_desc(&table);
  naive_buffer::OpDesc nb_desc(&pt_desc);
  TestAnyToCpp<naive_buffer::OpDesc>(&nb_desc);
}

}  // namespace lite
}  // namespace paddle
