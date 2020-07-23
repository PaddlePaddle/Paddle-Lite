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

#include "lite/model_parser/flatbuffers/vector_view.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "lite/model_parser/flatbuffers/framework_generated.h"

namespace paddle {
namespace lite {

TEST(VectorView, std_vector) {
  std::vector<int64_t> vector{1, 2, 3};
  VectorView<int64_t, Standard> vector_view(&vector);
  size_t i = 0;
  for (const auto& value : vector_view) {
    EXPECT_EQ(value, vector[i]);
    ++i;
  }
  for (size_t j = 0; j < vector_view.size(); ++j) {
    EXPECT_EQ(vector_view[i], vector[i]);
  }
}

TEST(VectorView, Flatbuffers) {
  using namespace flatbuffers;        // NOLINT
  using namespace paddle::lite::fbs;  // NOLINT

  auto create_desc = [](FlatBufferBuilder& fbb) {
    /* --------- Set --------- */
    // Attr
    std::vector<int32_t> ints({-1, 0, 1, 2, 3});
    auto string_0 = fbb.CreateString("string_0");
    auto string_1 = fbb.CreateString("string_1");
    std::vector<Offset<String>> strings;
    strings.push_back(string_0);
    strings.push_back(string_1);
    auto attr = proto::OpDesc_::CreateAttrDirect(fbb,
                                                 nullptr,
                                                 proto::AttrType_INT,
                                                 0,
                                                 0.0f,
                                                 nullptr,
                                                 &ints,
                                                 nullptr,
                                                 &strings);

    // OpDesc
    std::vector<Offset<proto::OpDesc_::Attr>> attrs;
    attrs.push_back(attr);
    auto op_desc =
        proto::CreateOpDescDirect(fbb, "hello!", nullptr, nullptr, &attrs);

    // BlockDesc 0
    std::vector<Offset<proto::OpDesc>> ops;
    ops.push_back(op_desc);
    auto block_0 = proto::CreateBlockDescDirect(fbb, 0, 0, nullptr, &ops);

    // BlockDesc 1
    auto block_1 = proto::CreateBlockDescDirect(fbb, 1);

    // ProgramDesc
    std::vector<Offset<proto::BlockDesc>> block_vector;
    block_vector.push_back(block_0);
    block_vector.push_back(block_1);
    auto orc = proto::CreateProgramDescDirect(fbb, &block_vector);
    fbb.Finish(orc);
  };

  FlatBufferBuilder fbb;
  create_desc(fbb);
  auto program = fbs::proto::GetProgramDesc(fbb.GetBufferPointer());

  // BlockDesc View
  VectorView<proto::BlockDesc*> block_view(program->blocks());
  EXPECT_EQ(block_view.size(), static_cast<size_t>(2));
  EXPECT_EQ(block_view[0]->idx(), 0);
  EXPECT_EQ(block_view[1]->idx(), 1);

  // OpDesc & Attr View
  VectorView<proto::OpDesc*> op_view(block_view[0]->ops());
  EXPECT_EQ(op_view[0]->type()->str(), std::string("hello!"));
  VectorView<proto::OpDesc_::Attr*> attr_view(op_view[0]->attrs());

  // int32_t View
  VectorView<int32_t> ints_view(attr_view[0]->ints());
  std::vector<int32_t> ints({-1, 0, 1, 2, 3});
  size_t cnt_0 = 0;
  for (const auto& i : ints_view) {
    EXPECT_EQ(i, ints[cnt_0]);
    ++cnt_0;
  }
  for (size_t i = 0; i < ints_view.size(); ++i) {
    EXPECT_EQ(ints_view[i], ints[i]);
  }
  std::vector<int32_t> ints_2(ints_view);
  for (size_t i = 0; i < ints_2.size(); ++i) {
    EXPECT_EQ(ints_2[i], ints[i]);
  }

  // String View
  VectorView<std::string> strings_view(attr_view[0]->strings());
  std::vector<std::string> strings({"string_0", "string_1"});
  EXPECT_EQ(strings_view.size(), strings.size());
  size_t cnt_1 = 0;
  for (const auto& s : strings_view) {
    EXPECT_EQ(s, strings[cnt_1]);
    ++cnt_1;
  }
  for (size_t i = 0; i < strings_view.size(); ++i) {
    EXPECT_EQ(strings_view[i], strings[i]);
  }
  std::vector<std::string> string_2(strings_view);
  for (size_t i = 0; i < string_2.size(); ++i) {
    EXPECT_EQ(string_2[i], strings[i]);
  }
}

}  // namespace lite
}  // namespace paddle
