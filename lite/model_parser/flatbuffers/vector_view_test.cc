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
  fbs::VectorView<int64_t, fbs::Standand> vector_view(&vector);
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
  int32_t idx = 0;
  flatbuffers::FlatBufferBuilder fbb;
  fbs::proto::BlockDescBuilder block_builder_1(fbb);
  block_builder_1.add_idx(idx);
  auto block_1 = block_builder_1.Finish();
  fbs::proto::BlockDescBuilder block_builder_2(fbb);
  block_builder_2.add_idx(idx);
  auto block_2 = block_builder_2.Finish();
  std::vector<flatbuffers::Offset<paddle::lite::fbs::proto::BlockDesc>>
      block_vector;
  block_vector.push_back(block_1);
  block_vector.push_back(block_2);
  auto orc = fbs::proto::CreateProgramDescDirect(fbb, &block_vector);
  fbb.Finish(orc);
  auto program = fbs::proto::GetProgramDesc(fbb.GetBufferPointer());
  fbs::VectorView<paddle::lite::fbs::proto::BlockDesc> vector_view(
      program->blocks());
  EXPECT_EQ(vector_view.size(), static_cast<size_t>(2));
}

}  // namespace lite
}  // namespace paddle
