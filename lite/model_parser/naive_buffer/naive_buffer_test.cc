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

#include "lite/model_parser/naive_buffer/naive_buffer.h"
#include <gtest/gtest.h>

namespace paddle {
namespace lite {
namespace naive_buffer {

TEST(NaiveBuffer, primary) {
  BinaryTable table;
  PrimaryBuilder<int32_t> p0(&table);
  PrimaryBuilder<float> p1(&table);
  StringBuilder p2(&table);
  ASSERT_EQ(p0.type(), Type::INT32);
  ASSERT_EQ(p1.type(), Type::FLOAT32);
  ASSERT_EQ(p2.type(), Type::STRING);

  p0.set(2008);
  p0.Save();
  p1.set(2.008);
  p1.Save();
  p2.set("hello world");
  p2.Save();

  table.SaveToFile("1.bf");

  BinaryTable table1;
  table1.LoadFromFile("1.bf");
  PrimaryBuilder<int32_t> p0_load(&table1);
  PrimaryBuilder<float> p1_load(&table1);
  StringBuilder p2_load(&table1);

  p0_load.Load();
  p1_load.Load();
  p2_load.Load();

  ASSERT_EQ(p0_load.data(), 2008);
  EXPECT_NEAR(p1_load.data(), 2.008, 1e-5);
  ASSERT_EQ(p2_load.data(), "hello world");
}

// Message structure 0
class NBTestMsg0 : public StructBuilder {
 public:
  explicit NBTestMsg0(BinaryTable* table) : StructBuilder(table) {
    NewInt32("int0");
    NewInt32("int1");
    NewInt32("int2");
    NewInt32("float");
    NewStr("str0");
    NewStr("str1");
  }
};

using enum_builder = EnumBuilder<Type>;
// Message structure composed of NBTestMsg0
class NBTestMsg1 : public StructBuilder {
 public:
  explicit NBTestMsg1(BinaryTable* table) : StructBuilder(table) {
    NewInt32("int0");
    New<enum_builder>("enum0");
    New<NBTestMsg0>("msg0");
  }
};

int32_t int0 = 1222112;
int32_t int1 = 23232839;
int32_t int2 = 5431566;
float float0 = 233.23212;
const char* str0 = "sdfalfjasngasdghsadfjafas;fj";
const char* str1 = "sdlkfjasdfafjcsasafasskdfjh  fsadfsafj;fj";

void SetMsg0(NBTestMsg0* msg0) {
  msg0->GetMutableField<Int32Builder>("int0")->set(int0);
  msg0->GetMutableField<Int32Builder>("int1")->set(int1);
  msg0->GetMutableField<Int32Builder>("int2")->set(int2);
  msg0->GetMutableField<Float32Builder>("float")->set(float0);
  msg0->GetMutableField<StringBuilder>("str0")->set(str0);
  msg0->GetMutableField<StringBuilder>("str1")->set(str1);
  msg0->Save();
}

void TestMsg0(const NBTestMsg0& msg0) {
  ASSERT_EQ(msg0.GetField<Int32Builder>("int0").data(), int0);
  ASSERT_EQ(msg0.GetField<Int32Builder>("int1").data(), int1);
  ASSERT_EQ(msg0.GetField<Int32Builder>("int2").data(), int2);
  EXPECT_NEAR(msg0.GetField<Float32Builder>("float").data(), float0, 1e-5);
  ASSERT_EQ(msg0.GetField<StringBuilder>("str0").data(), str0);
  ASSERT_EQ(msg0.GetField<StringBuilder>("str1").data(), str1);
}

TEST(NBTestMsg, msg0) {
  BinaryTable table;
  NBTestMsg0 msg0(&table);
  SetMsg0(&msg0);

  // write the table
  table.SaveToFile("1.bf");

  // load the table
  BinaryTable table1;
  table1.LoadFromFile("1.bf");
  NBTestMsg0 msg1(&table1);
  msg1.Load();
  TestMsg0(msg1);
}

TEST(NBTestMsg, msg1) {
  BinaryTable table;
  NBTestMsg1 msg(&table);

  auto* int0 = msg.GetMutableField<Int32Builder>("int0");
  auto* enum0 = msg.GetMutableField<enum_builder>("enum0");
  auto* msg0 = msg.GetMutableField<NBTestMsg0>("msg0");

  int0->set(2008);
  int0->Save();

  enum0->set(Type::INT64);
  enum0->Save();

  SetMsg0(msg0);

  table.SaveToFile("1.bf");

  BinaryTable table1;
  NBTestMsg1 msg1(&table1);
  table1.LoadFromFile("1.bf");

  msg1.Load();

  ASSERT_EQ(msg.GetField<Int32Builder>("int0").data(), 2008);
  ASSERT_EQ(msg.GetField<enum_builder>("enum0").data(), Type::INT64);
  TestMsg0(msg1.GetField<NBTestMsg0>("msg0"));
}

TEST(ListBuilder, basic) {
  BinaryTable table;
  ListBuilder<StringBuilder> li(&table);

  const int num_elems = 101;

  for (int i = 0; i < num_elems; i++) {
    auto* elem = li.New();
    elem->set("elem-" + paddle::lite::to_string(i));
  }
  li.Save();
  table.SaveToFile("2.bf");
  LOG(INFO) << "table.size " << table.size();

  BinaryTable table1;
  table1.LoadFromFile("2.bf");
  ASSERT_EQ(table1.size(), table.size());

  ListBuilder<StringBuilder> li1(&table1);
  li1.Load();

  for (int i = 0; i < num_elems; i++) {
    ASSERT_EQ(li1.Get(i).data(), "elem-" + paddle::lite::to_string(i));
  }
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
