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

#include "lite/model_parser/compatibility.h"
#include <gtest/gtest.h>
#include "lite/api/paddle_lite_factory_helper.h"

#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/cpp_desc.h"

USE_LITE_KERNEL(leaky_relu, kCUDA, kFloat, kNCHW, def);

namespace paddle {
namespace lite {

static constexpr int64_t version = 1005000;

TEST(CompatibleChecker, CppProgramDesc) {
  cpp::ProgramDesc program;
  program.SetVersion(version);
  auto* block = program.AddBlock<cpp::BlockDesc>();
  auto* op = block->AddOp<cpp::OpDesc>();
  op->SetType("leaky_relu");

  CompatibleChecker<cpp::ProgramDesc> checker(program);
  lite_api::Place place{TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW)};
  CHECK(checker(place));
}

}  // namespace lite
}  // namespace paddle
