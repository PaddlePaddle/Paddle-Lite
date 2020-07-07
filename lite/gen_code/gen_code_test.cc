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

#include "lite/gen_code/gen_code.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/context.h"
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/cpp_desc.h"
#include "lite/model_parser/model_parser.h"
#include "lite/model_parser/pb/program_desc.h"

DEFINE_string(optimized_model, "", "");
DEFINE_string(generated_code_file, "__generated_code__.cc", "");

namespace paddle {
namespace lite {
namespace gencode {

// Manually construct a program.
TEST(gen_code, manual) {
  // For holding the weights.
  lite::Scope scope;
  // For holding the temporary variables.
  auto &tmp_scope = scope.NewScope();

  // Create weight variables.
  auto *w0 = scope.Var("w0")->GetMutable<lite::Tensor>();
  // Create temporary variables.
  auto *a = tmp_scope.Var("x")->GetMutable<lite::Tensor>();
  tmp_scope.Var("out")->GetMutable<lite::Tensor>();

  // Set weights.
  std::vector<float> w0_data({0, 1, 2, 3});
  std::vector<float> a_data({0, 1, 2, 3});
#ifdef LITE_WITH_ARM
  w0->Assign<float, lite::DDim, TARGET(kARM)>(
      w0_data.data(), lite::DDim{std::vector<int64_t>({2, 2})});
  a->Assign<float, lite::DDim, TARGET(kARM)>(
      a_data.data(), lite::DDim{std::vector<int64_t>({2, 2})});
#else
  w0->Assign<float, lite::DDim, TARGET(kX86)>(
      w0_data.data(), lite::DDim{std::vector<int64_t>({2, 2})});
  a->Assign<float, lite::DDim, TARGET(kX86)>(
      a_data.data(), lite::DDim{std::vector<int64_t>({2, 2})});
#endif

  std::vector<Place> valid_places({
#ifdef LITE_WITH_ARM
      Place{TARGET(kARM), PRECISION(kFloat)},
#else
      Place{TARGET(kX86), PRECISION(kFloat)},
#endif
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kHost), PRECISION(kAny)},
  });
  auto mul_op = LiteOpRegistry::Global().Create("mul");
  cpp::OpDesc mul_op_desc;
  mul_op_desc.SetType("mul");
  mul_op_desc.SetInput("X", {"x"});
  mul_op_desc.SetInput("Y", {"w0"});
  mul_op_desc.SetAttr("x_num_col_dims", 1);
  mul_op_desc.SetAttr("y_num_col_dims", 1);
  mul_op_desc.SetOutput("Out", {"out"});

  mul_op->Attach(mul_op_desc, &tmp_scope);
  auto mul_kernel = std::move(mul_op->CreateKernels(valid_places).front());
#ifdef LITE_WITH_ARM
  auto fc_ctx = ContextScheduler::Global().NewContext(TARGET(kARM));
#else
  auto fc_ctx = ContextScheduler::Global().NewContext(TARGET(kX86));
#endif
  mul_op->CheckShape();
  mul_op->InferShape();
  mul_kernel->SetContext(std::move(fc_ctx));
  mul_kernel->Launch();
}

TEST(gen_code, auto_gen) {
  std::vector<float> w0_data({0, 1, 2, 3});
  TensorRepr w0(PRECISION(kFloat),
                std::vector<int64_t>({2, 2}),
                w0_data.data(),
                w0_data.size() * sizeof(float));

  std::vector<float> w1_data({0.01, 1.2, 2.3, 3.4, 1.1, 2.2});
  TensorRepr w1(PRECISION(kFloat),
                std::vector<int64_t>({3, 2}),
                w1_data.data(),
                w1_data.size() * sizeof(float));

  cpp::OpDesc op0;
  op0.SetType("mul");
  op0.SetInput("X", {"a", "b"});
  op0.SetOutput("Out", {"out0"});
  op0.SetAttr<std::string>("desc", "this is a desc");
  op0.SetAttr<int>("x_col", 1);
  op0.SetAttr<int>("y_col", 2);
#ifdef LITE_WITH_ARM
  op0.SetAttr<std::string>(kKernelTypeAttr, "arm");
#else
  op0.SetAttr<std::string>(kKernelTypeAttr, "x86");
#endif

  gencode::Module module;
  module.AddHeaderIncludeGenCode();

  module.AddNamespaceBegin();
  module.AddInitFuncBegin();

  module.AddMemberCast();

  module.AddWeight("w0", w0);
  module.AddWeight("w1", w1);
  module.AddTmpVar("a");
  module.AddTmpVar("b");

  module.AddOp(op0);

  module.AddInitFuncEnd();
  module.AddNamespaceEnd();

  LOG(INFO) << module.stream().str();
}

TEST(gen_code, optimized_program) {
  lite::Scope scope;
  cpp::ProgramDesc cpp_desc;
  std::string model_file = FLAGS_optimized_model + "/model";
  std::string param_file = FLAGS_optimized_model + "/params";
  LoadModelPb(
      FLAGS_optimized_model, model_file, param_file, &scope, &cpp_desc, true);

  framework::proto::ProgramDesc pb_proto_desc;
  lite::pb::ProgramDesc pb_desc(&pb_proto_desc);
  TransformProgramDescCppToAny(cpp_desc, &pb_desc);

  ProgramCodeGenerator codegen(pb_proto_desc, scope);

  std::ofstream file(FLAGS_generated_code_file);

  file << codegen.GenCode();

  file.close();
}

}  // namespace gencode
}  // namespace lite
}  // namespace paddle
