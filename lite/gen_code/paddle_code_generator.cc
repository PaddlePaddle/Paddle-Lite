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

#include <gflags/gflags.h>
#include "lite/gen_code/gen_code.h"
#include "lite/model_parser/model_parser.h"
#include "lite/model_parser/pb/program_desc.h"

DEFINE_string(optimized_model, "", "");
DEFINE_string(generated_code_file, "__generated_code__.cc", "");

namespace paddle {
namespace lite {
namespace gencode {

void GenCode(const std::string& model_dir, const std::string& out_file) {
  lite::Scope scope;
  cpp::ProgramDesc cpp_desc;
  std::string model_file = model_dir + "/model";
  std::string param_file = model_dir + "/params";
  LoadModelPb(model_dir, model_file, param_file, &scope, &cpp_desc);

  framework::proto::ProgramDesc pb_proto_desc;
  lite::pb::ProgramDesc pb_desc(&pb_proto_desc);
  TransformProgramDescCppToAny(cpp_desc, &pb_desc);

  ProgramCodeGenerator codegen(pb_proto_desc, scope);

  std::ofstream file(out_file);

  file << codegen.GenCode();

  file.close();
}

}  // namespace gencode
}  // namespace lite
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  paddle::lite::gencode::GenCode(FLAGS_optimized_model,
                                 FLAGS_generated_code_file);
  return 0;
}
