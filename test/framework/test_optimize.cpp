/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "../test_helper.h"
#include "framework/loader.h"
#include "framework/program/program-optimize/node.h"
#include "framework/program/program-optimize/program_optimize.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  //    "../../../test/models/googlenet"
  auto program = loader.Load(g_mobilenet_ssd, true);
  paddle_mobile::framework::ProgramOptimize optimize;
  //  program.originProgram->Description("origin");
  auto optimize_program = optimize.FusionOptimize(program.originProgram);
  if (optimize_program != nullptr) {
    //    optimize_program->Description("optimize");
  } else {
    LOG(paddle_mobile::kLOG_ERROR) << "optimize_program is null";
  }
  return 0;
}
