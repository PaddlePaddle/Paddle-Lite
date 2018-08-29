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

#include <string>

#include "../test_helper.h"
#include "io/loader.h"

int main() {
  paddle_mobile::Loader<paddle_mobile::CPU> loader;
  //  ../../../test/models/googlenet
  //  ../../../test/models/mobilenet
  //  auto program = loader.Load(g_googlenet, true);
  //  auto program = loader.Load(g_mobilenet_ssd, true);
  auto params = loader.Load(g_nlp, true);
  params.originProgram->Description("program desc: ");
  //  auto program = loader.Load(std::string(g_nlp) + "/model",
  //                             std::string(g_nlp) + "/params", false);
  //  program.originProgram->Description("program desc: ");
  return 0;
}
