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

#include <iostream>
#include <string>

#include "../test_helper.h"
#include "framework/loader.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::GPU_CL> loader;
  //  ../../../test/models/googlenet
  //  ../../../test/models/mobilenet

  std::cout << " Begin load mobilenet " << std::endl;

  auto program = loader.Load(std::string(g_mobilenet_mul), true);

  std::cout << " End load mobilenet " << std::endl;

  std::cout << " Begin load YOLO " << std::endl;

  auto program1 = loader.Load(std::string(g_yolo_mul), true);

  std::cout << " End load YOLO " << std::endl;

  //  auto program = loader.Load(g_mobilenet_ssd, true);

  //  auto program = loader.Load(std::string(g_ocr) + "/model",
  //                             std::string(g_ocr) + "/params", false);
  //  program.originProgram->Description("program desc: ");

  return 0;
}
