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

#include "../executor_for_test.h"
#include "../test_include.h"
#include "operators/prelu_op.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(g_resnet);
  PADDLE_MOBILE_ENFORCE(program.originProgram != nullptr,
                        "program file read fail");

  Executor4Test<paddle_mobile::CPU,
                paddle_mobile::operators::PReluOp<paddle_mobile::CPU, float>>
      executor(program, "prelu");

  // 1. input_tensors;
  vector<Tensor> input_tensors;

  Tensor input1;
  auto input1_data = CreateInput<float>(&input1, {1, 2, 3, 4}, -1, 1);
  input_tensors.push_back(input1);

  // 2. input_names
  vector<string> input_names({
      "batch_norm_0.tmp_2",
  });

  // 3. output_names
  vector<string> output_names({"batch_norm_0.tmp_3"});

  // 4. out_dims;
  vector<DDim> out_ddims;
  auto out_ddim = paddle_mobile::framework::make_ddim({1, 2, 3, 4});
  out_ddims.push_back(out_ddim);

  auto output = executor.Predict<LoDTensor>(input_tensors, input_names,
                                            output_names, out_ddims);

  auto output0_data = output[0]->data<float>();

  for (int j = 0; j < output[0]->numel(); ++j) {
    DLOG << " value of output: " << output0_data[j];
  }
  return 0;
}
