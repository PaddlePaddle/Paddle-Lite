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

#include "../test_include.h"
#include "operators/lrn_op.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(g_googlenet);
  PADDLE_MOBILE_ENFORCE(program.originProgram != nullptr,
                        "program file read fail");

  Executor4Test<paddle_mobile::CPU,
                paddle_mobile::operators::LrnOp<paddle_mobile::CPU, float>>
      executor(program, "lrn");

  // 1. input_tensors;
  vector<Tensor> input_tensors;

  Tensor input1;
  auto input1_data = CreateInput<float>(&input1, {3, 4, 2, 2}, 0, 1);
  input_tensors.push_back(input1);

  // 2. input_names
  vector<string> input_names({
      "pool2d_0.tmp_0",
  });

  // 3. output_names
  vector<string> output_names({"pool1_norm1.tmp_1"});

  // 4. out_dims;
  vector<DDim> out_ddims;
  auto out_ddim = paddle_mobile::framework::make_ddim({3, 4, 2, 2});
  out_ddims.push_back(out_ddim);

  auto output = executor.Predict<LoDTensor>(input_tensors, input_names,
                                            output_names, out_ddims);

  auto output0_data = output[0]->data<float>();

  DLOG << " LrnOp input: ";
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int c = 0; c < 2; c++) {
        for (int d = 0; d < 2; d++) {
          DLOGF("%f ", input1_data[i * 16 + j * 4 + c * 2 + d]);
        }
        DLOGF("\n");
      }
      DLOGF("\n");
    }
    DLOGF("\n");
  }
  DLOG << " LrnOp output: ";
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int c = 0; c < 2; c++) {
        for (int d = 0; d < 2; d++) {
          DLOGF("%f ", output0_data[i * 16 + j * 4 + c * 2 + d]);
        }
        DLOGF("\n");
      }
      DLOGF("\n");
    }
    DLOGF("\n");
  }
  DLOG << input1_data[0] << " / ((1 + 0.00002 * ( " << input1_data[0] << "^2 + "
       << input1_data[4] << "^2 + " << input1_data[8] << "^2 ))^0.75) = ";
  DLOG << output0_data[0];
  return 0;
}
