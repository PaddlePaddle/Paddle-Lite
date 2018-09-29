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
#include "operators/mul_op.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(g_resnet);
  PADDLE_MOBILE_ENFORCE(program.originProgram != nullptr,
                        "program file read fail");

  Executor4Test<paddle_mobile::CPU,
                paddle_mobile::operators::MulOp<paddle_mobile::CPU, float>>
      executor(program, "mul");

  // 1. input_tensors;
  vector<Tensor> input_tensors;

  Tensor input1;
  auto input1_data = CreateInput<float>(&input1, {3, 2, 1, 1}, 0, 1);
  input_tensors.push_back(input1);
  Tensor input2;
  auto input2_data = CreateInput<float>(&input2, {2, 3}, 0, 1);
  input_tensors.push_back(input2);

  // 2. input_names
  vector<string> input_names({
      "pool2d_0.tmp_0",
      "fc_0.w_0",
  });

  // 3. output_names
  vector<string> output_names({"fc_0.tmp_0"});

  // 4. out_dims;
  vector<DDim> out_ddims;
  auto out_ddim = paddle_mobile::framework::make_ddim({3, 3});
  out_ddims.push_back(out_ddim);

  auto output = executor.Predict<LoDTensor>(input_tensors, input_names,
                                            output_names, out_ddims);

  auto output0_data = output[0]->data<float>();

  auto dim_1 = input1.numel() / input1.dims()[0];
  DLOG << " input1 : ";
  for (int i = 0; i < input1.dims()[0]; ++i) {
    for (int j = 0; j < dim_1; ++j) {
      DLOGF("%f ", input1_data[i * dim_1 + j]);
    }
    DLOGF("\n");
  }

  auto dim_2 = input2.numel() / input2.dims()[0];
  DLOG << " input2 : ";
  for (int i = 0; i < input2.dims()[0]; ++i) {
    for (int j = 0; j < dim_2; ++j) {
      DLOGF("%f ", input2_data[i * dim_2 + j]);
    }
    DLOGF("\n");
  }

  auto dim_output0 = output[0]->numel() / output[0]->dims()[0];
  DLOG << " output : ";
  for (int i = 0; i < output[0]->dims()[0]; ++i) {
    for (int j = 0; j < dim_output0; ++j) {
      DLOGF("%f ", output0_data[i * dim_2 + j]);
    }
    DLOGF("\n");
  }

  /// output (3,3)
  DLOG << "output memory size : " << output[0]->memory_size();
  DLOG << "output numel : " << output[0]->numel();

  DLOG << input1_data[0] << " x " << input2_data[0] << " + " << input1_data[1]
       << " x " << input2_data[0 + 3] << " = " << output0_data[0];
  return 0;
}
