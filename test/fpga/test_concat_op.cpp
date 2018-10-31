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
#include "operators/concat_op.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::FPGA> loader;
  auto program = loader.Load(g_googlenet);
  PADDLE_MOBILE_ENFORCE(program.originProgram != nullptr,
                        "program file read fail");

  Executor4Test<paddle_mobile::FPGA,
                paddle_mobile::operators::ConcatOp<paddle_mobile::FPGA, float>>
      executor(program, "concat");

  // 1. input_tensors;
  vector<Tensor> input_tensors;

  Tensor input1;
  auto input1_data = CreateInput<float>(&input1, {4, 10, 2, 2}, 0, 1);
  input_tensors.push_back(input1);
  Tensor input2;
  auto input2_data = CreateInput<float>(&input2, {4, 20, 2, 2}, 0, 1);
  input_tensors.push_back(input2);
  Tensor input3;
  auto input3_data = CreateInput<float>(&input3, {4, 30, 2, 2}, 0, 1);
  input_tensors.push_back(input3);
  Tensor input4;
  auto input4_data = CreateInput<float>(&input4, {4, 40, 2, 2}, 0, 1);
  input_tensors.push_back(input4);
  // 2. input_names
  vector<string> input_names({
      "conv2d_3.tmp_1",
      "conv2d_5.tmp_1",
      "conv2d_7.tmp_1",
      "conv2d_8.tmp_1",
  });

  // 3. output_names
  vector<string> output_names({"concat_0.tmp_0"});

  // 4. out_dims;
  vector<DDim> out_ddims;
  auto out_ddim = paddle_mobile::framework::make_ddim({3, 100, 2, 2});
  out_ddims.push_back(out_ddim);

  auto output = executor.Predict<LoDTensor>(input_tensors, input_names,
                                            output_names, out_ddims);

  auto output0_data = output[0]->data<float>();

  // 5. test one example.
  int input_n = 1;
  int input_c = 2;
  int input_h = 0;
  int input_w = 1;
  int stride0 = input3.numel() / input3.dims()[0];
  int stride1 = input3.numel() / input3.dims()[0] / input3.dims()[1];
  int stride2 = input3.dims()[3];
  /// inputx1 (4,10,2,2),
  /// inputx2 (4,20,2,2),
  /// inputx3 (4,30,2,2),
  /// inputx4 (4,40,2,2),
  /// axis = 1
  /// output (4,100,2,2)
  int input_index =
      input_n * stride0 + input_c * stride1 + input_h * stride2 + input_w;
  int output_index = input_n * 100 * 2 * 2 +
                     (input_c + input1.dims()[1] + input2.dims()[1]) * 2 * 2 +
                     input_h * 2 + input_w;

  DLOG << " input3 [1, 2,0,1] = " << input3_data[input_index];
  DLOG << " output [1,32,0,1] = " << output0_data[output_index];
  return 0;
}
