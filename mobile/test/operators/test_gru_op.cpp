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
#include "../test_include.h"
#include "operators/gru_op.h"

namespace paddle_mobile {

template <typename Itype, typename Otype>
int TestGruOp(int in_channels, int out_channels, std::string opname) {
  size_t input_c = in_channels;
  size_t output_c = out_channels;
  paddle_mobile::framework::LoD lod{{0, input_c}};
  int batch_size = lod.size();
  framework::DDim input_shape = framework::make_ddim({input_c, output_c * 3});
  framework::DDim weight_shape = framework::make_ddim({output_c, output_c * 3});
  framework::DDim h0_shape = framework::make_ddim({batch_size, output_c});
  framework::DDim bias_shape = framework::make_ddim({batch_size, output_c * 3});

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["Input"] = std::vector<std::string>({"input"});
  inputs["Weight"] = std::vector<std::string>({"weight"});
  inputs["H0"] = std::vector<std::string>({"h0"});
  inputs["Bias"] = std::vector<std::string>({"bias"});

  outputs["BatchGate"] = std::vector<std::string>({"output_batch_gate"});
  outputs["BatchResetHiddenPrev"] =
      std::vector<std::string>({"output_batch_reset_hidden_prev"});
  outputs["BatchHidden"] = std::vector<std::string>({"output_batch_hidden"});
  outputs["Hidden"] = std::vector<std::string>({"output_hidden"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(input, input_shape, -127, 127);
  input->set_lod(lod);

  auto weight_var = scope.get()->Var("weight");
  auto weight = weight_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(weight, weight_shape, -127, 127);

  auto h0_var = scope.get()->Var("h0");
  auto h0 = h0_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(h0, h0_shape, -127, 127);

  auto bias_var = scope.get()->Var("bias");
  auto bias = bias_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(bias, bias_shape, -127, 127);

  auto batch_gate_var = scope.get()->Var("output_batch_gate");
  auto batch_reset_hidden_prev_var =
      scope.get()->Var("output_batch_reset_hidden_prev");
  auto batch_hidden_var = scope.get()->Var("output_batch_hidden");
  auto hidden_var = scope.get()->Var("output_hidden");

  framework::AttributeMap attrs;
  attrs["activation"].Set<std::string>(std::string("relu"));
  attrs["gate_activation"].Set<std::string>(std::string("sigmoid"));
  attrs["is_reverse"].Set<bool>(false);

  auto *op = new operators::GruOp<CPU, float>("gru", inputs, outputs, attrs,
                                              scope.get());
  op->InferShape();
  op->Init();
  for (int i = 0; i < 10; ++i) {
    op->Run();
  }
  auto time1 = time();
  for (int i = 0; i < 10; ++i) {
    op->Run();
  }
  auto time2 = time();
  std::ofstream out_file("./out_gru.txt", std::ios::app);
  out_file << opname << " cost :" << time_diff(time1, time2) / 10.0 << "ms"
           << std::endl;
  out_file.close();

  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main(int argc, char *argv[]) {
  paddle_mobile::TestGruOp<float, float>(384, 120, "gru_forward");
  return 0;
}
