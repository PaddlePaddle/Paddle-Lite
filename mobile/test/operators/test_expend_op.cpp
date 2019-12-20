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

#ifdef PADDLE_MOBILE_CL
#include "../executor_for_test_opencl.h"
#include "operators/expand_op.h"
#include "operators/feed_op.h"
#ifdef EXPAND_OP

int main() {
  const int IN_N = 1;
  const int IN_C = 1;
  const int IN_H = 2;
  const int IN_W = 3;

  const int EXPEND_N = 1;
  const int EXPEND_C = 1;
  const int EXPEND_H = 2;
  const int EXPEND_W = 2;

  const int OUT_N = IN_N * EXPEND_N;
  const int OUT_C = IN_C * EXPEND_C;
  const int OUT_H = IN_H * EXPEND_H;
  const int OUT_W = IN_W * EXPEND_W;

  framework::DDim in_dims = framework::make_ddim({IN_N, IN_C, IN_H, IN_W});
  framework::DDim out_dims = framework::make_ddim({OUT_N, OUT_C, OUT_H, OUT_W});
  VariableNameMap inputs;
  VariableNameMap outputs;
  AttributeMap attrs;
  inputs["X"] = std::vector<std::string>({"op_in"});
  outputs["Out"] = std::vector<std::string>({"op_out"});

  std::vector<int> expand_times = {EXPEND_N, EXPEND_C, EXPEND_H, EXPEND_W};
  attrs["expand_times"].Set<std::vector<int>>(expand_times);

  OpenClOpTester<operators::ExpandOp<GPU_CL, float>> tester;
  tester.Predict("expend", in_dims, out_dims, inputs, outputs, attrs);
}
#endif
#endif
