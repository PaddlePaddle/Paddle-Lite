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

#include <fstream>
#include "../test_helper.h"
#include "../test_include.h"

int main() {
  paddle_mobile::Loader<paddle_mobile::CPU> loader;
  auto time1 = time();
  auto program = loader.Load(g_mobilenet, false);
  auto time2 = time();
  DLOG << "load cost :" << time_diff(time1, time1) << "ms";
  paddle_mobile::Executor<paddle_mobile::CPU> executor(program, 2, false);

  std::vector<int64_t> dims{2, 3, 224, 224};
  Tensor input_tensor;
  SetupTensor<float>(&input_tensor, {2, 3, 224, 224}, static_cast<float>(0),
                     static_cast<float>(1));

  std::vector<float> input(input_tensor.data<float>(),
                           input_tensor.data<float>() + input_tensor.numel());
  auto time3 = time();
  auto vec_result = executor.Predict(input, dims);
  float sum = 0;
  for (const auto item : vec_result) {
    sum += item;
  }
  DLOG << "mobilenet output sum =" << sum;
  auto time4 = time();
  DLOG << "predict cost :" << time_diff(time3, time4) << "ms";
  return 0;
}
