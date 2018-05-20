/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "../framework/executor_for_test.h"
#include "../test_helper.h"
#include "./io.h"

int main() {
  paddle_mobile::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string("../models/googlenet"));
  if (program.originProgram == nullptr) {
    DLOG << "program file read fail";
  }

  Executor4Test<paddle_mobile::CPU,
                paddle_mobile::operators::ConvOp<paddle_mobile::CPU, float>>
      executor(program, "conv2d");

  paddle_mobile::framework::Tensor input;
  SetupTensor<float>(&input, {1, 3, 32, 32}, static_cast<float>(0),
                     static_cast<float>(1));

  auto output =
      executor.predict(input, "data", "conv2d_0.tmp_0", {1, 64, 56, 56});

  auto output_ptr = output->data<float>();
  for (int j = 0; j < output->numel(); ++j) {
    DLOG << " value of output: " << output_ptr[j];
  }
  return 0;
}
