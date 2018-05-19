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
#include "io.h"

int main() {
    paddle_mobile::Loader<paddle_mobile::CPU> loader;
    auto program = loader.Load(std::string("../models/googlenet"));
    if (program.originProgram == nullptr) {
        DLOG << "program read file";
    }

    Executor4Test<paddle_mobile::CPU,
                  paddle_mobile::operators::PoolOp<paddle_mobile::CPU, float>>
        executor(program, "pool2d");

    paddle_mobile::framework::Tensor input;
    SetupTensor<float>(&input, {1, 64, 112, 112}, static_cast<float>(0),
                       static_cast<float>(1));

    auto output = executor.predict(input, "conv2d_0.tmp_1", "pool2d_0.tmp_0",
                                   {1, 64, 56, 56});

    float *output_ptr = output->data<float>();
    for (int j = 0; j < output->numel(); ++j) {
        DLOG << " value of output: " << output_ptr[j];
    }
    return 0;
}
