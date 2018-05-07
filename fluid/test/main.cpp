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

#include "io.h"
#include "common/types.h"
#include "test_helper.h"
//
//template <typename T>
//void SetupTensor(paddle::framework::LoDTensor* input,
//                 paddle::framework::DDim dims, T lower, T upper) {
//    static unsigned int seed = 100;
//    std::mt19937 rng(seed++);
//    std::uniform_real_distribution<double> uniform_dist(0, 1);
//
//    T* input_ptr = input->mutable_data<T>(dims, paddle::platform::CPUPlace());
//    for (int i = 0; i < input->numel(); ++i) {
//        input_ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
//    }
//}

int main(){
    std::string data_set = "cifar10";
//
//    if (data_set == "cifar10") {
//        SetupTensor<float>(&input, {FLAGS_batch_size, 3, 32, 32},
//                           static_cast<float>(0), static_cast<float>(1));
//    } else if (data_set == "imagenet") {
//        SetupTensor<float>(&input, {FLAGS_batch_size, 3, 224, 224},
//                           static_cast<float>(0), static_cast<float>(1));
//    } else {
//        LOG(FATAL) << "Only cifar10 or imagenet is supported.";
//    }

    paddle_mobile::Loader<paddle_mobile::ARM> loader;
    loader.Load(std::string("../test/models/image_classification_resnet.inference.model"));
    paddle_mobile::framework::Tensor input;
//    SetupTensor<float>(&input, {1, 3, 32, 32},
//                       static_cast<float>(0), static_cast<float>(1));
    return 0;
}

