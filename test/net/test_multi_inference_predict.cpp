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

#include <iostream>
#include <thread>  // NOLINT
#include "../test_helper.h"
#include "../test_include.h"

void fun_yolo();
int fun_mobilenet();
int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile2;

  //  fun_yolo();
  //  fun_mobilenet();

  std::thread t1(fun_yolo);
  std::thread t2(fun_mobilenet);

  t1.join();
  t2.join();

  return 0;
}

void fun_yolo() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(4);
  //  ../../../test/models/googlenet
  //  ../../../test/models/mobilenet
  auto time1 = time();
  if (paddle_mobile.Load(g_yolo, true)) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time1) << "ms" << std::endl;

    vector<int64_t> dims{1, 3, 227, 227};
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 227, 227}, static_cast<float>(0),
                       static_cast<float>(1));

    vector<float> input(input_tensor.data<float>(),
                        input_tensor.data<float>() + input_tensor.numel());

    auto time3 = time();
    for (int i = 0; i < 10; ++i) {
      paddle_mobile.Predict(input, dims);
    }
    auto time4 = time();
    std::cout << "thread 1:   predict cost :" << time_diff(time3, time4) / 10
              << "ms" << std::endl;
  }
}

int fun_mobilenet() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(4);
  auto time1 = time();
  //  auto isok = paddle_mobile.Load(std::string(g_mobilenet_detect) + "/model",
  //                     std::string(g_mobilenet_detect) + "/params", true);

  auto isok = paddle_mobile.Load(g_mobilenet, true);
  if (isok) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time1) << "ms" << std::endl;

    vector<float> input;
    vector<int64_t> dims{1, 3, 224, 224};
    GetInput<float>(g_test_image_1x3x224x224_banana, &input, dims);

    auto vec_result = paddle_mobile.Predict(input, dims);
    auto biggest = max_element(begin(vec_result), end(vec_result));
    std::cout << " Max element is " << *biggest << " at position "
              << distance(begin(vec_result), biggest) << std::endl;

    // 预热十次
    for (int i = 0; i < 10; ++i) {
      auto vec_result = paddle_mobile.Predict(input, dims);
    }
    auto time3 = time();
    for (int i = 0; i < 10; ++i) {
      auto vec_result = paddle_mobile.Predict(input, dims);
    }
    DLOG << vec_result;
    auto time4 = time();
    std::cout << "thread 2:  predict cost :" << time_diff(time3, time4) / 10
              << "ms" << std::endl;
  }

  std::cout << "如果结果Nan请查看: test/images/g_test_image_1x3x224x224_banana "
               "是否存在?"
            << std::endl;
  return 0;
}
