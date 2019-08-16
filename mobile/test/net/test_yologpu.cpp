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
#include "../../src/common/types.h"
#include "../../src/io/paddle_test_inference_api.h"
#include "../test_helper.h"
#include "../test_include.h"
void t1() {
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile_gpu;
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile_cpu;
  paddle_mobile::PaddleTester<paddle_mobile::CPU> paddle_test_cpu;
  paddle_mobile::PaddleTester<paddle_mobile::GPU_CL> paddle_test_gpu;
  printf("cpu time:%f\n", paddle_test_cpu.CaculatePredictTime());
  std::string path = "/data/local/tmp/bin";
  printf("gpu time:%f\n", paddle_test_gpu.CaculatePredictTime(&path));
  //    paddle_mobile.SetThreadNum(4);
#ifdef PADDLE_MOBILE_CL
  paddle_mobile_gpu.SetCLPath("/data/local/tmp/bin");
#endif
  auto time1 = paddle_mobile::time();
  auto isok =
      paddle_mobile_gpu.Load(std::string(g_yolo_vision) + "/model",
                             std::string(g_yolo_vision) + "/params", true);

  //  auto isok = paddle_mobile.Load(std::string(g_yolo_mul), true);
  if (isok) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time2) << "ms"
              << std::endl;

    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 416, 416};
    GetInput<float>(g_test_image_1x3x416x416_vision_yolo_input, &input, dims);

    std::vector<float> vec_result;
    //            = paddle_mobile.Predict(input, dims);

    auto time3 = paddle_mobile::time();
    int max = 1;
    for (int i = 0; i < max; ++i) {
      vec_result = paddle_mobile_gpu.Predict(input, dims);
    }
    auto time4 = paddle_mobile::time();

    //    auto time3 = paddle_mobile::time();

    //    for (int i = 0; i < 10; ++i) {
    //      auto vec_result = paddle_mobile.Predict(input, dims);
    //    }

    //    auto time4 = paddle_mobile::time();

    std::cout << "predict cost :"
              << paddle_mobile::time_diff(time3, time4) / max << "ms"
              << std::endl;
    std::vector<float>::iterator biggest =
        std::max_element(std::begin(vec_result), std::end(vec_result));
    std::cout << " Max element is " << *biggest << " at position "
              << std::distance(std::begin(vec_result), biggest) << std::endl;
    //        for (float i : vec_result) {
    //            std::cout << i << std::endl;
    //        }
  }
}

void t2() {
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile;
  //    paddle_mobile.SetThreadNum(4);
#ifdef PADDLE_MOBILE_CL
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#endif
  auto time1 = paddle_mobile::time();
  auto isok = paddle_mobile.Load(std::string(g_yolo_mul) + "/model",
                                 std::string(g_yolo_mul) + "/params", true);

  //  auto isok = paddle_mobile.Load(std::string(g_yolo_mul), true);
  if (isok) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time2) << "ms"
              << std::endl;

    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 416, 416};
    GetInput<float>(g_yolo_img, &input, dims);

    std::vector<float> vec_result;
    //            = paddle_mobile.Predict(input, dims);

    auto time3 = paddle_mobile::time();
    int max = 10;
    for (int i = 0; i < max; ++i) {
      vec_result = paddle_mobile.Predict(input, dims);
    }
    auto time4 = paddle_mobile::time();

    //    auto time3 = paddle_mobile::time();

    //    for (int i = 0; i < 10; ++i) {
    //      auto vec_result = paddle_mobile.Predict(input, dims);
    //    }

    //    auto time4 = paddle_mobile::time();

    std::cout << "predict cost :"
              << paddle_mobile::time_diff(time3, time4) / max << "ms"
              << std::endl;
    std::vector<float>::iterator biggest =
        std::max_element(std::begin(vec_result), std::end(vec_result));
    std::cout << " Max element is " << *biggest << " at position "
              << std::distance(std::begin(vec_result), biggest) << std::endl;
    //        for (float i : vec_result) {
    //            std::cout << i << std::endl;
    //        }
  }
}

void t3() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  //    paddle_mobile.SetThreadNum(4);
  // #ifdef PADDLE_MOBILE_CL
  //  paddle_mobile.SetCLPath("/data/local/tmp/bin");
  // #endif
  auto time1 = paddle_mobile::time();
  auto isok = paddle_mobile.Load(std::string(g_yolo_mul) + "/model",
                                 std::string(g_yolo_mul) + "/params", true);

  //  auto isok = paddle_mobile.Load(std::string(g_yolo_mul), true);
  if (isok) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time2) << "ms"
              << std::endl;

    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 416, 416};
    GetInput<float>(g_yolo_img, &input, dims);

    std::vector<float> vec_result = paddle_mobile.Predict(input, dims);

    auto time3 = paddle_mobile::time();
    int max = 10;
    for (int i = 0; i < max; ++i) {
      vec_result = paddle_mobile.Predict(input, dims);
    }
    auto time4 = paddle_mobile::time();

    //    auto time3 = paddle_mobile::time();

    //    for (int i = 0; i < 10; ++i) {
    //      auto vec_result = paddle_mobile.Predict(input, dims);
    //    }

    //    auto time4 = paddle_mobile::time();

    std::cout << "predict cost :"
              << paddle_mobile::time_diff(time3, time4) / max << "ms"
              << std::endl;
    std::vector<float>::iterator biggest =
        std::max_element(std::begin(vec_result), std::end(vec_result));
    std::cout << " Max element is " << *biggest << " at position "
              << std::distance(std::begin(vec_result), biggest) << std::endl;
    //        for (float i : vec_result) {
    //            std::cout << i << std::endl;
    //        }
  }
}

int main() {
  //  std::thread th1(t1);
  //      std::thread th2(t2);
  //  std::thread th3(t3);
  std::thread th1(t1);
  //  th1.join();
  //      th2.join();
  //  th3.join();
  th1.join();
  return 0;
}
