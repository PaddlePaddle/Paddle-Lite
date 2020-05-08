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
#include "../../src/common/types.h"
#include "../test_helper.h"
#include "../test_include.h"

int main(int argc, char **argv) {
  // init input args
  string model_dir = g_mobilenet;
  int64_t N = 1;
  int64_t C = 3;
  int64_t H = 224;
  int64_t W = 224;
  int repeats = 10;
  int warmup = 10;
  int print_output_elem = 0;

  std::cout << "argc:" << argc << std::endl;
  if (argc > 1 && argc < 9) {
    std::cout << "usage:" << argv[0] << "\n"
              << " <model_dir>\n"
              << " <input_n>\n"
              << " <input_c>\n"
              << " <input_h>\n"
              << " <input_w>\n"
              << " <repeats>\n"
              << " <warmup>\n"
              << " <print_output>" << std::endl;
    return 0;
  }

  if (argc >= 9) {
    model_dir = argv[1];
    N = atoi(argv[2]);
    C = atoi(argv[3]);
    H = atoi(argv[4]);
    W = atoi(argv[5]);
    repeats = atoi(argv[6]);
    warmup = atoi(argv[7]);
    print_output_elem = atoi(argv[8]);
  }

  std::cout << "input shape(NCHW):" << N << " " << C << " " << H << " " << W
            << std::endl;
  std::cout << "repeats:" << repeats << std::endl;
  std::cout << "model_dir:" << model_dir << std::endl;

  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile;
  //    paddle_mobile.SetThreadNum(4);
  auto load_start = paddle_mobile::time();
#ifdef PADDLE_MOBILE_CL
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#endif

  auto load_model_status = paddle_mobile.Load(std::string(model_dir), true);
  if (!load_model_status) {
    std::cout << "failed to load model from:" << model_dir << std::endl;
    return 0;
  }

  auto load_end = paddle_mobile::time();
  std::cout << "load cost:" << paddle_mobile::time_diff(load_start, load_end)
            << " ms" << std::endl;

  // input tensor
  std::vector<float> input;
  std::vector<int64_t> dims{N, C, H, W};
  GetInput<float>(g_test_image_1x3x224x224_banana, &input, dims);

  // warmup
  std::vector<float> vec_result = paddle_mobile.Predict(input, dims);
  for (int widx = 0; widx < warmup; ++widx) {
    paddle_mobile.Predict(input, dims);
  }

  // benchmark
  float sum_duration = 0.0f;
  float min_duration = 1e5f;
  float max_duration = 1e-5f;
  float ave_duration = -1;
  for (int ridx = 0; ridx < repeats; ++ridx) {
    auto start = paddle_mobile::time();
    vec_result = paddle_mobile.Predict(input, dims);
    auto end = paddle_mobile::time();
    auto duration = paddle_mobile::time_diff(start, end);
    sum_duration += duration;
    min_duration = (duration > min_duration) ? min_duration : duration;
    max_duration = (duration < max_duration) ? max_duration : duration;
    std::cout << "ridx:" << ridx + 1 << "/" << repeats << " " << duration
              << " ms" << std::endl;
  }

  // benchmark result
  ave_duration = sum_duration / static_cast<float>(repeats);

  // output result
  float output_sum = 0;
  float output_ave = -1;
  for (size_t oidx = 0; oidx < vec_result.size(); ++oidx) {
    output_sum += vec_result[oidx];
    if (print_output_elem) {
      std::cout << "out_idx:" << oidx << " " << vec_result[oidx] << std::endl;
    }
  }
  output_ave = output_sum / static_cast<float>(vec_result.size());
  std::vector<float>::iterator biggest =
      std::max_element(std::begin(vec_result), std::end(vec_result));

  // summary
  std::cout << "===== predict benchmark ====" << std::endl
            << "run repeats:" << repeats << std::endl
            << "sum_duration:" << sum_duration << " ms" << std::endl
            << "ave_duration:" << ave_duration << " ms" << std::endl
            << "max_duration:" << max_duration << " ms" << std::endl
            << "min_duration:" << min_duration << " ms" << std::endl
            << "\n===== predict result ====" << std::endl
            << "output_sum:" << output_sum << std::endl
            << "output_ave:" << output_ave << std::endl
            << "output_size:" << vec_result.size() << std::endl
            << "Max element is " << *biggest << " at position "
            << std::distance(std::begin(vec_result), biggest) << std::endl
            << "Note: 如果结果Nan请查看:"
               " test/images/g_test_image_1x3x224x224_banana "
               "是否存在?"
            << std::endl;
  return 0;
}
