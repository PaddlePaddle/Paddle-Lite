// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <sys/time.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "lac.h"  // NOLINT

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0] << " model_path conf_path input_path "
              << "label_path test_num" << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  std::string conf_path = argv[2];
  std::string input_path = argv[3];
  std::string label_path = argv[4];
  int test_num = atoi(argv[5]);
  int threads = 1;
  std::cout << "model_path:" << model_path << std::endl;
  std::cout << "conf_path:" << conf_path << std::endl;
  std::cout << "input_path:" << input_path << std::endl;
  std::cout << "label_path:" << label_path << std::endl;
  std::cout << "test_num:" << test_num << std::endl;

  LAC lac(model_path, conf_path, threads);
  std::string query;
  std::string output_str;
  std::string refer_str;
  struct timeval start;
  struct timeval end;
  int64_t cnt = 0;
  int64_t char_cnt = 0;
  std::fstream input_fs(input_path);
  std::fstream label_fs(label_path);
  if (!input_fs.is_open() || !label_fs.is_open()) {
    std::cerr << "open input or label file error";
    return 1;
  }
  int i = 0, right_num = 0, error_num = 0;
  gettimeofday(&start, NULL);
  while (!input_fs.eof()) {
    std::getline(input_fs, query);
    std::getline(label_fs, refer_str);
    cnt++;
    char_cnt += query.length();
    auto result = lac.lexer(query);
    output_str = "";
    for (int i = 0; i < result.size(); i++) {
      if (result[i].tag.length() == 0) {
        output_str += (result[i].word + " ");
      } else {
        output_str += (result[i].word + "\001" + result[i].tag + " ");
      }
    }
    if (output_str == refer_str) {
      right_num++;
    } else {
      error_num++;
    }
    i++;
    if (i >= test_num) {
      break;
    }
  }
  gettimeofday(&end, NULL);
  double time =
      end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  std::cerr << "using time: " << time << " \t qps:" << cnt / time
            << "\tc/s:" << char_cnt / time << std::endl;
  std::cerr << "right_num :" << right_num << "\t error num:" << error_num
            << "\t ratio:"
            << static_cast<float>(right_num) / (right_num + error_num)
            << std::endl;
}
