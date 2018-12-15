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
#include <iostream>
#include "../test_helper.h"
#include "../test_include.h"

void load_images(const char *image_dir, const char *images_list,
                 std::vector<std::string> *image_names,
                 std::vector<std::pair<int, int>> *image_shapes) {
  int height, width;
  std::string filename;
  std::ifstream if_list(images_list, std::ios::in);
  while (!if_list.eof()) {
    if_list >> height >> width >> filename;
    image_shapes->push_back(std::make_pair(height, width));
    image_names->push_back(filename);
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: ./test_ocr model_dir image_dir images_list."
              << std::endl;
    return 1;
  }
  char *model_dir = argv[1];
  char *image_dir = argv[2];
  char *images_list = argv[3];

  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(8);
  auto isok = paddle_mobile.Load(std::string(model_dir) + "/model",
                                 std::string(model_dir) + "/params", true,
                                 false, 1, true);
  DLOG << "pass init model";
  std::vector<std::string> image_names;
  std::vector<std::pair<int, int>> image_shapes;
  load_images(image_dir, images_list, &image_names, &image_shapes);
  DLOG << "pass load images";

  for (int i = 0; i < image_names.size(); i++) {
    std::string file_name = image_names[i];
    std::vector<float> input;
    std::vector<int64_t> dims{1, 1, 48, 512};
    dims[2] = image_shapes[i].first;
    dims[3] = image_shapes[i].second;
    // load input image
    std::string img_path = std::string(image_dir) + "/" + file_name;
    std::cerr << "img_path: " << img_path << std::endl;
    std::cerr << "shape = [" << dims[0] << ", " << dims[1] << ", " << dims[2]
              << ", " << dims[3] << "]" << std::endl;
    GetInput<float>(img_path, &input, dims);
    // predict
    auto output = paddle_mobile.Predict(input, dims);
    // print result
    std::cerr << file_name << std::endl;
    std::cerr << output[0];
    for (int j = 1; j < output.size(); ++j) {
      std::cerr << " " << output[j];
    }
    std::cerr << std::endl;
  }
  return 0;
}
