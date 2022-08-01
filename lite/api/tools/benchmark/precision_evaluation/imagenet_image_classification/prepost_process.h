// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_API_TOOLS_BENCHMARK_PRECISION_EVALUATION_IMAGENET_IMAGE_CLASSIFICATION_PREPOST_PROCESS_H_
#define LITE_API_TOOLS_BENCHMARK_PRECISION_EVALUATION_IMAGENET_IMAGE_CLASSIFICATION_PREPOST_PROCESS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/tools/benchmark/precision_evaluation/utils.h"

namespace paddle {
namespace lite_api {

class ImagenetClassification {
 public:
  struct RESULT {
    std::string class_name;
    int class_id;
    float score;
  };

  ImagenetClassification();
  ~ImagenetClassification();

  void PreProcess(std::shared_ptr<PaddlePredictor> predictor,
                  const std::map<std::string, std::string> &config,
                  const std::vector<std::string> &image_files,
                  const int cnt);

  std::vector<RESULT> PostProcess(
      std::shared_ptr<PaddlePredictor> predictor,
      const std::map<std::string, std::string> &config,
      const std::vector<std::string> &image_files,
      const std::vector<std::string> &word_labels,
      const int cnt,
      const bool repeat_flag = true);

  std::vector<float> topk_accuracies(const int k, const int repeats);

 private:
  std::vector<float> topk_accuracies_;
};

}  // namespace lite_api
}  // namespace paddle

#endif  // LITE_API_TOOLS_BENCHMARK_PRECISION_EVALUATION_IMAGENET_IMAGE_CLASSIFICATION_PREPOST_PROCESS_H_
