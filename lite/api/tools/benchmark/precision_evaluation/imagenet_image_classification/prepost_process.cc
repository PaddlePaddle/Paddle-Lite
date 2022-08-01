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

#include "lite/api/tools/benchmark/precision_evaluation/imagenet_image_classification/prepost_process.h"
#include <algorithm>
#include <utility>
#include "lite/api/paddle_api.h"
#include "lite/utils/model_util.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite_api {

constexpr int TOPK_MAX = 10;
ImagenetClassification::ImagenetClassification() {
  topk_accuracies_.resize(TOPK_MAX, 0.f);
}

ImagenetClassification::~ImagenetClassification() {}

void ImagenetClassification::PreProcess(
    std::shared_ptr<PaddlePredictor> predictor,
    const std::map<std::string, std::string> &config,
    const std::vector<std::string> &image_files,
    const int cnt) {
  if (image_files.empty()) return;

  // Read image
  // std::cout << "image: " << image_files.at(cnt) << std::endl;
  cv::Mat img = cv::imread(image_files.at(cnt), cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  // Reshape image
  int resize_short_size = stoi(config.at("resize_short_size"));
  int crop_size = stoi(config.at("crop_size"));
  cv::Mat resize_image = ResizeImage(img, resize_short_size);
  cv::Mat crop_image = CenterCropImg(resize_image, crop_size);

  // Prepare input data from image
  cv::Mat img_fp;
  const double alpha = 1.0 / 255.0;
  crop_image.convertTo(img_fp, CV_32FC3, alpha);
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, img_fp.channels(), img_fp.rows, img_fp.cols});
  const auto mean = lite::Split<float>(config.at("mean"), ",");
  const auto scale = lite::Split<float>(config.at("scale"), ",");
  const float *dimg = reinterpret_cast<const float *>(img_fp.data);

  auto *input0 = input_tensor->mutable_data<float>();
  NeonMeanScale(dimg, input0, img_fp.rows * img_fp.cols, mean, scale);
}

std::vector<ImagenetClassification::RESULT> ImagenetClassification::PostProcess(
    std::shared_ptr<PaddlePredictor> predictor,
    const std::map<std::string, std::string> &config,
    const std::vector<std::string> &image_files,
    const std::vector<std::string> &word_labels,
    const int cnt,
    const bool repeat_flag) {
  std::vector<RESULT> results;
  if (image_files.empty()) return results;

  size_t output_tensor_num = predictor->GetOutputNames().size();
  if (output_tensor_num != 1) {
    std::cerr << "The number of ouptut tensors should be equal to 1, but got "
              << output_tensor_num << " Abort!" << std::endl;
    std::abort();
  }
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto *output_data = output_tensor->data<float>();
  auto out_shape = output_tensor->shape();
  auto out_data = output_tensor->data<float>();
  auto ele_num = lite::ShapeProduction(out_shape);

  const int TOPK = stoi(config.at("topk"));
  std::vector<int> max_indices(TOPK, 0);
  std::vector<double> max_scores(TOPK, 0.);
  for (int i = 0; i < ele_num; i++) {
    float score = output_data[i];
    int index = i;
    for (int j = 0; j < TOPK; j++) {
      if (score > max_scores[j]) {
        index += max_indices[j];
        max_indices[j] = index - max_indices[j];
        index -= max_indices[j];
        score += max_scores[j];
        max_scores[j] = score - max_scores[j];
        score -= max_scores[j];
      }
    }
  }

  results.resize(TOPK);
  for (int i = 0; i < results.size(); i++) {
    results[i].class_name = "Unknown";
    if (max_indices[i] >= 0 && max_indices[i] < word_labels.size()) {
      results[i].class_name = word_labels[max_indices[i]];
    }
    results[i].score = max_scores[i];
    results[i].class_id = max_indices[i];

    const int label = cnt;
    if ((label == results[i].class_id) && repeat_flag) {
      for (auto j = i; j < TOPK; j++) {
        topk_accuracies_[j] += 1;
      }
      break;
    }
  }

  if (stoi(config.at("store_result_as_image")) == 1) {
    std::cout << "=== clas result for image: " << image_files.at(cnt)
              << "===" << std::endl;
    cv::Mat img = cv::imread(image_files.at(cnt), cv::IMREAD_COLOR);
    cv::Mat output_image(img);
    for (int i = 0; i < results.size(); i++) {
      auto text = lite::string_format(
          "Top-%d, class_id: %d, class_name: %s, score: %.3f",
          i + 1,
          results[i].class_id,
          results[i].class_name.c_str(),
          results[i].score);
      cv::putText(output_image,
                  text,
                  cv::Point2d(5, i * 18 + 20),
                  cv::FONT_HERSHEY_PLAIN,
                  1,
                  cv::Scalar(51, 255, 255));

      std::cout << text << std::endl;
    }
    std::string output_image_path = "./" + std::to_string(cnt) + ".png";
    cv::imwrite(output_image_path, output_image);
    std::cout << "Save output image into " << output_image_path << std::endl;
  }

  if (repeat_flag) {
    for (int i = 0; i < results.size(); i++) {
      std::cout << "top-" << i + 1 << ":" << topk_accuracies_[i] / (cnt + 1)
                << std::endl;
    }
  }

  return results;
}

std::vector<float> ImagenetClassification::topk_accuracies(const int k,
                                                           const int repeats) {
  topk_accuracies_.resize(k);
  std::transform(topk_accuracies_.begin(),
                 topk_accuracies_.begin() + k,
                 topk_accuracies_.begin(),
                 [&repeats](float &c) {
                   std::cout << c << std::endl;
                   return c / static_cast<float>(repeats);
                 });
  return topk_accuracies_;
}

}  // namespace lite_api
}  // namespace paddle
