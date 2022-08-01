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

#ifndef LITE_API_TOOLS_BENCHMARK_PRECISION_EVALUATION_UTILS_H_
#define LITE_API_TOOLS_BENCHMARK_PRECISION_EVALUATION_UTILS_H_

#include <map>
#include <opencv2/opencv.hpp>
#include <string>  // NOLINT
#include <vector>  // NOLINT

namespace paddle {
namespace lite_api {

const std::string GetAbsPath(const std::string file_name);
const std::vector<std::string> ReadDict(std::string path);
const std::map<std::string, std::string> LoadConfigTxt(std::string config_path);
void PrintConfig(const std::map<std::string, std::string> &config);
cv::Mat ResizeImage(const cv::Mat &img, const int resize_short_size);
cv::Mat CenterCropImg(const cv::Mat &img, const int crop_size);
void NeonMeanScale(const float *din,
                   float *dout,
                   int size,
                   const std::vector<float> mean,
                   const std::vector<float> scale);

}  // namespace lite_api
}  // namespace paddle

#endif  // LITE_API_TOOLS_BENCHMARK_PRECISION_EVALUATION_UTILS_H_
