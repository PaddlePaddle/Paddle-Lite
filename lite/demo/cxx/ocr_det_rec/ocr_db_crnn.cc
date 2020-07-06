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

#include <fstream>
#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"    // NOLINT
#include "post_process.h"  // NOLINT
#include "pre_process.h"   // NOLINT

using namespace paddle::lite_api;  // NOLINT

void RunRecModel(std::vector<std::vector<std::vector<int>>> boxes,
                 cv::Mat img,
                 std::string rec_model_path,
                 std::string dict_path,
                 std::string img_name,
                 std::string output_path,
                 float* rec_time) {
  // Create predictor
  MobileConfig config;
  config.set_model_from_file(rec_model_path);
  std::shared_ptr<PaddlePredictor> predictor_crnn =
      CreatePaddlePredictor<MobileConfig>(config);

  // Load dict
  auto charactor_dict = ReadDict(dict_path);

  img_name = img_name.substr(0, img_name.find_last_of("."));
  std::string save_output_path = output_path + "/result_" + img_name + ".txt";
  std::ofstream ofs(save_output_path, std::ios::out);
  if (!ofs.is_open()) {
    std::cerr << "Open output file error:" << save_output_path << std::endl;
  }

  cv::Mat src_img;
  cv::Mat crop_img;
  cv::Mat resize_img;
  img.copyTo(src_img);
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  auto start = GetCurrentUS();
  int index = 1;

  for (int i = boxes.size() - 1; i >= 0; i--) {
    // Set input
    crop_img = GetRotateCropImage(src_img, boxes[i]);
    float wh_ratio =
        static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);
    resize_img = CrnnResizeImg(crop_img, wh_ratio);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);
    const float* img_data = reinterpret_cast<const float*>(resize_img.data);

    std::unique_ptr<Tensor> input_tensor = predictor_crnn->GetInput(0);
    input_tensor->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto* input_data = input_tensor->mutable_data<float>();
    NeonMeanScale(
        img_data, input_data, resize_img.rows * resize_img.cols, mean, scale);

    // Run CRNN predictor
    predictor_crnn->Run();

    // Get output and run postprocess
    std::unique_ptr<const Tensor> output_tensor_0 =
        predictor_crnn->GetOutput(0);
    auto* rec_idx = output_tensor_0->data<int>();
    auto rec_idx_lod = output_tensor_0->lod();
    auto shape_out = output_tensor_0->shape();

    std::vector<int> pred_idx;
    for (int n = static_cast<int>(rec_idx_lod[0][0]);
         n < static_cast<int>(rec_idx_lod[0][1] * 2);
         n += 2) {
      pred_idx.push_back(static_cast<int>(rec_idx[n]));
    }
    if (pred_idx.empty()) {
      continue;
    }

    ofs << index++ << "\t";
    for (int n = 0; n < pred_idx.size(); n++) {
      ofs << charactor_dict[pred_idx[n]];
    }

    // Get score
    std::unique_ptr<const Tensor> output_tensor_1 =
        predictor_crnn->GetOutput(1);
    auto* predict_batch = output_tensor_1->data<float>();
    auto predict_shape = output_tensor_1->shape();
    auto predict_lod = output_tensor_1->lod();

    float score = 0.f;
    int count = 0;
    int blank = predict_shape[1];
    for (int n = predict_lod[0][0]; n < predict_lod[0][1] - 1; n++) {
      int argmax_idx = static_cast<int>(
          Argmax(&predict_batch[n * blank], &predict_batch[(n + 1) * blank]));
      float max_value = *std::max_element(&predict_batch[n * blank],
                                          &predict_batch[(n + 1) * blank]);
      if (blank - 1 - argmax_idx > 0) {
        score += max_value;
        count++;
      }
    }
    score /= count;
    ofs << "\tscore: " << score << std::endl;
  }
  ofs.close();
  auto end = GetCurrentUS();
  *rec_time = (end - start) / 1000;

  std::cout << "Finish Recogintion, the result saved in " << save_output_path
            << std::endl;
}

std::vector<std::vector<std::vector<int>>> RunDetModel(std::string model_file,
                                                       cv::Mat img,
                                                       std::string img_name,
                                                       std::string output_path,
                                                       float* det_time) {
  // Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // Process img
  auto start = GetCurrentUS();
  const int max_side_len = 960;
  float ratio_h{};
  float ratio_w{};

  cv::Mat src_img;
  img.copyTo(src_img);
  img = DetResizeImg(img, max_side_len, &ratio_h, &ratio_w);
  img.convertTo(img, CV_32FC3, 1.0 / 255.f);

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, img.rows, img.cols});
  auto* input_data = input_tensor->mutable_data<float>();

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  const float* dimg = reinterpret_cast<const float*>(img.data);
  NeonMeanScale(dimg, input_data, img.rows * img.cols, mean, scale);

  // Run predictor
  predictor->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t out_numl = 1;
  for (auto i : shape_out) {
    out_numl *= i;
  }

  float pred[shape_out[2]][shape_out[3]];
  unsigned char cbuf[shape_out[2]][shape_out[3]];
  for (int i = 0; i < static_cast<int>(shape_out[2] * shape_out[3]); i++) {
    pred[static_cast<int>(i / static_cast<int>(shape_out[3]))]
        [static_cast<int>(i % shape_out[3])] = static_cast<float>(outptr[i]);
    cbuf[static_cast<int>(i / static_cast<int>(shape_out[3]))]
        [static_cast<int>(i % shape_out[3])] =
            static_cast<unsigned char>((outptr[i]) * 255);
  }

  cv::Mat cbuf_map(shape_out[2],
                   shape_out[3],
                   CV_8UC1,
                   reinterpret_cast<unsigned char*>(cbuf));
  cv::Mat pred_map(
      shape_out[2], shape_out[3], CV_32F, reinterpret_cast<float*>(pred));

  const double threshold = 0.3 * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  auto boxes = BoxesFromBitmap(pred_map, bit_map);
  std::vector<std::vector<std::vector<int>>> filter_boxes =
      FilterTagDetRes(boxes, ratio_h, ratio_w, src_img);
  auto end = GetCurrentUS();
  *det_time = (end - start) / 1000;

  // Visualization
  cv::Point rook_points[filter_boxes.size()][4];
  for (int n = 0; n < filter_boxes.size(); n++) {
    for (int m = 0; m < filter_boxes[0].size(); m++) {
      rook_points[n][m] = cv::Point(static_cast<int>(filter_boxes[n][m][0]),
                                    static_cast<int>(filter_boxes[n][m][1]));
    }
  }

  cv::Mat img_vis;
  src_img.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point* ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }
  std::string img_save_path = output_path + "/result_" + img_name;
  cv::imwrite(img_save_path, img_vis);

  std::cout << "Finish detection, the result saved in " + img_save_path
            << std::endl;
  return filter_boxes;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "Input error." << std::endl;
    std::cerr << "Usage: " << argv[0] << " det_model_opt_path "
              << "rec_model_opt_path dict_path image_path output_path\n";
    exit(1);
  }
  std::string det_model_path = argv[1];
  std::string rec_model_path = argv[2];
  std::string dict_path = argv[3];
  std::string img_path = argv[4];
  std::string output_path = argv[5];
  size_t pos = img_path.find_last_of("/");
  std::string img_name = img_path.substr(pos + 1);

  cv::Mat src_img = cv::imread(img_path, cv::IMREAD_COLOR);
  float det_time = 0;
  float rec_time = 0;
  auto boxes =
      RunDetModel(det_model_path, src_img, img_name, output_path, &det_time);
  RunRecModel(boxes,
              src_img,
              rec_model_path,
              dict_path,
              img_name,
              output_path,
              &rec_time);
  std::cout << "It took " << det_time + rec_time << " ms" << std::endl;

  return 0;
}
