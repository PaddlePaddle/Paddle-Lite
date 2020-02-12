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

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"  // NOLINT

using namespace paddle::lite_api;  // NOLINT

struct Object {
  int batch_id;
  cv::Rect rec;
  int class_id;
  float prob;
};

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float* din,
                     float* dout,
                     int size,
                     const std::vector<float> mean,
                     const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3\n";
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float* dout_c0 = dout;
  float* dout_c1 = dout + size;
  float* dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}

void pre_process(const cv::Mat& img,
                 int width,
                 int height,
                 const std::vector<float>& mean,
                 const std::vector<float>& scale,
                 float* data,
                 bool is_scale = false) {
  cv::Mat resized_img;
  cv::resize(
      img, resized_img, cv::Size(width, height), 0.f, 0.f, cv::INTER_CUBIC);
  cv::Mat imgf;
  float scale_factor = is_scale ? 1.f / 256 : 1.f;
  resized_img.convertTo(imgf, CV_32FC3, scale_factor);
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  neon_mean_scale(dimg, data, width * height, mean, scale);
}

void RunModel(std::string det_model_dir,
              std::string class_model_dir,
              std::string img_path) {
  // Prepare
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  float shrink = 0.2;
  int width = img.cols;
  int height = img.rows;
  int s_width = static_cast<int>(width * shrink);
  int s_height = static_cast<int>(height * shrink);

  // Detection
  MobileConfig config;
  config.set_model_dir(det_model_dir);

  // Create Predictor For Detction Model
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // Get Input Tensor
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize({1, 3, s_height, s_width});
  auto* data = input_tensor0->mutable_data<float>();

  // Do PreProcess
  std::vector<float> detect_mean = {104.f, 117.f, 123.f};
  std::vector<float> detect_scale = {0.007843, 0.007843, 0.007843};
  pre_process(img, s_width, s_height, detect_mean, detect_scale, data, false);

  // Detection Model Run
  predictor->Run();

  // Get Output Tensor
  std::unique_ptr<const Tensor> output_tensor0(
      std::move(predictor->GetOutput(0)));
  auto* outptr = output_tensor0->data<float>();
  auto shape_out = output_tensor0->shape();
  int64_t out_len = ShapeProduction(shape_out);

  // Filter Out Detection Box
  float detect_threshold = 0.3;
  std::vector<Object> detect_result;
  for (int i = 0; i < out_len / 6; ++i) {
    if (outptr[1] >= detect_threshold) {
      Object obj;
      int xmin = static_cast<int>(width * outptr[2]);
      int ymin = static_cast<int>(height * outptr[3]);
      int xmax = static_cast<int>(width * outptr[4]);
      int ymax = static_cast<int>(height * outptr[5]);
      int w = xmax - xmin;
      int h = ymax - ymin;
      cv::Rect rec_clip =
          cv::Rect(xmin, ymin, w, h) & cv::Rect(0, 0, width, height);
      obj.rec = rec_clip;
      detect_result.push_back(obj);
    }
    outptr += 6;
  }

  // Classification
  config.set_model_dir(class_model_dir);

  // Create Predictor For Classification Model
  predictor = CreatePaddlePredictor<MobileConfig>(config);

  // Get Input Tensor
  std::unique_ptr<Tensor> input_tensor1(std::move(predictor->GetInput(0)));
  int classify_w = 128;
  int classify_h = 128;
  input_tensor1->Resize({1, 3, classify_h, classify_w});
  auto* input_data = input_tensor1->mutable_data<float>();
  int detect_num = detect_result.size();
  std::vector<float> classify_mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> classify_scale = {1.f, 1.f, 1.f};
  float classify_threshold = 0.5;
  for (int i = 0; i < detect_num; ++i) {
    cv::Rect rec_clip = detect_result[i].rec;
    cv::Mat roi = img(rec_clip);

    // Do PreProcess
    pre_process(roi,
                classify_w,
                classify_h,
                classify_mean,
                classify_scale,
                input_data,
                true);

    // Classification Model Run
    predictor->Run();

    // Get Output Tensor
    std::unique_ptr<const Tensor> output_tensor1(
        std::move(predictor->GetOutput(1)));
    auto* outptr = output_tensor1->data<float>();

    // Draw Detection and Classification Results
    cv::rectangle(img, rec_clip, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    std::string text = outptr[1] > classify_threshold ? "wear mask" : "no mask";
    int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double font_scale = 1.f;
    int thickness = 1;
    cv::Size text_size =
        cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
    float new_font_scale = rec_clip.width * 0.7 * font_scale / text_size.width;
    text_size =
        cv::getTextSize(text, font_face, new_font_scale, thickness, nullptr);
    cv::Point origin;
    origin.x = rec_clip.x + 5;
    origin.y = rec_clip.y + text_size.height + 5;
    cv::putText(img,
                text,
                origin,
                font_face,
                new_font_scale,
                cv::Scalar(0, 255, 255),
                thickness,
                cv::LINE_AA);

    std::cout << "detect face, location: x=" << rec_clip.x
              << ", y=" << rec_clip.y << ", width=" << rec_clip.width
              << ", height=" << rec_clip.height
              << ", wear mask: " << (outptr[1] > classify_threshold)
              << std::endl;
  }

  // Write Result to Image File
  int start = img_path.find_last_of("/");
  int end = img_path.find_last_of(".");
  std::string img_name = img_path.substr(start + 1, end - start - 1);
  std::string result_name = img_name + "_mask_detection_result.jpg";
  cv::imwrite(result_name, img);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " detction_model_dir classification_model_dir image_path\n";
    exit(1);
  }
  std::string detect_model_dir = argv[1];
  std::string classify_model_dir = argv[2];
  std::string img_path = argv[3];
  RunModel(detect_model_dir, classify_model_dir, img_path);
  return 0;
}
