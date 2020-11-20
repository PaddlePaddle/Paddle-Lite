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
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

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

const char* class_names[] = {
    "background", "aeroplane",   "bicycle", "bird",  "boat",
    "bottle",     "bus",         "car",     "cat",   "chair",
    "cow",        "diningtable", "dog",     "horse", "motorbike",
    "person",     "pottedplant", "sheep",   "sofa",  "train",
    "tvmonitor"};

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
  float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);

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

void pre_process(const cv::Mat& img, int width, int height, float* data) {
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {0.5f, 0.5f, 0.5f};
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  neon_mean_scale(dimg, data, width * height, mean, scale);
}

std::vector<Object> detect_object(const float* data,
                                  int count,
                                  float thresh,
                                  cv::Mat& image) {  // NOLINT
  if (data == nullptr) {
    std::cerr << "[ERROR] data can not be nullptr\n";
    exit(1);
  }
  std::vector<Object> rect_out;
  for (int iw = 0; iw < count; iw++) {
    int oriw = image.cols;
    int orih = image.rows;
    if (data[1] > thresh && static_cast<int>(data[0]) > 0) {
      Object obj;
      int x = static_cast<int>(data[2] * oriw);
      int y = static_cast<int>(data[3] * orih);
      int w = static_cast<int>(data[4] * oriw) - x;
      int h = static_cast<int>(data[5] * orih) - y;
      cv::Rect rec_clip =
          cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
      obj.batch_id = 0;
      obj.class_id = static_cast<int>(data[0]);
      obj.prob = data[1];
      obj.rec = rec_clip;
      if (w > 0 && h > 0 && obj.prob <= 1) {
        rect_out.push_back(obj);
        cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        std::string str_prob = std::to_string(obj.prob);
        std::string text = std::string(class_names[obj.class_id]) + ": " +
                           str_prob.substr(0, str_prob.find(".") + 4);
        int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double font_scale = 1.f;
        int thickness = 2;
        cv::Size text_size =
            cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
        float new_font_scale = w * 0.35 * font_scale / text_size.width;
        text_size = cv::getTextSize(
            text, font_face, new_font_scale, thickness, nullptr);
        cv::Point origin;
        origin.x = x + 10;
        origin.y = y + text_size.height + 10;
        cv::putText(image,
                    text,
                    origin,
                    font_face,
                    new_font_scale,
                    cv::Scalar(0, 255, 255),
                    thickness,
                    cv::LINE_AA);

        std::cout << "detection, image size: " << image.cols << ", "
                  << image.rows
                  << ", detect object: " << class_names[obj.class_id]
                  << ", score: " << obj.prob << ", location: x=" << x
                  << ", y=" << y << ", width=" << w << ", height=" << h
                  << std::endl;
      }
    }
    data += 6;
  }
  return rect_out;
}

void RunModel(std::string model_file, std::string img_path) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  const int in_width = 300;
  const int in_height = 300;
  input_tensor->Resize({1, 3, in_height, in_width});
  auto* data = input_tensor->mutable_data<float>();
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  pre_process(img, in_width, in_height, data);

  // 4. Run predictor
  predictor->Run();

  // 5. Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t cnt = ShapeProduction(shape_out);
  auto rec_out = detect_object(outptr, static_cast<int>(cnt / 6), 0.6f, img);
  int start = img_path.find_last_of("/");
  int end = img_path.find_last_of(".");
  std::string img_name = img_path.substr(start + 1, end - start - 1);
  std::string result_name = img_name + "_ssd_detection_result.jpg";
  cv::imwrite(result_name, img);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0] << " model_file image_path\n";
    exit(1);
  }
  std::string model_file = argv[1];
  std::string img_path = argv[2];
  RunModel(model_file, img_path);
  return 0;
}
