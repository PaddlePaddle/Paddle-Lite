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

#pragma once

#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "common/common.h"
#include "common/log.h"
#include "framework/ddim.h"
#include "framework/tensor.h"

static const char *g_ocr = "../models/ocr";
static const char *g_mobilenet_ssd = "../models/mobilenet+ssd";
static const char *g_genet_combine = "../models/enet";
static const char *g_eng = "../models/eng_20conv_1_9_fc";
static const char *g_mobilenet_ssd_gesture = "../models/mobilenet+ssd_gesture";
static const char *g_mobilenet_combined = "../models/mobilenet_combine";
static const char *g_googlenetv1_combined = "../models/googlenetv1_combine";
static const char *g_mobilenet_detect = "../models/mobilenet-detect";
static const char *g_squeezenet = "../models/squeezenet";
static const char *g_googlenet = "../models/googlenet";
static const char *g_googlenet_quali = "../models/googlenet_combine_quali";
static const char *g_mobilenet = "../models/mobilenet";
static const char *g_mobilenet_mul = "../models/r";
static const char *g_alexnet = "../models/alexnet";
static const char *g_inceptionv4 = "../models/inceptionv4";
static const char *g_inceptionv3 =
    "../models/InceptionV3_Spatial_Attention_Model";
static const char *g_nlp = "../models/nlp";
static const char *g_super = "../models/superresoltion";
static const char *g_resnet_50 = "../models/resnet_50";
static const char *g_resnet = "../models/resnet";
static const char *g_googlenet_combine = "../models/googlenet_combine";
static const char *g_yolo = "../models/yolo";
static const char *g_yolo_combined = "../models/yolo_combined";
static const char *g_yolo_mul = "../models/d";
static const char *g_fluid_fssd_new = "../models/fluid_fssd_new";
static const char *g_vgg16_ssd_combined = "../models/vgg16_ssd_combined";
static const char *g_mobilenet_vision = "../models/vision_mobilenet";
static const char *g_yolo_vision = "../models/vision_yolo";
static const char *g_test_image_1x3x224x224 =
    "../images/test_image_1x3x224x224_float";
static const char *g_test_image_1x3x224x224_banana =
    "../images/input_3x224x224_banana";
static const char *g_test_image_desktop_1_3_416_416_nchw_float =
    "../images/in_put_1_3_416_416_2";
static const char *g_hand = "../images/hand_image";
static const char *g_moto = "../images/moto_300x300_float";
static const char *g_imgfssd_ar = "../images/test_image_ssd_ar";
static const char *g_imgfssd_ar1 = "../images/003_0001.txt";
static const char *g_img = "../images/img.bin";
static const char *g_yolo_img = "../images/in_put_1_3_416_416_2";
static const char *g_super_img = "../images/mingren_input_data";
static const char *g_mobilenet_img = "../images/image";
static const char *g_test_image_1x3x224x224_vision_mobilenet_input =
    "../images/vision_mobilenet_input";
static const char *g_test_image_1x3x416x416_vision_yolo_input =
    "../images/yolo_input";

using paddle_mobile::framework::DDim;
using paddle_mobile::framework::Tensor;
using namespace paddle_mobile;  // NOLINT

template <typename T>
void SetupTensor(paddle_mobile::framework::Tensor *input,
                 paddle_mobile::framework::DDim dims, T lower, T upper) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T *input_ptr = input->mutable_data<T>(dims);
  for (int i = 0; i < input->numel(); ++i) {
    input_ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

template <>
void SetupTensor<bool>(paddle_mobile::framework::Tensor *input,
                       paddle_mobile::framework::DDim dims, bool lower,
                       bool upper) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  bool *input_ptr = input->mutable_data<bool>(dims);
  if (lower == upper) {
    for (int i = 0; i < input->numel(); ++i) {
      input_ptr[i] = lower;
    }
  } else {
    for (int i = 0; i < input->numel(); ++i) {
      input_ptr[i] = uniform_dist(rng) > 0.5;
    }
  }
}

template <typename T>
T *CreateInput(Tensor *input, DDim dims, T low, T up) {
  SetupTensor<T>(input, dims, static_cast<float>(low), static_cast<float>(up));
  return input->data<T>();
}

template <typename T>
void GetInput(const std::string &input_name, std::vector<T> *input,
              const std::vector<int64_t> &dims) {
  int size = 1;
  for (const auto &dim : dims) {
    size *= dim;
  }

  T *input_ptr = reinterpret_cast<T *>(malloc(sizeof(T) * size));
  std::ifstream in(input_name, std::ios::in | std::ios::binary);
  in.read(reinterpret_cast<char *>(input_ptr), size * sizeof(T));
  in.close();
  for (int i = 0; i < size; ++i) {
    input->push_back(input_ptr[i]);
  }
  free(input_ptr);
}

template <typename T>
void GetInput(const std::string &input_name,
              paddle_mobile::framework::Tensor *input,
              paddle_mobile::framework::DDim dims) {
  T *input_ptr = input->mutable_data<T>(dims);

  std::ifstream in(input_name, std::ios::in | std::ios::binary);
  in.read(reinterpret_cast<char *>(input_ptr), input->numel() * sizeof(T));
  in.close();
}
