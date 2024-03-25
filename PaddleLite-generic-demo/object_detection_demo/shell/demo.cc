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

#include <paddle_api.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <opencv2/core/core.hpp>  // NOLINT
#include <opencv2/highgui.hpp>    // NOLINT
#include <opencv2/opencv.hpp>     // NOLINT
#include <sstream>                // NOLINT
#include <vector>                 // NOLINT

int WARMUP_COUNT = 1;
int REPEAT_COUNT = 5;
const int CPU_THREAD_NUM = 1;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_NO_BIND;

#ifdef __QNX__
#include <devctl.h>
#include <fcntl.h>
inline int64_t get_current_us() {
  auto fd = open("/dev/qgptp", O_RDONLY);
  if (fd < 0) {
    printf("open '/dev/qgptp' failed.");
  }
  uint64_t time_nsec;
#define GPTP_GETTIME __DIOF(_DCMD_MISC, 1, int)
  if (EOK != devctl(fd, GPTP_GETTIME, &time_nsec, sizeof(time_nsec), NULL)) {
    printf("devctl failed.");
  }
  if (close(fd) < 0) {
    printf("close fd failed.");
  }
  return time_nsec / 1000;
}
#else
inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}
#endif

template <typename T>
void get_value_from_sstream(std::stringstream *ss, T *value) {
  (*ss) >> (*value);
}

template <>
void get_value_from_sstream<std::string>(std::stringstream *ss,
                                         std::string *value) {
  *value = ss->str();
}

template <typename T>
std::vector<T> split_string(const std::string &str, char sep) {
  std::stringstream ss;
  std::vector<T> values;
  T value;
  values.clear();
  for (auto c : str) {
    if (c != sep) {
      ss << c;
    } else {
      get_value_from_sstream<T>(&ss, &value);
      values.push_back(std::move(value));
      ss.str({});
      ss.clear();
    }
  }
  if (!ss.str().empty()) {
    get_value_from_sstream<T>(&ss, &value);
    values.push_back(std::move(value));
    ss.str({});
    ss.clear();
  }
  return values;
}

bool read_file(const std::string &filename,
               std::vector<char> *contents,
               bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char *ptr = reinterpret_cast<char *>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

bool write_file(const std::string &filename,
                const std::vector<char> &contents,
                bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "wb" : "w");
  if (!fp) return false;
  size_t size = contents.size();
  size_t offset = 0;
  const char *ptr = reinterpret_cast<const char *>(&(contents.at(0)));
  while (offset < size) {
    size_t already_written = fwrite(ptr, 1, size - offset, fp);
    offset += already_written;
    ptr += already_written;
  }
  fclose(fp);
  return true;
}

int64_t shape_production(std::vector<int64_t> shape) {
  int64_t s = 1;
  for (int64_t dim : shape) {
    s *= dim;
  }
  return s;
}

void nhwc32nc3hw(const float *src,
                 float *dst,
                 const float *mean,
                 const float *std,
                 int width,
                 int height) {
  int size = height * width;
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t vmean0 = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vmean1 = vdupq_n_f32(mean ? mean[1] : 0.0f);
  float32x4_t vmean2 = vdupq_n_f32(mean ? mean[2] : 0.0f);
  float32x4_t vscale0 = vdupq_n_f32(std ? (1.0f / std[0]) : 1.0f);
  float32x4_t vscale1 = vdupq_n_f32(std ? (1.0f / std[1]) : 1.0f);
  float32x4_t vscale2 = vdupq_n_f32(std ? (1.0f / std[2]) : 1.0f);
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(src);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dst_c0, vs0);
    vst1q_f32(dst_c1, vs1);
    vst1q_f32(dst_c2, vs2);
    src += 12;
    dst_c0 += 4;
    dst_c1 += 4;
    dst_c2 += 4;
  }
#endif
  for (; i < size; i++) {
    *(dst_c0++) = (*(src++) - mean[0]) / std[0];
    *(dst_c1++) = (*(src++) - mean[1]) / std[1];
    *(dst_c2++) = (*(src++) - mean[2]) / std[2];
  }
}

void nhwc12nc1hw(const float *src,
                 float *dst,
                 const float *mean,
                 const float *std,
                 int width,
                 int height) {
  int size = height * width;
  int i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t vmean = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vscale = vdupq_n_f32(std ? (1.0f / std[0]) : 1.0f);
  for (; i < size - 3; i += 4) {
    float32x4_t vin = vld1q_f32(src);
    float32x4_t vsub = vsubq_f32(vin, vmean);
    float32x4_t vs = vmulq_f32(vsub, vscale);
    vst1q_f32(dst, vs);
    src += 4;
    dst += 4;
  }
#endif
  for (; i < size; i++) {
    *(dst++) = (*(src++) - mean[0]) / std[0];
  }
}

typedef struct {
  // type = 1 indicates the model(0: image) has only one input and one output.
  // type = 2 indicates the model has two inputs(0: image, 1: scale_factor) and
  // one output.
  // type = 3 indicates the model has three inputs(0: im_shape, 1: image, 2:
  // scale_factor) and one output.
  int type;
  int width;
  int height;
  std::vector<float> mean;
  std::vector<float> std;
  float draw_threshold{0.0f};
  std::vector<std::string> label_list;
} CONFIG;

std::vector<std::string> load_label(const std::string &path) {
  std::vector<char> buffer;
  if (!read_file(path, &buffer, false)) {
    printf("Failed to load the label file %s\n", path.c_str());
    exit(-1);
  }
  std::string content(buffer.begin(), buffer.end());
  auto lines = split_string<std::string>(content, '\n');
  if (lines.empty()) {
    printf("The label file %s should not be empty!\n", path.c_str());
    exit(-1);
  }
  return lines;
}

CONFIG load_config(const std::string &path) {
  CONFIG config;
  std::vector<char> buffer;
  if (!read_file(path, &buffer, false)) {
    printf("Failed to load the config file %s\n", path.c_str());
    exit(-1);
  }
  std::string dir = ".";
  auto pos = path.find_last_of("/");
  if (pos != std::string::npos) {
    dir = path.substr(0, pos);
  }
  printf("dir: %s\n", dir.c_str());
  std::string content(buffer.begin(), buffer.end());
  auto lines = split_string<std::string>(content, '\n');
  std::map<std::string, std::string> values;
  for (auto &line : lines) {
    auto value = split_string<std::string>(line, ':');
    if (value.size() != 2) {
      printf("Format error at '%s', it should be '<key>:<value>'.\n",
             line.c_str());
      exit(-1);
    }
    values[value[0]] = value[1];
  }
  // type
  if (!values.count("type")) {
    printf("Missing the key 'type'!\n");
    exit(-1);
  }
  config.type = atoi(values["type"].c_str());
  if (config.type < 1 || config.type > 3) {
    printf("The key 'type' only supports 1,2 or 3, but receive %d!\n",
           config.type);
    exit(-1);
  }
  printf("type: %d\n", config.type);
  // width
  if (!values.count("width")) {
    printf("Missing the key 'width'!\n");
    exit(-1);
  }
  config.width = atoi(values["width"].c_str());
  if (config.width <= 0) {
    printf("The key 'width' should > 0, but receive %d!\n", config.width);
    exit(-1);
  }
  printf("width: %d\n", config.width);
  // height
  if (!values.count("height")) {
    printf("Missing the key 'height' !\n");
    exit(-1);
  }
  config.height = atoi(values["height"].c_str());
  if (config.height <= 0) {
    printf("The key 'height' should > 0, but receive %d!\n", config.height);
    exit(-1);
  }
  printf("height: %d\n", config.height);
  // mean
  if (!values.count("mean")) {
    printf("Missing the key 'mean'!\n");
    exit(-1);
  }
  config.mean = split_string<float>(values["mean"], ',');
  if (config.mean.size() != 3) {
    printf("The key 'mean' should contain 3 values, but receive %u!\n",
           config.mean.size());
    exit(-1);
  }
  printf("mean: %f,%f,%f\n", config.mean[0], config.mean[1], config.mean[2]);
  // std
  if (!values.count("std")) {
    printf("Missing the key 'std' !\n");
    exit(-1);
  }
  config.std = split_string<float>(values["std"], ',');
  if (config.std.size() != 3) {
    printf("The key 'std' should contain 3 values, but receive %u!\n",
           config.std.size());
    exit(-1);
  }
  printf("std: %f,%f,%f\n", config.std[0], config.std[1], config.std[2]);
  // draw_threshold(optional)
  if (values.count("draw_threshold")) {
    config.draw_threshold = atof(values["draw_threshold"].c_str());
    if (config.draw_threshold < 0.f) {
      printf("The key 'draw_threshold' should >= 0.f, but receive %f!\n",
             config.draw_threshold);
      exit(-1);
    }
  }
  printf("draw_threshold: %f\n", config.draw_threshold);
  // label_list(optional)
  if (values.count("label_list")) {
    std::string label_list = values["label_list"];
    if (!label_list.empty()) {
      config.label_list = load_label(dir + "/" + label_list);
    }
  }
  printf("label_list size: %u\n", config.label_list.size());
  return config;
}

std::vector<std::string> load_dataset(const std::string &path) {
  std::vector<char> buffer;
  if (!read_file(path, &buffer, false)) {
    printf("Failed to load the dataset list file %s\n", path.c_str());
    exit(-1);
  }
  std::string content(buffer.begin(), buffer.end());
  auto lines = split_string<std::string>(content, '\n');
  if (lines.empty()) {
    printf("The dataset list file %s should not be empty!\n", path.c_str());
    exit(-1);
  }
  return lines;
}

std::vector<cv::Scalar> generate_color_map(int num_classes) {
  if (num_classes < 10) {
    num_classes = 10;
  }
  std::vector<cv::Scalar> color_map = std::vector<cv::Scalar>(num_classes);
  for (int i = 0; i < num_classes; i++) {
    int j = 0;
    int label = i;
    int R = 0, G = 0, B = 0;
    while (label) {
      R |= (((label >> 0) & 1) << (7 - j));
      G |= (((label >> 1) & 1) << (7 - j));
      B |= (((label >> 2) & 1) << (7 - j));
      j++;
      label >>= 3;
    }
    color_map[i] = cv::Scalar(R, G, B);
  }
  return color_map;
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
             const std::string &config_path,
             const std::string &dataset_dir) {
  // Parse the config file to extract the model info
  auto config = load_config(config_path);
  // Load dataset list
  auto dataset = load_dataset(dataset_dir + "/list.txt");
  // Prepare for inference and warmup
  std::unique_ptr<paddle::lite_api::Tensor> image_tensor = nullptr;
  std::unique_ptr<paddle::lite_api::Tensor> scale_factor_tensor = nullptr;
  std::unique_ptr<paddle::lite_api::Tensor> im_shape_tensor = nullptr;
  if (config.type == 1) {
    image_tensor = predictor->GetInput(0);
  } else if (config.type == 2) {
    image_tensor = predictor->GetInput(0);
    scale_factor_tensor = predictor->GetInput(1);
  } else if (config.type == 3) {
    image_tensor = predictor->GetInput(1);
    scale_factor_tensor = predictor->GetInput(2);
    // Fill im_shape tensor with a constant float value
    im_shape_tensor = predictor->GetInput(0);
    im_shape_tensor->Resize({1, 2});
    auto im_shape_data = im_shape_tensor->mutable_data<float>();
    im_shape_data[0] = config.width;
    im_shape_data[1] = config.height;
  } else {
    printf("Unknown type(%d)!\n", config.type);
    exit(-1);
  }
  image_tensor->Resize({1, 3, config.height, config.width});
  auto image_data = image_tensor->mutable_data<float>();
  float *scale_factor_data = nullptr;
  if (scale_factor_tensor) {
    scale_factor_tensor->Resize({1, 2});
    scale_factor_data = scale_factor_tensor->mutable_data<float>();
  }
  predictor->Run();  // Warmup
  // Traverse the list of the dataset and run inference on each sample
  double cur_costs[3];
  double total_costs[3] = {0, 0, 0};
  double max_costs[3] = {0, 0, 0};
  double min_costs[3] = {std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
  int iter_count = 0;
  auto sample_count = dataset.size();
  auto color_map = generate_color_map(config.label_list.size());
  for (size_t i = 0; i < sample_count; i++) {
    auto sample_name = dataset[i];
    printf("[%u/%u] Processing %s\n", i + 1, sample_count, sample_name.c_str());
    auto input_path = dataset_dir + "/inputs/" + sample_name;
    auto output_path = dataset_dir + "/outputs/" + sample_name;
    // Check if input and output is accessable
    if (access(input_path.c_str(), R_OK) != 0) {
      printf("%s not found or readable!\n", input_path.c_str());
      exit(-1);
    }
    // Preprocess
    double start = get_current_us();
    // image tensor
    cv::Mat origin_image = cv::imread(input_path);
    cv::Mat resized_image;
    cv::resize(origin_image,
               resized_image,
               cv::Size(config.width, config.height),
               0,
               0);
    if (resized_image.channels() == 3) {
      cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    } else if (resized_image.channels() == 4) {
      cv::cvtColor(resized_image, resized_image, cv::COLOR_BGRA2RGB);
    } else {
      printf("The channel size should be 4 or 3, but receive %d!\n",
             resized_image.channels());
      exit(-1);
    }
    resized_image.convertTo(resized_image, CV_32FC3, 1 / 255.f);
    nhwc32nc3hw(reinterpret_cast<const float *>(resized_image.data),
                image_data,
                config.mean.data(),
                config.std.data(),
                config.width,
                config.height);
    if (scale_factor_data) {
      scale_factor_data[0] =
          static_cast<float>(config.height) / origin_image.rows;
      scale_factor_data[1] =
          static_cast<float>(config.width) / origin_image.cols;
    }
    double end = get_current_us();
    cur_costs[0] = (end - start) / 1000.0f;
    // Inference
    start = get_current_us();
    predictor->Run();
    end = get_current_us();
    cur_costs[1] = (end - start) / 1000.0f;
    // Postprocess
    start = get_current_us();
    auto output_tensor = predictor->GetOutput(0);
    auto output_data = output_tensor->data<float>();
    auto output_size = shape_production(output_tensor->shape());
    int output_index = 0;
    for (int64_t j = 0; j < output_size; j += 6) {
      auto class_id = static_cast<int>(round(output_data[j]));
      auto score = output_data[j + 1];
      if (score < config.draw_threshold) continue;
      std::string class_name =
          class_id >= 0 && class_id < config.label_list.size()
              ? config.label_list[class_id]
              : "Unknown";
      float x0, y0, x1, y1;
      x0 = output_data[j + 2];
      y0 = output_data[j + 3];
      x1 = output_data[j + 4];
      y1 = output_data[j + 5];
      printf("[%d] %s - %f [%f,%f,%f,%f]\n",
             output_index,
             class_name.c_str(),
             score,
             x0,
             y0,
             x1,
             y1);
      if (!scale_factor_tensor && !im_shape_tensor) {
        x0 *= origin_image.cols;
        y0 *= origin_image.rows;
        x1 *= origin_image.cols;
        y1 *= origin_image.rows;
      }
      int lx = std::max(static_cast<int>(x0), 0);
      int ly = std::max(static_cast<int>(y0), 0);
      int w = std::max(
          std::min(static_cast<int>(x1), origin_image.cols - 1) - lx, 0);
      int h = std::max(
          std::min(static_cast<int>(y1), origin_image.rows - 1) - ly, 0);
      if (w > 0 && h > 0) {
        cv::Scalar color = color_map[class_id % color_map.size()];
        cv::rectangle(origin_image, cv::Rect(lx, ly, w, h), color);
        cv::rectangle(origin_image,
                      cv::Point2d(lx, ly),
                      cv::Point2d(lx + w, ly - 10),
                      color,
                      -1);
        cv::putText(origin_image,
                    std::to_string(output_index) + "." + class_name + ":" +
                        std::to_string(score),
                    cv::Point2d(lx, ly),
                    cv::FONT_HERSHEY_PLAIN,
                    1,
                    cv::Scalar(255, 255, 255));
      }
      output_index++;
    }
    cv::imwrite(output_path, origin_image);
    end = get_current_us();
    cur_costs[2] = (end - start) / 1000.0f;
    // Statisics
    for (size_t j = 0; j < 3; j++) {
      total_costs[j] += cur_costs[j];
      if (cur_costs[j] > max_costs[j]) {
        max_costs[j] = cur_costs[j];
      }
      if (cur_costs[j] < min_costs[j]) {
        min_costs[j] = cur_costs[j];
      }
    }
    printf(
        "[%d] Preprocess time: %f ms Prediction time: %f ms Postprocess time: "
        "%f ms\n",
        iter_count,
        cur_costs[0],
        cur_costs[1],
        cur_costs[2]);
    iter_count++;
  }
  printf("Preprocess time: avg %f ms, max %f ms, min %f ms\n",
         total_costs[0] / iter_count,
         max_costs[0],
         min_costs[0]);
  printf("Prediction time: avg %f ms, max %f ms, min %f ms\n",
         total_costs[1] / iter_count,
         max_costs[1],
         min_costs[1]);
  printf("Postprocess time: avg %f ms, max %f ms, min %f ms\n",
         total_costs[2] / iter_count,
         max_costs[2],
         min_costs[2]);
  printf("Done.\n");
}

int main(int argc, char **argv) {
  if (argc < 10) {
    printf(
        "Usage: \n"
        "./demo model_dir config_path dataset_dir nnadapter_device_names "
        "nnadapter_context_properties nnadapter_model_cache_dir "
        "nnadapter_model_cache_token nnadapter_subgraph_partition_config_path "
        "nnadapter_mixed_precision_quantization_config_path");
    return -1;
  }
  std::string model_dir = argv[1];
  std::string config_path = argv[2];
  std::string dataset_dir = argv[3];
  std::vector<std::string> nnadapter_device_names =
      split_string<std::string>(argv[4], ',');
  if (nnadapter_device_names.empty()) {
    printf("No device specified.");
    return -1;
  }
  std::string nnadapter_context_properties =
      strcmp(argv[5], "null") == 0 ? "" : argv[5];
  std::string nnadapter_model_cache_dir =
      strcmp(argv[6], "null") == 0 ? "" : argv[6];
  std::string nnadapter_model_cache_token =
      strcmp(argv[7], "null") == 0 ? "" : argv[7];
  std::string nnadapter_subgraph_partition_config_path =
      strcmp(argv[8], "null") == 0 ? "" : argv[8];
  std::string nnadapter_mixed_precision_quantization_config_path =
      strcmp(argv[9], "null") == 0 ? "" : argv[9];

  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
#ifdef USE_FULL_API
  // Run inference by using full api with CxxConfig
  paddle::lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(model_dir);
  cxx_config.set_threads(CPU_THREAD_NUM);
  cxx_config.set_power_mode(CPU_POWER_MODE);
  std::vector<paddle::lite_api::Place> valid_places;
  if (std::find(nnadapter_device_names.begin(),
                nnadapter_device_names.end(),
                "xpu") != nnadapter_device_names.end()) {
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kXPU), PRECISION(kInt8)});
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kXPU), PRECISION(kFloat)});
  } else if (std::find(nnadapter_device_names.begin(),
                       nnadapter_device_names.end(),
                       "opencl") != nnadapter_device_names.end()) {
    valid_places.push_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)});
    valid_places.push_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageFolder)});
    valid_places.push_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)});
    valid_places.push_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)});
    valid_places.push_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageFolder)});
    valid_places.push_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)});
    valid_places.push_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kInt32), DATALAYOUT(kNCHW)});
  } else if (std::find(nnadapter_device_names.begin(),
                       nnadapter_device_names.end(),
                       "cpu") == nnadapter_device_names.end()) {
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
    cxx_config.set_nnadapter_device_names(nnadapter_device_names);
    cxx_config.set_nnadapter_context_properties(nnadapter_context_properties);
    cxx_config.set_nnadapter_model_cache_dir(nnadapter_model_cache_dir);
    // Set the mixed precision quantization configuration file
    if (!nnadapter_mixed_precision_quantization_config_path.empty()) {
      std::vector<char> nnadapter_mixed_precision_quantization_config_buffer;
      if (read_file(nnadapter_mixed_precision_quantization_config_path,
                    &nnadapter_mixed_precision_quantization_config_buffer,
                    false)) {
        if (!nnadapter_mixed_precision_quantization_config_buffer.empty()) {
          std::string nnadapter_mixed_precision_quantization_config_string(
              nnadapter_mixed_precision_quantization_config_buffer.data(),
              nnadapter_mixed_precision_quantization_config_buffer.size());
          cxx_config.set_nnadapter_mixed_precision_quantization_config_buffer(
              nnadapter_mixed_precision_quantization_config_string);
        }
      } else {
        printf(
            "Failed to load the mixed precision quantization configuration "
            "file %s\n",
            nnadapter_mixed_precision_quantization_config_path.c_str());
      }
    }
    // Set the subgraph partition configuration file
    if (!nnadapter_subgraph_partition_config_path.empty()) {
      std::vector<char> nnadapter_subgraph_partition_config_buffer;
      if (read_file(nnadapter_subgraph_partition_config_path,
                    &nnadapter_subgraph_partition_config_buffer,
                    false)) {
        if (!nnadapter_subgraph_partition_config_buffer.empty()) {
          std::string nnadapter_subgraph_partition_config_string(
              nnadapter_subgraph_partition_config_buffer.data(),
              nnadapter_subgraph_partition_config_buffer.size());
          cxx_config.set_nnadapter_subgraph_partition_config_buffer(
              nnadapter_subgraph_partition_config_string);
        }
      } else {
        printf("Failed to load the subgraph partition configuration file %s\n",
               nnadapter_subgraph_partition_config_path.c_str());
      }
    }
  }
#if defined(__arm__) || defined(__aarch64__)
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
#elif defined(__x86_64__)
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kX86), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kX86), PRECISION(kFloat)});
#endif
  cxx_config.set_valid_places(valid_places);
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor(cxx_config);
    if (std::find(nnadapter_device_names.begin(),
                  nnadapter_device_names.end(),
                  "xpu") != nnadapter_device_names.end()) {
      process(predictor, config_path, dataset_dir);
    }
    predictor->SaveOptimizedModel(
        model_dir, paddle::lite_api::LiteModelType::kNaiveBuffer);
  } catch (std::exception e) {
    printf("An internal error occurred in PaddleLite(cxx config).\n");
    return -1;
  }
#endif

  // Run inference by using light api with MobileConfig
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_dir + ".nb");
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(CPU_POWER_MODE);
  if (std::find(nnadapter_device_names.begin(),
                nnadapter_device_names.end(),
                "xpu") != nnadapter_device_names.end()) {
#ifndef USE_FULL_API
    printf("XPU does not support light api!\n");
#endif
    return 0;
  } else if (std::find(nnadapter_device_names.begin(),
                       nnadapter_device_names.end(),
                       "opencl") != nnadapter_device_names.end()) {
    // Check device valid
    if (!paddle::lite_api::IsOpenCLBackendValid()) {
      printf(
          "OpenCL is not supported by the current device, please contact the "
          "device's vendor!\n");
      return 0;
    }
  } else if (std::find(nnadapter_device_names.begin(),
                       nnadapter_device_names.end(),
                       "cpu") == nnadapter_device_names.end()) {
    mobile_config.set_nnadapter_device_names(nnadapter_device_names);
    mobile_config.set_nnadapter_context_properties(
        nnadapter_context_properties);
    // Set the model cache buffer and directory
    mobile_config.set_nnadapter_model_cache_dir(nnadapter_model_cache_dir);
    if (!nnadapter_model_cache_token.empty() &&
        !nnadapter_model_cache_dir.empty()) {
      std::vector<char> nnadapter_model_cache_buffer;
      auto nnadapter_model_cache_path = nnadapter_model_cache_dir + "/" +
                                        nnadapter_model_cache_token + ".nnc";
      if (!read_file(nnadapter_model_cache_path,
                     &nnadapter_model_cache_buffer,
                     true)) {
        printf("Failed to load the cache model file %s\n",
               nnadapter_model_cache_path.c_str());
      }
      if (!nnadapter_model_cache_buffer.empty()) {
        mobile_config.set_nnadapter_model_cache_buffers(
            nnadapter_model_cache_token, nnadapter_model_cache_buffer);
      }
    }
  }
  try {
    predictor =
        paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
            mobile_config);
    process(predictor, config_path, dataset_dir);
  } catch (std::exception e) {
    printf("An internal error occurred in PaddleLite(mobile config).\n");
    return -1;
  }
  return 0;
}
