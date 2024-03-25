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
  int width;
  int height;
  std::vector<float> mean;
  std::vector<float> std;
  float draw_threshold{0.5f};
  bool use_dark_decode{true};
} CONFIG;

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
  // use_dark_decode(optional)
  if (values.count("use_dark_decode")) {
    config.use_dark_decode = !(values["use_dark_decode"] == "false" ||
                               values["use_dark_decode"] == "0");
  }
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

#define PI 3.1415926535
#define HALF_CIRCLE_DEGREE 180

cv::Point2f get_3rd_point(cv::Point2f &a, cv::Point2f &b) {  // NOLINT
  cv::Point2f direct{a.x - b.x, a.y - b.y};
  return cv::Point2f(a.x - direct.y, a.y + direct.x);
}

std::vector<float> get_dir(float src_point_x,
                           float src_point_y,
                           float rot_rad) {
  float sn = sin(rot_rad);
  float cs = cos(rot_rad);
  std::vector<float> src_result{0.0, 0.0};
  src_result[0] = src_point_x * cs - src_point_y * sn;
  src_result[1] = src_point_x * sn + src_point_y * cs;
  return src_result;
}

void affine_tranform(float pt_x,
                     float pt_y,
                     cv::Mat &trans,             // NOLINT
                     std::vector<float> &preds,  // NOLINT
                     int p) {                    // NOLINT
  double new1[3] = {pt_x, pt_y, 1.0};
  cv::Mat new_pt(3, 1, trans.type(), new1);
  cv::Mat w = trans * new_pt;
  preds[p * 3 + 1] = static_cast<float>(w.at<double>(0, 0));
  preds[p * 3 + 2] = static_cast<float>(w.at<double>(1, 0));
}

void get_affine_transform(std::vector<float> &center,  // NOLINT
                          std::vector<float> &scale,   // NOLINT
                          float rot,
                          std::vector<int> &output_size,  // NOLINT
                          cv::Mat &trans,                 // NOLINT
                          int inv) {
  float src_w = scale[0];
  float dst_w = static_cast<float>(output_size[0]);
  float dst_h = static_cast<float>(output_size[1]);
  float rot_rad = rot * PI / HALF_CIRCLE_DEGREE;
  std::vector<float> src_dir = get_dir(-0.5 * src_w, 0, rot_rad);
  std::vector<float> dst_dir{static_cast<float>(-0.5) * dst_w, 0.0};
  cv::Point2f srcPoint2f[3], dstPoint2f[3];
  srcPoint2f[0] = cv::Point2f(center[0], center[1]);
  srcPoint2f[1] = cv::Point2f(center[0] + src_dir[0], center[1] + src_dir[1]);
  srcPoint2f[2] = get_3rd_point(srcPoint2f[0], srcPoint2f[1]);

  dstPoint2f[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
  dstPoint2f[1] =
      cv::Point2f(dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]);
  dstPoint2f[2] = get_3rd_point(dstPoint2f[0], dstPoint2f[1]);
  if (inv == 0) {
    trans = cv::getAffineTransform(srcPoint2f, dstPoint2f);
  } else {
    trans = cv::getAffineTransform(dstPoint2f, srcPoint2f);
  }
}

void transform_preds(std::vector<float> &coords,         // NOLINT
                     std::vector<float> &center,         // NOLINT
                     std::vector<float> &scale,          // NOLINT
                     std::vector<int> &output_size,      // NOLINT
                     std::vector<int64_t> &shape,        // NOLINT
                     std::vector<float> &target_coords,  // NOLINT
                     bool affine = false) {
  if (affine) {
    cv::Mat trans(2, 3, CV_64FC1);
    get_affine_transform(center, scale, 0, output_size, trans, 1);
    for (int p = 0; p < shape[1]; ++p) {
      affine_tranform(
          coords[p * 2], coords[p * 2 + 1], trans, target_coords, p);
    }
  } else {
    float heat_w = static_cast<float>(output_size[0]);
    float heat_h = static_cast<float>(output_size[1]);
    float x_scale = scale[0] / heat_w;
    float y_scale = scale[1] / heat_h;
    float offset_x = center[0] - scale[0] / 2.;
    float offset_y = center[1] - scale[1] / 2.;
    for (int i = 0; i < shape[1]; i++) {
      target_coords[i * 3 + 1] = x_scale * coords[i * 2] + offset_x;
      target_coords[i * 3 + 2] = y_scale * coords[i * 2 + 1] + offset_y;
    }
  }
}

void dark_parse(const float *heatmap,
                std::vector<int64_t> &shape,  // NOLINT
                std::vector<float> &coords,   // NOLINT
                int px,
                int py,
                int index,
                int ch) {
  /*DARK postpocessing, Zhang et al. Distribution-Aware Coordinate
  Representation for Human Pose Estimation (CVPR 2020).
  1) offset = - hassian.inv() * derivative
  2) dx = (heatmap[x+1] - heatmap[x-1])/2.
  3) dxx = (dx[x+1] - dx[x-1])/2.
  4) derivative = Mat([dx, dy])
  5) hassian = Mat([[dxx, dxy], [dxy, dyy]])
  */
  std::vector<float> heatmap_ch(heatmap + index,
                                heatmap + index + shape[2] * shape[3]);
  cv::Mat heatmap_mat = cv::Mat(heatmap_ch).reshape(0, shape[2]);
  heatmap_mat.convertTo(heatmap_mat, CV_32FC1);
  cv::GaussianBlur(heatmap_mat, heatmap_mat, cv::Size(3, 3), 0, 0);
  heatmap_mat = heatmap_mat.reshape(1, 1);
  heatmap_ch = std::vector<float>(heatmap_mat.reshape(1, 1));

  float epsilon = 1e-10;
  // sample heatmap to get values in around target location
  float xy = log(fmax(heatmap_ch[py * shape[3] + px], epsilon));
  float xr = log(fmax(heatmap_ch[py * shape[3] + px + 1], epsilon));
  float xl = log(fmax(heatmap_ch[py * shape[3] + px - 1], epsilon));

  float xr2 = log(fmax(heatmap_ch[py * shape[3] + px + 2], epsilon));
  float xl2 = log(fmax(heatmap_ch[py * shape[3] + px - 2], epsilon));
  float yu = log(fmax(heatmap_ch[(py + 1) * shape[3] + px], epsilon));
  float yd = log(fmax(heatmap_ch[(py - 1) * shape[3] + px], epsilon));
  float yu2 = log(fmax(heatmap_ch[(py + 2) * shape[3] + px], epsilon));
  float yd2 = log(fmax(heatmap_ch[(py - 2) * shape[3] + px], epsilon));
  float xryu = log(fmax(heatmap_ch[(py + 1) * shape[3] + px + 1], epsilon));
  float xryd = log(fmax(heatmap_ch[(py - 1) * shape[3] + px + 1], epsilon));
  float xlyu = log(fmax(heatmap_ch[(py + 1) * shape[3] + px - 1], epsilon));
  float xlyd = log(fmax(heatmap_ch[(py - 1) * shape[3] + px - 1], epsilon));

  // Compute dx/dy and dxx/dyy with sampled values
  float dx = 0.5 * (xr - xl);
  float dy = 0.5 * (yu - yd);
  float dxx = 0.25 * (xr2 - 2 * xy + xl2);
  float dxy = 0.25 * (xryu - xryd - xlyu + xlyd);
  float dyy = 0.25 * (yu2 - 2 * xy + yd2);

  // Finally get offset by derivative and hassian, which combined by dx/dy and
  // dxx/dyy
  if (dxx * dyy - dxy * dxy != 0) {
    float M[2][2] = {dxx, dxy, dxy, dyy};
    float D[2] = {dx, dy};
    cv::Mat hassian(2, 2, CV_32F, M);
    cv::Mat derivative(2, 1, CV_32F, D);
    cv::Mat offset = -hassian.inv() * derivative;
    coords[ch * 2] += offset.at<float>(0, 0);
    coords[ch * 2 + 1] += offset.at<float>(1, 0);
  }
}

void get_final_preds(const float *heatmap,
                     std::vector<int64_t> &shape,  // NOLINT
                     const int64_t *index,
                     std::vector<float> &center,  // NOLINT
                     std::vector<float> scale,
                     std::vector<float> &preds,  // NOLINT
                     bool DARK) {
  std::vector<float> coords;
  coords.resize(shape[1] * 2);
  int heatmap_height = shape[2];
  int heatmap_width = shape[3];

  for (int j = 0; j < shape[1]; ++j) {
    int offset = j * shape[2] * shape[3];
    int idx = index[j];
    preds[j * 3] = heatmap[offset + idx];
    coords[j * 2] = idx % heatmap_width;
    coords[j * 2 + 1] = idx / heatmap_width;

    int px = static_cast<int>(coords[j * 2] + 0.5);
    int py = static_cast<int>(coords[j * 2 + 1] + 0.5);

    if (DARK && px > 1 && px < heatmap_width - 2) {
      dark_parse(heatmap, shape, coords, px, py, offset, j);
    } else {
      if (px > 0 && px < heatmap_width - 1) {
        float diff_x = heatmap[offset + py * shape[3] + px + 1] -
                       heatmap[offset + py * shape[3] + px - 1];
        coords[j * 2] += diff_x > 0 ? 1 : -1 * 0.25;
      }
      if (py > 0 && py < heatmap_height - 1) {
        float diff_y = heatmap[offset + (py + 1) * shape[3] + px] -
                       heatmap[offset + (py - 1) * shape[3] + px];
        coords[j * 2 + 1] += diff_y > 0 ? 1 : -1 * 0.25;
      }
    }
  }

  std::vector<int> img_size{heatmap_width, heatmap_height};
  transform_preds(coords, center, scale, img_size, shape, preds);
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
             const std::string &config_path,
             const std::string &dataset_dir) {
  // Parse the config file to extract the model info
  auto config = load_config(config_path);
  // Load dataset list
  auto dataset = load_dataset(dataset_dir + "/list.txt");
  // Prepare for inference and warmup
  auto image_tensor = predictor->GetInput(0);
  image_tensor->Resize({1, 3, config.height, config.width});
  auto image_data = image_tensor->mutable_data<float>();
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
    double end = get_current_us();
    cur_costs[0] = (end - start) / 1000.0f;
    // Inference
    start = get_current_us();
    predictor->Run();
    end = get_current_us();
    cur_costs[1] = (end - start) / 1000.0f;
    // Postprocess
    start = get_current_us();
    auto heatmap_tensor = predictor->GetOutput(0);
    auto heatmap_data = heatmap_tensor->data<float>();
    auto heatmap_shape = heatmap_tensor->shape();
    auto heatmap_size = shape_production(heatmap_shape);
    if (heatmap_size < 6) {
      printf("[WARNING] No object detected.\n");
    }
    auto index_tensor = predictor->GetOutput(1);
    auto index_data = index_tensor->data<int64_t>();
    auto index_shape = index_tensor->shape();
    auto index_size = shape_production(index_shape);
    std::vector<float> preds(heatmap_shape[1] * 3, 0);
    std::vector<float> center = {static_cast<float>(origin_image.cols) / 2.0f,
                                 static_cast<float>(origin_image.rows) / 2.0f};
    std::vector<float> scale = {static_cast<float>(origin_image.cols),
                                static_cast<float>(origin_image.rows)};
    get_final_preds(heatmap_data,
                    heatmap_shape,
                    index_data,
                    center,
                    scale,
                    preds,
                    config.use_dark_decode);
    const int edge[][2] = {{0, 1},
                           {0, 2},
                           {1, 3},
                           {2, 4},
                           {3, 5},
                           {4, 6},
                           {5, 7},
                           {6, 8},
                           {7, 9},
                           {8, 10},
                           {5, 11},
                           {6, 12},
                           {11, 13},
                           {12, 14},
                           {13, 15},
                           {14, 16},
                           {11, 12}};
    auto num_joints = heatmap_shape[1];
    for (int i = 0; i < num_joints; i++) {
      if (preds[i * 3] > config.draw_threshold) {
        int x_coord = static_cast<int>(preds[i * 3 + 1]);
        int y_coord = static_cast<int>(preds[i * 3 + 2]);
        cv::circle(origin_image,
                   cv::Point2d(x_coord, y_coord),
                   1,
                   cv::Scalar(0, 0, 255),
                   2);
      }
      if (preds[edge[i][0] * 3] > config.draw_threshold &&
          preds[edge[i][1] * 3] > config.draw_threshold) {
        int x_start = static_cast<int>(preds[edge[i][0] * 3 + 1]);
        int y_start = static_cast<int>(preds[edge[i][0] * 3 + 2]);
        int x_end = static_cast<int>(preds[edge[i][1] * 3 + 1]);
        int y_end = static_cast<int>(preds[edge[i][1] * 3 + 2]);
        cv::line(origin_image,
                 cv::Point2d(x_start, y_start),
                 cv::Point2d(x_end, y_end),
                 cv::Scalar(0, 255, 255),
                 1);
      }
      printf("[%d] %f, %f - %f\n",
             i,
             preds[i * 3 + 1],
             preds[i * 3 + 2],
             preds[i * 3]);
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
