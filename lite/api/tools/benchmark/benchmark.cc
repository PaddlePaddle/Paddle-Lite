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

#include "lite/api/tools/benchmark/benchmark.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/version.h"
#include "lite/utils/timer.h"

int main(int argc, char *argv[]) {
  return paddle::lite_api::Benchmark(argc, argv);
}

namespace paddle {
namespace lite_api {

int Benchmark(int argc, char **argv) {
  gflags::SetVersionString(lite::version());
  gflags::SetUsageMessage(PrintUsage());
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Check flags validation
  if (!CheckFlagsValid()) {
    std::cout << gflags::ProgramUsage();
    exit(0);
  }

  // Get optimized model file if necessary
  auto model_file = OutputOptModel(FLAGS_optimized_model_file);

  // Get input shapes
  auto input_shapes = lite::GetShapes(FLAGS_input_shape);

  // Run
  Run(model_file, input_shapes);

  return 0;
}

std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cerr << "Failed to open file " << path << std::endl;
    std::abort();
  }
  return m_vec;
}

std::map<std::string, std::string> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, std::string> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = lite::SplitString(config[i]);
    std::cout << "key: " << res[0] << "\t value: " << res[1] << std::endl;
    dict[res[0]] = res[1];
  }
  return dict;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  std::cout << "======= Configuration for Precision Benchmark ======"
            << std::endl;
  for (auto iter = config.begin(); iter != config.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << std::endl;
}

cv::Mat ResizeImage(const cv::Mat &img, const int resize_short_size) {
  int w = img.cols;
  int h = img.rows;

  cv::Mat resize_img;

  float ratio = 1.f;
  if (h < w) {
    ratio = resize_short_size / static_cast<float>(h);
  } else {
    ratio = resize_short_size / static_cast<float>(w);
  }
  int resize_h = round(h * ratio);
  int resize_w = round(w * ratio);

  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  return resize_img;
}

cv::Mat CenterCropImg(const cv::Mat &img, const int crop_size) {
  int resize_w = img.cols;
  int resize_h = img.rows;
  int w_start = static_cast<int>((resize_w - crop_size) / 2);
  int h_start = static_cast<int>((resize_h - crop_size) / 2);
  cv::Rect rect(w_start, h_start, crop_size, crop_size);
  cv::Mat crop_img = img(rect);
  return crop_img;
}

// Fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void NeonMeanScale(const float *din,
                   float *dout,
                   int size,
                   const std::vector<float> mean,
                   const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3!" << std::endl;
    std::abort();
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

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

struct RESULT {
  std::string class_name;
  int class_id;
  float score;
};

std::vector<RESULT> PostProcess(
    std::shared_ptr<PaddlePredictor> predictor,
    const std::map<std::string, std::string> &config,
    const std::vector<std::string> &image_files,
    const std::vector<std::string> &word_labels,
    const int cnt) {
  std::vector<RESULT> results;
  if (image_files.empty()) return results;

  size_t output_tensor_num = predictor->GetOutputNames().size();
  CHECK_EQ(output_tensor_num, 1);
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
  }

  if (stoi(config.at("store_result_as_image")) == 1) {
    cv::Mat img = cv::imread(image_files.at(cnt), cv::IMREAD_COLOR);
    cv::Mat output_image(img);
    for (int i = 0; i < results.size(); i++) {
      auto text = lite::string_format(
          "Top-%d, class_id: %d, class_name: %s, score: %.3f\n",
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
    }
    std::string output_image_path = "./" + std::to_string(cnt) + ".png";
    cv::imwrite(output_image_path, output_image);
    std::cout << "Save output image into " << output_image_path << std::endl;
  }

  return results;
}

void PreProcess(std::shared_ptr<PaddlePredictor> predictor,
                const std::map<std::string, std::string> &config,
                const std::vector<std::string> &image_files,
                const int cnt) {
  if (image_files.empty()) return;

  // Read image
  std::cout << "image: " << image_files.at(cnt) << std::endl;
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
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  const float *dimg = reinterpret_cast<const float *>(img_fp.data);

  auto *input0 = input_tensor->mutable_data<float>();
  NeonMeanScale(dimg, input0, img_fp.rows * img_fp.cols, mean, scale);
}

std::shared_ptr<PaddlePredictor> CreatePredictor(
    const std::string &model_file) {
  MobileConfig config;
  config.set_model_from_file(model_file);
  config.set_threads(FLAGS_threads);
  config.set_power_mode(static_cast<PowerMode>(FLAGS_power_mode));

  // Set backend config info
  SetBackendConfig(config);
  auto predictor = CreatePaddlePredictor(config);

  return predictor;
}

const std::string GetAbsPath(const std::string file_name) {
  char abs_path_buff[PATH_MAX];
  if (realpath(file_name.c_str(), abs_path_buff)) {
    return std::string(abs_path_buff);
  } else {
    std::cerr << "Get abs path error!" << std::endl;
    std::abort();
  }
}

void Run(const std::string &model_file,
         const std::vector<std::vector<int64_t>> &input_shapes) {
  lite::Timer timer;
  std::vector<float> perf_vct;
  std::map<std::string, std::string> config;
  std::vector<std::string> image_files;
  std::vector<std::string> word_labels;

  // Create predictor
  timer.Start();
  auto predictor = CreatePredictor(model_file);
  float init_time = timer.Stop();

  // Set inputs
  if (FLAGS_validation_set.empty()) {
    for (size_t i = 0; i < input_shapes.size(); i++) {
      auto input_tensor = predictor->GetInput(i);
      input_tensor->Resize(input_shapes[i]);
      // NOTE: Change input data type to other type as you need.
      auto input_data = input_tensor->mutable_data<float>();
      auto input_num = lite::ShapeProduction(input_shapes[i]);
      if (FLAGS_input_data_path.empty()) {
        for (auto j = 0; j < input_num; j++) {
          input_data[j] = 1.f;
        }
      } else {
        auto paths = lite::SplitString(FLAGS_input_data_path);
        std::ifstream fs(paths[i]);
        if (!fs.is_open()) {
          std::cerr << "Open input image " << paths[i] << " error."
                    << std::endl;
        }
        for (int k = 0; k < input_num; k++) {
          fs >> input_data[k];
        }
        fs.close();
      }
    }
  } else {
    std::cout << "FLAGS_config_path: " << FLAGS_config_path << std::endl;
    config = LoadConfigTxt(FLAGS_config_path);
    PrintConfig(config);
    word_labels = lite::ReadLines(config.at("label_path"));

    std::vector<std::string> image_labels =
        lite::ReadLines(config.at("ground_truth_images_path"));
    image_files.reserve(image_labels.size());
    for (auto line : image_labels) {
      auto path = GetAbsPath(config.at("ground_truth_images_path"));
      auto dir = path.substr(0, path.find_last_of("/") + 1);
      line = dir + line;
      std::string image_file = line.substr(0, line.find(" "));
      std::string label_id = line.substr(line.find(" ") + 1, line.length());
      image_files.push_back(image_file);
    }
  }

  // Warmup
  for (int i = 0; i < FLAGS_warmup; ++i) {
    if (i == 0) {
      timer.Start();
      PreProcess(predictor, config, image_files, i);
      predictor->Run();
      auto results =
          PostProcess(predictor, config, image_files, word_labels, i);
      perf_vct.push_back(timer.Stop());
      std::cout << "===clas result for image: " << image_files.at(i)
                << "===" << std::endl;
      for (int i = 0; i < results.size(); i++) {
        std::cout << "\t"
                  << "Top-" << i + 1 << ", class_id: " << results[i].class_id
                  << ", class_name: " << results[i].class_name
                  << ", score: " << results[i].score << std::endl;
      }
    } else {
      predictor->Run();
    }
    timer.SleepInMs(FLAGS_run_delay);
  }

  // Run
  for (int i = 0; i < FLAGS_repeats; ++i) {
    timer.Start();
    predictor->Run();
    perf_vct.push_back(timer.Stop());
    timer.SleepInMs(FLAGS_run_delay);
  }

  // Get output
  size_t output_tensor_num = predictor->GetOutputNames().size();
  std::stringstream out_ss;
  out_ss << "output tensor num: " << output_tensor_num;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const Tensor> output_tensor = predictor->GetOutput(tidx);
    out_ss << "\n--- output tensor " << tidx << " ---\n";
    auto out_shape = output_tensor->shape();
    auto out_data = output_tensor->data<float>();
    auto ele_num = lite::ShapeProduction(out_shape);
    auto out_mean = lite::compute_mean<float>(out_data, ele_num);
    auto out_std_dev = lite::compute_standard_deviation<float>(
        out_data, ele_num, true, out_mean);

    out_ss << "output shape(NCHW): " << lite::ShapePrint(out_shape)
           << std::endl;
    out_ss << "output tensor " << tidx << " elem num: " << ele_num << std::endl;
    out_ss << "output tensor " << tidx << " mean value: " << out_mean
           << std::endl;
    out_ss << "output tensor " << tidx << " standard deviation: " << out_std_dev
           << std::endl;

    if (FLAGS_show_output_elem) {
      for (int i = 0; i < ele_num; ++i) {
        out_ss << "out[" << tidx << "][" << i
               << "]:" << output_tensor->data<float>()[i] << std::endl;
      }
    }
  }

  // Save benchmark info
  float first_time = perf_vct[0];
  if (FLAGS_warmup > 0) {
    perf_vct.erase(perf_vct.cbegin());
  }
  std::stable_sort(perf_vct.begin(), perf_vct.end());
  float perf_avg =
      std::accumulate(perf_vct.begin(), perf_vct.end(), 0.0) / FLAGS_repeats;

  std::stringstream ss;
  ss.precision(3);
#ifdef __ANDROID__
  ss << "\n======= Device Info =======\n";
  ss << GetDeviceInfo();
#endif
  ss << "\n======= Model Info =======\n";
  ss << "optimized_model_file: " << model_file << std::endl;
  ss << "input_data_path: "
     << (FLAGS_input_data_path.empty() ? "All 1.f" : FLAGS_input_data_path)
     << std::endl;
  ss << "input_shape: " << FLAGS_input_shape << std::endl;
  ss << out_ss.str();
  ss << "\n======= Runtime Info =======\n";
  ss << "benchmark_bin version: " << lite::version() << std::endl;
  ss << "threads: " << FLAGS_threads << std::endl;
  ss << "power_mode: " << FLAGS_power_mode << std::endl;
  ss << "warmup: " << FLAGS_warmup << std::endl;
  ss << "repeats: " << FLAGS_repeats << std::endl;
  if (FLAGS_run_delay > 0.f) {
    ss << "run_delay(sec): " << FLAGS_run_delay << std::endl;
  }
  ss << "result_path: " << FLAGS_result_path << std::endl;
  ss << "\n======= Backend Info =======\n";
  ss << "backend: " << FLAGS_backend << std::endl;
  ss << "cpu precision: " << FLAGS_cpu_precision << std::endl;
  if (FLAGS_backend == "opencl,arm" || FLAGS_backend == "opencl" ||
      FLAGS_backend == "opencl,x86" || FLAGS_backend == "x86_opencl") {
    ss << "gpu precision: " << FLAGS_gpu_precision << std::endl;
    ss << "opencl_cache_dir: " << FLAGS_opencl_cache_dir << std::endl;
    ss << "opencl_kernel_cache_file: " << FLAGS_opencl_kernel_cache_file
       << std::endl;
    ss << "opencl_tuned_file: " << FLAGS_opencl_tuned_file << std::endl;
  }
  ss << "\n======= Perf Info =======\n";
  ss << std::fixed << std::left;
  ss << "Time(unit: ms):\n";
  ss << "init  = " << std::setw(12) << init_time << std::endl;
  ss << "first = " << std::setw(12) << first_time << std::endl;
  ss << "min   = " << std::setw(12) << perf_vct.front() << std::endl;
  ss << "max   = " << std::setw(12) << perf_vct.back() << std::endl;
  ss << "avg   = " << std::setw(12) << perf_avg << std::endl;
  if (FLAGS_enable_memory_profile) {
    ss << "\nMemory Usage(unit: kB):\n";
    ss << "init  = " << std::setw(12) << "Not supported yet" << std::endl;
    ss << "avg   = " << std::setw(12) << "Not supported yet" << std::endl;
  }
  std::cout << ss.str() << std::endl;
  StoreBenchmarkResult(ss.str());
}

}  // namespace lite_api
}  // namespace paddle
