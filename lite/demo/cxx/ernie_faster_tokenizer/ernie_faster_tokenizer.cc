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

#include "ernie_faster_tokenizer.h"  // NOLINT
#include <sys/time.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api;  // NOLINT
using namespace paddlenlp;         // NOLINT

template <typename T>
class MyTensor {
 public:
  MyTensor() {}
  ~MyTensor() {
    if (data_) delete data_;
  }
  const T* data() { return reinterpret_cast<const T*>(data_); }
  T* mutable_data() {
    if (data_ != nullptr) delete data_;
    data_ = new T[numel_];
    return data_;
  }
  shape_t shape() { return shape_; }
  void Resize(shape_t shape) {
    shape_ = shape;
    numel_ = 1;
    for (int i = 0; i < shape.size(); i++) numel_ *= shape[i];
  }
  void CopyFrom(const T* src) { memcpy(data_, src, numel_ * sizeof(T)); }
  int64_t numel() { return numel_; }

 private:
  int64_t numel_{1};
  shape_t shape_;
  T* data_{nullptr};
};

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

// Only useful for axis = -1
template <typename T>
void Softmax(MyTensor<T>& input, MyTensor<T>* output) {  // NOLINT
  auto softmax_func = [](const T* score_vec, T* softmax_vec, int label_num) {
    double score_max = *(std::max_element(score_vec, score_vec + label_num));
    double e_sum = 0;
    for (int j = 0; j < label_num; j++) {
      softmax_vec[j] = std::exp(score_vec[j] - score_max);
      e_sum += softmax_vec[j];
    }
    for (int k = 0; k < label_num; k++) {
      softmax_vec[k] /= e_sum;
    }
  };
  output->Resize(input.shape());
  T* output_ptr = output->mutable_data();
  const T* input_ptr = input.data();
  int label_num = output->shape().back();
  int batch_size = ShapeProduction(input.shape()) / label_num;
  int offset = 0;
  for (int i = 0; i < batch_size; ++i) {
    softmax_func(input_ptr + offset, output_ptr + offset, label_num);
    offset += label_num;
  }
}

// Only useful for axis = -1
template <typename T>
void Max(MyTensor<T>& input, MyTensor<T>* output) {  // NOLINT
  shape_t output_shape;
  for (int i = 0; i < input.shape().size() - 1; ++i) {
    output_shape.push_back(input.shape()[i]);
  }
  output_shape.push_back(1);
  output->Resize(output_shape);
  T* output_ptr = output->mutable_data();
  const T* input_ptr = input.data();
  int batch_size = ShapeProduction(output_shape);
  int label_num = input.shape().back();
  int offset = 0;
  for (int i = 0; i < batch_size; ++i) {
    output_ptr[i] =
        *(std::max_element(input_ptr + offset, input_ptr + offset + label_num));
    offset += label_num;
  }
}

template <typename T>
void ViterbiDecode(MyTensor<T>& slot_logits,        // NOLINT
                   MyTensor<T>& trans,              // NOLINT
                   MyTensor<int64_t>* best_path) {  // NOLINT
  int batch_size = slot_logits.shape()[0];
  int seq_len = slot_logits.shape()[1];
  int num_tags = slot_logits.shape()[2];
  best_path->Resize({batch_size, seq_len});
  int64_t* best_path_ptr = best_path->mutable_data();
  const T* slot_logits_ptr = reinterpret_cast<const T*>(slot_logits.data());
  const T* trans_ptr = reinterpret_cast<const T*>(trans.data());
  std::vector<T> scores(num_tags);
  std::copy(slot_logits_ptr, slot_logits_ptr + num_tags, scores.begin());
  std::vector<std::vector<T>> M(num_tags, std::vector<T>(num_tags));
  for (int b = 0; b < batch_size; ++b) {
    std::vector<std::vector<int>> paths;
    const T* curr_slot_logits_ptr = slot_logits_ptr + b * seq_len * num_tags;
    int64_t* curr_best_path_ptr = best_path_ptr + b * seq_len;
    for (int t = 1; t < seq_len; t++) {
      for (size_t i = 0; i < num_tags; i++) {
        for (size_t j = 0; j < num_tags; j++) {
          auto trans_idx = i * num_tags * num_tags + j * num_tags;
          auto slot_logit_idx = t * num_tags + j;
          M[i][j] = scores[i] + trans_ptr[trans_idx] +
                    curr_slot_logits_ptr[slot_logit_idx];
        }
      }
      std::vector<int> idxs;
      for (size_t i = 0; i < num_tags; i++) {
        T max = 0.0f;
        int idx = 0;
        for (size_t j = 0; j < num_tags; j++) {
          if (M[j][i] > max) {
            max = M[j][i];
            idx = j;
          }
        }
        scores[i] = max;
        idxs.push_back(idx);
      }
      paths.push_back(idxs);
    }
    int scores_max_index = 0;
    float scores_max = 0.0f;
    for (size_t i = 0; i < scores.size(); i++) {
      if (scores[i] > scores_max) {
        scores_max = scores[i];
        scores_max_index = i;
      }
    }
    curr_best_path_ptr[seq_len - 1] = scores_max_index;
    for (int i = seq_len - 2; i >= 0; i--) {
      int index = curr_best_path_ptr[i + 1];
      curr_best_path_ptr[i] = paths[i][index];
    }
  }
}

void LoadTransitionFromFile(const std::string& file,
                            std::vector<float>* transitions,
                            int* num_tags) {
  std::ifstream fin(file);
  std::string curr_transition;
  float transition;
  int i = 0;
  while (fin) {
    std::getline(fin, curr_transition);
    std::istringstream iss(curr_transition);
    while (iss) {
      iss >> transition;
      transitions->push_back(transition);
    }
    if (curr_transition != "") {
      ++i;
    }
  }
  *num_tags = i;
}

void RunModel(std::string model_dir,
              int batch,
              int seq_len,
              const std::vector<int64_t>& input_ids,
              const std::vector<int64_t>& token_type_ids,
              size_t repeats,
              size_t warmup,
              size_t power_mode,
              size_t thread_num,
              size_t accelerate_opencl) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_dir);

  // NOTE: Use android gpu with opencl, you should ensure:
  //  first, [compile **cpu+opencl** paddlelite
  //    lib](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/opencl.md);
  //  second, [convert and use opencl nb
  //    model](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/user_guides/opt/opt_bin.md).
  bool is_opencl_backend_valid =
      ::IsOpenCLBackendValid(/*check_fp16_valid = false*/);
  std::cout << "is_opencl_backend_valid:"
            << (is_opencl_backend_valid ? "true" : "false") << std::endl;
  if (is_opencl_backend_valid) {
    if (accelerate_opencl != 0) {
      // Set opencl kernel binary.
      // Large addtitional prepare time is cost due to algorithm selecting and
      // building kernel from source code.
      // Prepare time can be reduced dramitically after building algorithm file
      // and OpenCL kernel binary on the first running.
      // The 1st running time will be a bit longer due to the compiling time if
      // you don't call `set_opencl_binary_path_name` explicitly.
      // So call `set_opencl_binary_path_name` explicitly is strongly
      // recommended.

      // Make sure you have write permission of the binary path.
      // We strongly recommend each model has a unique binary name.
      const std::string bin_path = "/data/local/tmp/";
      const std::string bin_name = "lite_opencl_kernel.bin";
      config.set_opencl_binary_path_name(bin_path, bin_name);

      // opencl tune option
      // CL_TUNE_NONE: 0
      // CL_TUNE_RAPID: 1
      // CL_TUNE_NORMAL: 2
      // CL_TUNE_EXHAUSTIVE: 3
      const std::string tuned_path = "/data/local/tmp/";
      const std::string tuned_name = "lite_opencl_tuned.bin";
      config.set_opencl_tune(CL_TUNE_NORMAL, tuned_path, tuned_name);

      // opencl precision option
      // CL_PRECISION_AUTO: 0, first fp16 if valid, default
      // CL_PRECISION_FP32: 1, force fp32
      // CL_PRECISION_FP16: 2, force fp16
      config.set_opencl_precision(CL_PRECISION_FP16);
    }
  } else {
    std::cout << "*** nb model will be running on cpu. ***" << std::endl;
    // you can give backup cpu nb model instead
    // config.set_model_from_file(cpu_nb_model_dir);
  }

  // NOTE: To load model transformed by model_optimize_tool before
  // release/v2.3.0, plese use `set_model_dir` API as listed below.
  // config.set_model_dir(model_dir);
  config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));
  config.set_threads(thread_num);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data
  for (int i = 0; i < input_ids.size(); i++) {
    auto input_tensor = predictor->GetInput(0);
    input_tensor->Resize({batch, seq_len});
    auto input_data = input_tensor->mutable_data<int64_t>();
    for (int i = 0; i < input_ids.size(); ++i) {
      input_data[i] = input_ids[i];
    }
  }
  for (int i = 0; i < token_type_ids.size(); i++) {
    auto input_tensor = predictor->GetInput(1);
    input_tensor->Resize({batch, seq_len});
    auto input_data = input_tensor->mutable_data<int64_t>();
    for (int i = 0; i < token_type_ids.size(); ++i) {
      input_data[i] = token_type_ids[i];
    }
  }

  // 4. Run predictor
  double first_duration{-1};
  for (size_t widx = 0; widx < warmup; ++widx) {
    if (widx == 0) {
      auto start = GetCurrentUS();
      predictor->Run();
      first_duration = (GetCurrentUS() - start) / 1000.0;
    } else {
      predictor->Run();
    }
  }

  double sum_duration = 0.0;  // millisecond;
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    auto start = GetCurrentUS();

    predictor->Run();

    auto duration = (GetCurrentUS() - start) / 1000.0;
    sum_duration += duration;
    max_duration = duration > max_duration ? duration : max_duration;
    min_duration = duration < min_duration ? duration : min_duration;
    std::cout << "run_idx:" << ridx + 1 << " / " << repeats << ": " << duration
              << " ms" << std::endl;
    if (first_duration < 0) {
      first_duration = duration;
    }
  }
  avg_duration = sum_duration / static_cast<float>(repeats);
  std::cout << "\n======= infer benchmark summary =======\n"
            << "model_dir:" << model_dir << "\n"
            << "warmup:" << warmup << "\n"
            << "repeats:" << repeats << "\n"
            << "power_mode:" << power_mode << "\n"
            << "thread_num:" << thread_num << "\n"
            << "*** time info(ms) ***\n"
            << "1st_duration:" << first_duration << "\n"
            << "max_duration:" << max_duration << "\n"
            << "min_duration:" << min_duration << "\n"
            << "avg_duration:" << avg_duration << "\n";

  // Postprocess
  std::unique_ptr<const paddle::lite_api::Tensor> tensor0 =
      predictor->GetOutput(0);
  std::unique_ptr<const paddle::lite_api::Tensor> tensor1 =
      predictor->GetOutput(1);
  std::unique_ptr<const paddle::lite_api::Tensor> tensor2 =
      predictor->GetOutput(2);
  MyTensor<float> output_tensor0, output_tensor1, output_tensor2;
  auto start = GetCurrentUS();
  output_tensor0.Resize(tensor0->shape());
  output_tensor0.mutable_data();
  output_tensor0.CopyFrom(tensor0->data<float>());
  output_tensor1.Resize(tensor1->shape());
  output_tensor1.mutable_data();
  output_tensor1.CopyFrom(tensor1->data<float>());
  output_tensor2.Resize(tensor2->shape());
  output_tensor2.mutable_data();
  output_tensor2.CopyFrom(tensor2->data<float>());
  MyTensor<float> domain_probs, intent_probs;
  Softmax<float>(output_tensor0, &domain_probs);
  Softmax<float>(output_tensor1, &intent_probs);
  MyTensor<float> domain_max_probs, intent_max_probs;
  Max<float>(domain_probs, &domain_max_probs);
  Max<float>(intent_probs, &intent_max_probs);
  std::vector<float> transition;
  int num_tags;
  LoadTransitionFromFile("joint_transition.txt", &transition, &num_tags);
  MyTensor<float> trans;
  trans.Resize((shape_t){num_tags, num_tags});
  float* trans_ptr = trans.mutable_data();
  memcpy(trans_ptr, transition.data(), transition.size() * sizeof(float));
  MyTensor<int64_t> best_path;
  ViterbiDecode<float>(output_tensor2, trans, &best_path);
  auto duration = (GetCurrentUS() - start) / 1000.0;
  std::cout << "\n\nPostProcess cost " << duration << " ms\n";
  std::cout << "\n=============== Output Message ==============\n";
  auto batch_size = best_path.shape()[0];
  auto seq_len_out = best_path.shape()[1];
  const int64_t* best_path_ptr =
      reinterpret_cast<const int64_t*>(best_path.data());
  for (int i = 0; i < batch_size; i++) {
    std::cout << "\nbatch = " << i << "\n";
    std::cout << "domain_max_probs = " << domain_max_probs.data()[i] << "\n";
    std::cout << "intent_max_probs = " << intent_max_probs.data()[i] << "\n";
    std::cout << "best_path[" << i << "] = ";
    for (int j = 0; j < seq_len_out; ++j) {
      std::cout << best_path_ptr[i * seq_len_out + j] << ", ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char** argv) {
  // 1. Define a ernie faster tokenizer
  faster_tokenizer::tokenizers_impl::ErnieFasterTokenizer tokenizer(
      "ernie_vocab.txt");
  std::vector<faster_tokenizer::core::EncodeInput> strings_list = {
      "导航去科技园二号楼", "屏幕亮度为我减小一点吧"};
  std::vector<faster_tokenizer::core::Encoding> encodings;
  auto start = GetCurrentUS();
  tokenizer.EncodeBatchStrings(strings_list, &encodings);
  size_t batch_size = strings_list.size();
  size_t seq_len = encodings[0].GetLen();
  for (auto&& encoding : encodings) {
    std::cout << encoding.DebugString() << std::endl;
  }
  std::string model_dir = "./model.nb";

  // 2. Construct input vector
  // 2.1 Convert encodings to input_ids, token_type_ids
  std::vector<int64_t> input_ids, token_type_ids;
  for (int i = 0; i < encodings.size(); ++i) {
    auto&& curr_input_ids = encodings[i].GetIds();
    auto&& curr_type_ids = encodings[i].GetTypeIds();
    input_ids.insert(
        input_ids.end(), curr_input_ids.begin(), curr_input_ids.end());
    token_type_ids.insert(
        token_type_ids.end(), curr_type_ids.begin(), curr_type_ids.end());
  }
  auto init_time = (GetCurrentUS() - start) / 1000.0;
  std::cout << "EncodeBatchStrings cost " << init_time << " ms\n\n";

  size_t repeat = 1;
  size_t warm_up = 0;
  size_t power_mode = 0;
  size_t thread_num = 1;
  size_t accelerate_opencl = 0;
  if (argc >= 2) {
    repeat = atoi(argv[1]);
    warm_up = atoi(argv[2]);
    power_mode = atoi(argv[3]);
    thread_num = atoi(argv[4]);
    accelerate_opencl = atoi(argv[5]);
  }

  // 3. infer
  RunModel(model_dir,
           batch_size,
           seq_len,
           input_ids,
           token_type_ids,
           repeat,
           warm_up,
           power_mode,
           thread_num,
           accelerate_opencl);
}
