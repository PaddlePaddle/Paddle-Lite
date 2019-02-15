// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "io/api_paddle_mobile.h"
#include <vector>
#include "common/enforce.h"
#include "framework/tensor.h"

namespace paddle_mobile {

template <typename Device, typename T>
PaddleMobilePredictor<Device, T>::PaddleMobilePredictor(
    const PaddleMobileConfig &config) {
  PADDLE_MOBILE_ENFORCE(Init(config) == true,
                        "paddle mobile predictor init failed!");
  config_ = config;
}

template <typename Device, typename T>
bool PaddleMobilePredictor<Device, T>::Init(const PaddleMobileConfig &config) {
  paddle_mobile_.reset(new PaddleMobile<Device, T>());
#ifdef PADDLE_MOBILE_CL
  paddle_mobile_->SetCLPath(config.cl_path);
#endif
  if (config.memory_pack.from_memory) {
    DLOG << "load from memory!";
    paddle_mobile_->LoadCombinedMemory(config.memory_pack.model_size,
                                       config.memory_pack.model_buf,
                                       config.memory_pack.combined_params_size,
                                       config.memory_pack.combined_params_buf);
  } else if (!config.model_dir.empty()) {
    paddle_mobile_->Load(config.model_dir, config.optimize,
                         config.quantification, config.batch_size);
  } else if (!config.prog_file.empty() && !config.param_file.empty()) {
    paddle_mobile_->Load(config.prog_file, config.param_file, config.optimize,
                         config.quantification, config.batch_size);
  } else {
    LOG(kLOG_ERROR) << "fail to load inference model!";
    return false;
  }
  // If the openmp is open, set the thread num
  paddle_mobile_->SetThreadNum(config.thread_num);
  return true;
}
template <typename Device, typename T>
bool PaddleMobilePredictor<Device, T>::Run(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data, int batch_size) {
  if (inputs.empty()) {
    LOG(kLOG_ERROR) << "At least one output should be set with tensors' names.";
    return false;
  }
  auto input = inputs[0];

  if (input.shape.size() != 4) {
    LOG(kLOG_ERROR) << "input shape not equal to 4!";
    return false;
  }
  std::vector<int64_t> dims;
  for (auto d : input.shape) {
    dims.push_back(static_cast<int64_t>(d));
  }

  // use tensor
  framework::DDim ddim =
      framework::make_ddim({dims[0], dims[1], dims[2], dims[3]});

  framework::Tensor input_tensor;
  input_tensor.Resize(ddim);
  int input_length = framework::product(ddim);
  auto input_ptr = input_tensor.mutable_data<T>();

  memcpy(input_ptr, static_cast<T *>(input.data.data()),
         input_length * sizeof(T));
  paddle_mobile_->Predict(input_tensor);
  auto output_tensor = paddle_mobile_->Fetch();

  if (output_data->empty()) {
    LOG(kLOG_ERROR) << "At least one output should be set with tensors' names.";
    return false;
  }

  auto &output = (*output_data)[0];
  int output_length = output_tensor->numel();
  std::vector<int64_t> tensor_shape =
      framework::vectorize(output_tensor->dims());

  for (auto d : tensor_shape) {
    output.shape.push_back(static_cast<int>(d));
  }

  if (output.data.length() < output_length * sizeof(T)) {
    output.data.Resize(output_length * sizeof(T));
  }

  memcpy(output.data.data(), output_tensor->template data<T>(),
         output_length * sizeof(T));

  return true;
}

#ifdef PADDLE_MOBILE_FPGA
template <typename Device, typename T>
bool PaddleMobilePredictor<Device, T>::Run(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data, std::vector<int> *index_data,
    int batch_size) {
  if (inputs.empty()) {
    LOG(kLOG_ERROR) << "At least one output should be set with tensors' names.";
    return false;
  }
  auto input = inputs[0];

  if (input.shape.size() != 4) {
    LOG(kLOG_ERROR) << "input shape not equal to 4!";
    return false;
  }
  std::vector<int64_t> dims;
  for (auto d : input.shape) {
    dims.push_back(static_cast<int64_t>(d));
  }

  // use tensor
  framework::DDim ddim =
      framework::make_ddim({dims[0], dims[1], dims[2], dims[3]});

  framework::Tensor input_tensor;
  input_tensor.Resize(ddim);
  int input_length = framework::product(ddim);
  auto input_ptr = input_tensor.mutable_data<T>();

  memcpy(input_ptr, static_cast<T *>(input.data.data()),
         input_length * sizeof(T));
  paddle_mobile_->Predict(input_tensor);
  auto num_result = index_data->size();
  if (output_data->size() != num_result) {
    LOG(kLOG_ERROR) << "index and output number don't match";
    return false;
  }

  for (int i = 0; i < num_result; i++) {
    auto output_tensor = paddle_mobile_->FetchResult((*index_data)[i]);

    if (output_data->empty()) {
      LOG(kLOG_ERROR)
          << "At least one output should be set with tensors' names.";
      return false;
    }

    auto &output = (*output_data)[i];
    int output_length = output_tensor->numel();
    std::vector<int64_t> tensor_shape =
        framework::vectorize(output_tensor->dims());

    for (auto d : tensor_shape) {
      output.shape.push_back(static_cast<int>(d));
    }

    if (output.data.length() < output_length * sizeof(T)) {
      output.data.Resize(output_length * sizeof(T));
    }

    memcpy(output.data.data(), output_tensor->template data<T>(),
           output_length * sizeof(T));
  }

  return true;
}
template <typename Device, typename T>
void PaddleMobilePredictor<Device, T>::FeedData(
    const std::vector<void *> &inputs) {
  paddle_mobile_->FeedData(inputs);
}

template <typename Device, typename T>
void PaddleMobilePredictor<Device, T>::GetResults(
    std::vector<void *> *outputs) {
  paddle_mobile_->GetResults(outputs);
}

template <typename Device, typename T>
void PaddleMobilePredictor<Device, T>::Predict_From_To(int start, int end) {
  paddle_mobile_->Predict_From_To(start, end);
}

#endif
template <typename Device, typename T>
PaddleMobilePredictor<Device, T>::~PaddleMobilePredictor() {
  paddle_mobile_->Clear();
}

// A factory to help create difference predictor.
template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<PaddleMobileConfig, PaddleEngineKind::kPaddleMobile>(
    const PaddleMobileConfig &config) {
  std::unique_ptr<PaddlePredictor> x;
  if (config.precision == PaddleMobileConfig::FP32) {
    if (config.device == PaddleMobileConfig::kCPU) {
      x.reset(new PaddleMobilePredictor<CPU, float>(config));
    } else if (config.device == PaddleMobileConfig::kFPGA) {
      x.reset(new PaddleMobilePredictor<FPGA, float>(config));
    } else if (config.device == PaddleMobileConfig::kGPU_MALI) {
      x.reset(new PaddleMobilePredictor<GPU_MALI, float>(config));
    } else if (config.device == PaddleMobileConfig::kGPU_CL) {
      x.reset(new PaddleMobilePredictor<GPU_CL, float>(config));
    } else {
      LOG(kLOG_ERROR) << "unsupport device type!";
      return nullptr;
    }
  } else {
    LOG(kLOG_ERROR) << "unsupport precision type!";
    return nullptr;
  }
  return std::move(x);
}

}  // namespace paddle_mobile
