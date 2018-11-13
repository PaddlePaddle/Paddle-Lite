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
#include "framework/tensor.h"

namespace paddle_mobile {

template <typename Dtype, Precision P>
PaddleMobilePredictor<Dtype, P>::PaddleMobilePredictor(
    const PaddleMobileConfig &config) {
  PADDLE_MOBILE_ENFORCE(Init(config) == true,
                        "paddle mobile predictor init failed!");
  config_ = config;
}

template <typename Dtype, Precision P>
bool PaddleMobilePredictor<Dtype, P>::Init(const PaddleMobileConfig &config) {
  paddle_mobile_.reset(new PaddleMobile<Dtype, P>());
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
template <typename Dtype, Precision P>
bool PaddleMobilePredictor<Dtype, P>::Run(
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
  typedef typename PrecisionTrait<P>::ptype PType;
  auto input_ptr = input_tensor.mutable_data<PType>();

  memcpy(input_ptr, static_cast<PType *>(input.data.data()),
         input_length * sizeof(PType));
  auto output_tensor = paddle_mobile_->Predict(input_tensor);

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

  if (output.data.length() < output_length * sizeof(PType)) {
    output.data.Resize(output_length * sizeof(PType));
  }

  memcpy(output.data.data(), output_tensor->template data<PType>(),
         output_length * sizeof(PType));

  return true;
}

template <typename Dtype, Precision P>
PaddleMobilePredictor<Dtype, P>::~PaddleMobilePredictor() {
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
      x.reset(new PaddleMobilePredictor<CPU, Precision::FP32>(config));
    } else if (config.device == PaddleMobileConfig::kFPGA) {
      x.reset(new PaddleMobilePredictor<FPGA, Precision::FP32>(config));
    } else if (config.device == PaddleMobileConfig::kGPU_MALI) {
      x.reset(new PaddleMobilePredictor<GPU_MALI, Precision::FP32>(config));
    } else if (config.device == PaddleMobileConfig::kGPU_CL) {
      x.reset(new PaddleMobilePredictor<GPU_CL, Precision::FP32>(config));
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
