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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/enforce.h"
#include "framework/tensor.h"
#ifdef PADDLE_MOBILE_FPGA
#include <fpga/common/fpga_common.h>
#endif

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
    paddle_mobile_->LoadCombinedMemory(
        config.memory_pack.model_size, config.memory_pack.model_buf,
        config.memory_pack.combined_params_size,
        config.memory_pack.combined_params_buf, config.optimize,
        config.quantification, config.batch_size, config.lod_mode);
  } else if (!config.model_dir.empty()) {
    paddle_mobile_->Load(config.model_dir, config.optimize,
                         config.quantification, config.batch_size,
                         config.lod_mode);
  } else if (!config.prog_file.empty() && !config.param_file.empty()) {
    paddle_mobile_->Load(config.prog_file, config.param_file, config.optimize,
                         config.quantification, config.batch_size,
                         config.lod_mode);
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

  if (input.lod.size() == 0 && input.shape.size() != 4) {
    LOG(kLOG_ERROR) << "input shape not equal to 4!";
    return false;
  }
  std::vector<int64_t> dims;
  for (auto d : input.shape) {
    dims.push_back(static_cast<int64_t>(d));
  }

  // use tensor
  framework::DDim ddim = framework::make_ddim(dims);

  framework::Tensor input_tensor;
  framework::LoDTensor input_lod_tensor;
  paddle_mobile::framework::LoD lod{{}};
  for (int i = 0; i < input.lod.size(); ++i) {
    lod[0].push_back(input.lod[i]);
  }
  input_lod_tensor.set_lod(lod);

  int input_length = framework::product(ddim);
  if (input.lod.size() > 0) {
    input_lod_tensor.Resize(ddim);
    memcpy(input_lod_tensor.mutable_data<T>(),
           static_cast<T *>(input.data.data()), input_length * sizeof(T));
    paddle_mobile_->Predict(input_lod_tensor);
  } else {
    input_tensor.Resize(ddim);
    memcpy(input_tensor.mutable_data<T>(), static_cast<T *>(input.data.data()),
           input_length * sizeof(T));
    paddle_mobile_->Predict(input_tensor);
  }

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
void ConvertPaddleTensors(const PaddleTensor &src, framework::Tensor *des) {
  des->Resize(framework::make_ddim(src.shape));
  des->external_data = src.data.data();
  des->set_type(src.dtypeid);
  des->layout =
      src.layout == LAYOUT_HWC ? framework::LAYOUT_HWC : framework::LAYOUT_CHW;
}

void ConvertTensors(const framework::Tensor &src, PaddleTensor *des) {
  des->shape = framework::vectorize2int(src.dims());
  des->dtypeid = src.type();
  des->layout = src.layout == framework::LAYOUT_HWC ? LAYOUT_HWC : LAYOUT_CHW;

  auto num = src.numel();
  if (src.type() == type_id<float>()) {
    des->data.Reset(const_cast<float *>(src.data<float>()),
                    num * sizeof(float));
  } else if (src.type() == type_id<half>()) {
    des->data.Reset(const_cast<int16_t *>(src.data<int16_t>()),
                    num * sizeof(int16_t));
  } else {
    des->data.Reset(const_cast<int8_t *>(src.data<int8_t>()),
                    num * sizeof(int8_t));
  }
}

template <typename Device, typename T>
void PaddleMobilePredictor<Device, T>::FeedPaddleTensors(
    const std::vector<PaddleTensor> &inputs) {
  auto num = inputs.size();
  std::vector<framework::Tensor> tensors(num, framework::Tensor());
  for (int i = 0; i < num; i++) {
    if (inputs[i].dtypeid == type_id<int8_t>().hash_code()) {
      tensors[i].init(type_id<int8_t>().hash_code());
    } else {
      tensors[i].init(type_id<float>().hash_code());
    }
    ConvertPaddleTensors(inputs[i], &tensors[i]);
  }
  paddle_mobile_->FeedTensorData(tensors);
}

template <typename Device, typename T>
void PaddleMobilePredictor<Device, T>::FetchPaddleTensors(
    std::vector<PaddleTensor> *outputs) {
  //  auto num = outputs->size();
  //  PADDLE_MOBILE_ENFORCE(num > 0, "0 output pointers is not permitted");
  //  std::vector<framework::Tensor *> tensors(num, nullptr);
  outputs->clear();
  std::vector<framework::Tensor *> tensors;
  paddle_mobile_->GetTensorResults(&tensors);
  auto num = tensors.size();
  outputs->resize(num, PaddleTensor());
  for (int i = 0; i < num; i++) {
    ConvertTensors(*tensors[i], &(*outputs)[i]);
  }
}

template <typename Device, typename T>
void PaddleMobilePredictor<Device, T>::FetchPaddleTensors(PaddleTensor *output,
                                                          int id) {
  std::shared_ptr<framework::Tensor> tensor_ptr =
      paddle_mobile_->FetchResult(id);
  void *data_addr = nullptr;
  int data_sizeof = 1;
  if (tensor_ptr.get()->type() == type_id<half>().hash_code()) {
    data_addr = tensor_ptr.get()->data<half>();
    data_sizeof = sizeof(half);
  } else if (tensor_ptr.get()->type() == type_id<float>().hash_code()) {
    data_addr = tensor_ptr.get()->data<float>();
    data_sizeof = sizeof(float);
  } else if (tensor_ptr.get()->type() == type_id<int8_t>().hash_code()) {
    data_addr = tensor_ptr.get()->data<int8_t>();
    data_sizeof = sizeof(int8_t);
  } else {
    PADDLE_MOBILE_ENFORCE(0, "output typeid is not supported");
  }
  size_t size = tensor_ptr.get()->numel() * data_sizeof;
  fpga::fpga_invalidate(data_addr, size);
  ConvertTensors(*(tensor_ptr.get()), output);
  return;
}
template <typename Device, typename T>
void PaddleMobilePredictor<Device, T>::GetPaddleTensor(const std::string &name,
                                                       PaddleTensor *output) {
  framework::Tensor *t = paddle_mobile_->GetTensorByName(name);
  ConvertTensors(*t, output);
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
