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

#include "lite/api/cxx_api.h"
#include <memory>
#include <mutex>  //NOLINT
#include <string>
#include "lite/api/paddle_api.h"
#include "lite/core/device_info.h"
#include "lite/core/version.h"

#if (defined LITE_WITH_X86) && (defined PADDLE_WITH_MKLML) && \
    !(defined LITE_ON_MODEL_OPTIMIZE_TOOL)
#include <omp.h>
#include "lite/backends/x86/mklml.h"
#endif

namespace paddle {
namespace lite {

void CxxPaddleApiImpl::Init(const lite_api::CxxConfig &config) {
  config_ = config;
#ifdef LITE_WITH_CUDA
  Env<TARGET(kCUDA)>::Init();
#endif
#ifdef LITE_WITH_MLU
  Env<TARGET(kMLU)>::Init();
  mlu_core_version_ = config.mlu_core_version();
  mlu_core_number_ = config.mlu_core_number();
#endif  // LITE_WITH_MLU
  auto places = config.valid_places();
  std::vector<std::string> passes{};
  auto use_layout_preprocess_pass =
      config.model_dir().find("OPENCL_PRE_PRECESS");
  VLOG(1) << "use_layout_preprocess_pass:" << use_layout_preprocess_pass;
  if (places[0].target == TARGET(kOpenCL) &&
      use_layout_preprocess_pass != std::string::npos) {
    passes = {"type_layout_cast_preprocess_pass"};
    VLOG(1) << "add pass:" << passes[0];
  }
  raw_predictor_.Build(config, places, passes);
  mode_ = config.power_mode();
  threads_ = config.threads();

#if (defined LITE_WITH_X86) && (defined PADDLE_WITH_MKLML) && \
    !(defined LITE_ON_MODEL_OPTIMIZE_TOOL)
  int num_threads = config.x86_math_library_num_threads();
  int real_num_threads = num_threads > 1 ? num_threads : 1;
  paddle::lite::x86::MKL_Set_Num_Threads(real_num_threads);
  omp_set_num_threads(real_num_threads);
  VLOG(3) << "set_x86_math_library_math_threads() is set successfully and the "
             "number of threads is:"
          << num_threads;
#endif
}

std::unique_ptr<lite_api::Tensor> CxxPaddleApiImpl::GetInput(int i) {
  auto *x = raw_predictor_.GetInput(i);
  return std::unique_ptr<lite_api::Tensor>(new lite_api::Tensor(x));
}

std::unique_ptr<const lite_api::Tensor> CxxPaddleApiImpl::GetOutput(
    int i) const {
  const auto *x = raw_predictor_.GetOutput(i);
  return std::unique_ptr<lite_api::Tensor>(new lite_api::Tensor(x));
}

std::vector<std::string> CxxPaddleApiImpl::GetInputNames() {
  return raw_predictor_.GetInputNames();
}

std::vector<std::string> CxxPaddleApiImpl::GetOutputNames() {
  return raw_predictor_.GetOutputNames();
}

void CxxPaddleApiImpl::Run() {
#ifdef LITE_WITH_ARM
  lite::DeviceInfo::Global().SetRunMode(mode_, threads_);
#endif
#ifdef LITE_WITH_MLU
  lite::DeviceInfo::Global().SetMLURunMode(mlu_core_version_, mlu_core_number_);
#endif
  raw_predictor_.Run();
}

std::shared_ptr<lite_api::PaddlePredictor> CxxPaddleApiImpl::Clone() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto predictor = std::make_shared<lite::CxxPaddleApiImpl>();
  predictor->Init(config_);
  return predictor;
}

std::string CxxPaddleApiImpl::GetVersion() const { return version(); }

std::unique_ptr<const lite_api::Tensor> CxxPaddleApiImpl::GetTensor(
    const std::string &name) const {
  auto *x = raw_predictor_.GetTensor(name);
  return std::unique_ptr<const lite_api::Tensor>(new lite_api::Tensor(x));
}

std::unique_ptr<lite_api::Tensor> CxxPaddleApiImpl::GetInputByName(
    const std::string &name) {
  return std::unique_ptr<lite_api::Tensor>(
      new lite_api::Tensor(raw_predictor_.GetInputByName(name)));
}

void CxxPaddleApiImpl::SaveOptimizedModel(const std::string &model_dir,
                                          lite_api::LiteModelType model_type,
                                          bool record_info) {
  raw_predictor_.SaveModel(model_dir, model_type, record_info);
}

}  // namespace lite

namespace lite_api {

template <>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(
    const CxxConfig &config) {
  auto x = std::make_shared<lite::CxxPaddleApiImpl>();
  x->Init(config);
  return x;
}

}  // namespace lite_api
}  // namespace paddle
