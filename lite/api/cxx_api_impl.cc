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
#include <string>
#include "lite/api/paddle_api.h"
#include "lite/core/device_info.h"
#include "lite/core/version.h"

namespace paddle {
namespace lite {

void CxxPaddleApiImpl::Init(const lite_api::CxxConfig &config) {
#ifdef LITE_WITH_CUDA
  Env<TARGET(kCUDA)>::Init();
#endif
  auto places = config.valid_places();
  raw_predictor_.Build(config, places);
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

void CxxPaddleApiImpl::Run() { raw_predictor_.Run(); }

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
