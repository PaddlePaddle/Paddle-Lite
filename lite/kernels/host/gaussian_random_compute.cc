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

#include "lite/kernels/host/gaussian_random_compute.h"
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

uint64_t GetRandomSeed() {
  std::random_device rd;
  // double has 53 bit significant, so limit uint64 to 53 bits
  return ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
}

std::shared_ptr<std::mt19937_64> GetCPURandomEngine(uint64_t seed) {
  auto engine = std::make_shared<std::mt19937_64>();
  if (seed == 0) {
    seed = GetRandomSeed();
    VLOG(4) << "Use default random engine with random seed = " << seed;
  } else {
    VLOG(4) << "Use default random engine with fixed random seed = " << seed;
  }
  engine->seed(seed);
  return engine;
}

void GaussRandomCompute::Run() {
  auto& param = Param<operators::GaussRandomParam>();
  float mean = param.mean;
  float gstd = param.gauss_std;

  // output shape
  if (param.ShapeTensor != nullptr) {
    std::vector<int64_t> tmp{};
    auto ptr = param.ShapeTensor->data<int>();
    for (int i = 0; i < param.ShapeTensor->numel(); i++) {
      int64_t shap_tensor = static_cast<int64_t>(ptr[i]);
      tmp.push_back(shap_tensor);
    }
    DDimLite dims(tmp);
    param.Out->Resize(dims);
  } else if (param.ShapeTensorList.size() > 0) {
    std::vector<int64_t> tmp{};
    for (size_t i = 0; i < param.ShapeTensorList.size(); ++i) {
      auto tmp_tensor_ptr = param.ShapeTensorList[i]->data<int>();
      tmp.push_back(static_cast<int64_t>(tmp_tensor_ptr[0]));
    }
    DDimLite dims(tmp);
    param.Out->Resize(dims);
  } else {
    DDimLite dims(param.shape);
    param.Out->Resize(dims);
  }
  auto data = param.Out->mutable_data<float>();
  int size = param.Out->numel();
  std::normal_distribution<float> dist(mean, gstd);
  unsigned int useed = static_cast<unsigned int>(param.seed);
  auto engine = GetCPURandomEngine(useed);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gaussian_random,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::GaussRandomCompute,
                     def)
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("ShapeTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
