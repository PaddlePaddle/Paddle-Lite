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
#include "lite/core/optimizer/mir/pass_manager.h"
#include "lite/core/optimizer/mir/post_quant_dynamic_pass.h"
#include "lite/core/optimizer/mir/sparse_conv_detect_pass.h"
#include "lite/core/version.h"
#ifdef LITE_USE_THREAD_POOL
#include "lite/core/parallel_defines.h"
#include "lite/core/thread_pool.h"
#endif
#ifndef LITE_ON_TINY_PUBLISH
#include "lite/api/paddle_use_passes.h"
#endif

#if (defined LITE_WITH_X86) && (defined PADDLE_WITH_MKLML) && \
    !(defined LITE_ON_MODEL_OPTIMIZE_TOOL)
#if !defined(__APPLE__)
#include <omp.h>
#endif
#include "lite/backends/x86/mklml.h"
#endif
namespace paddle {
namespace lite {

void CxxPaddleApiImpl::Init(const lite_api::CxxConfig &config) {
  config_ = config;
  mode_ = config.power_mode();
  threads_ = config.threads();
#ifdef LITE_USE_THREAD_POOL
  int thread_num = ThreadPool::Init(threads_);
  if (thread_num > 1) {
    ThreadPool::AcquireThreadPool();
  }
#endif
  if (!status_is_cloned_) {
    auto places = config.valid_places();
    std::vector<std::string> passes = config.get_passes_internal();
#ifdef LITE_WITH_CUDA
    // if kCUDA is included in valid places, it should be initialized first,
    // otherwise skip this step.
    for (auto &p : places) {
      if (p.target == TARGET(kCUDA)) {
        Env<TARGET(kCUDA)>::Init();
        if (config_.multi_stream()) {
          passes.push_back("multi_stream_analysis_pass");
          VLOG(3) << "add pass: " << passes[0];
        }
        break;
      }
    }
#endif
#ifdef LITE_WITH_MLU
    Env<TARGET(kMLU)>::Init();
    lite::TargetWrapperMlu::SetMLURunMode(config.mlu_core_version(),
                                          config.mlu_core_number(),
                                          config.mlu_input_layout(),
                                          config.mlu_firstconv_param());
#endif  // LITE_WITH_MLU

#ifdef LITE_WITH_BM
    Env<TARGET(kBM)>::Init();
    int device_id = 0;
    if (const char *c_id = getenv("BM_VISIBLE_DEVICES")) {
      device_id = static_cast<int>(*c_id) - 48;
    }
    TargetWrapper<TARGET(kBM)>::SetDevice(device_id);
#endif  // LITE_WITH_BM

#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
    // Use scope to store the model-level configuration for the subgraph kernel
    Context<TargetType::kNNAdapter>::SetNNAdapterDeviceNames(
        raw_predictor_->scope(), config.nnadapter_device_names());
    Context<TargetType::kNNAdapter>::SetNNAdapterContextProperties(
        raw_predictor_->scope(), config.nnadapter_context_properties());
    Context<TargetType::kNNAdapter>::SetNNAdapterContextCallback(
        raw_predictor_->scope(), config.nnadapter_context_callback());
    Context<TargetType::kNNAdapter>::SetNNAdapterModelCacheDir(
        raw_predictor_->scope(), config.nnadapter_model_cache_dir());
    Context<TargetType::kNNAdapter>::SetNNAdapterModelCacheBuffers(
        raw_predictor_->scope(), config.nnadapter_model_cache_buffers());
    Context<TargetType::kNNAdapter>::SetNNAdapterSubgraphPartitionConfigPath(
        raw_predictor_->scope(),
        config.nnadapter_subgraph_partition_config_path());
    Context<TargetType::kNNAdapter>::SetNNAdapterSubgraphPartitionConfigBuffer(
        raw_predictor_->scope(),
        config.nnadapter_subgraph_partition_config_buffer());
    Context<TargetType::kNNAdapter>::
        SetNNAdapterMixedPrecisionQuantizationConfigPath(
            raw_predictor_->scope(),
            config.nnadapter_mixed_precision_quantization_config_path());
    Context<TargetType::kNNAdapter>::
        SetNNAdapterMixedPrecisionQuantizationConfigBuffer(
            raw_predictor_->scope(),
            config.nnadapter_mixed_precision_quantization_config_buffer());
    Context<TargetType::kNNAdapter>::SetNNAdapterDynamicShapeInfo(
        raw_predictor_->scope(), config.nnadapter_dynamic_shape_info());
#endif

    auto use_layout_preprocess_pass =
        config.model_dir().find("OPENCL_PRE_PRECESS");
    VLOG(1) << "use_layout_preprocess_pass:" << use_layout_preprocess_pass;
    if (places[0].target == TARGET(kOpenCL) &&
        use_layout_preprocess_pass != std::string::npos) {
      passes.push_back("type_layout_cast_preprocess_pass");
      VLOG(1) << "add pass:" << passes[0];
    }

    if (config.quant_model()) {
      passes.push_back("post_quant_dynamic_pass");
      auto *pass = mir::PassManager::Global().LookUp<mir::PostQuantDynamicPass>(
          "post_quant_dynamic_pass");
      CHECK(pass);
      pass->SetQuantType(config.quant_type());
    }

    auto *sparse_detect_pass =
        mir::PassManager::Global().LookUp<mir::SparseConvDetectPass>(
            "sparse_conv_detect_pass");
    CHECK(sparse_detect_pass);
    if (config.sparse_model()) {
      sparse_detect_pass->SetSparseThreshold(config.sparse_threshold());
    } else {
      // Pass in a value greater than 1.0 to turn off the sparse pass
      // internally.
      sparse_detect_pass->SetSparseThreshold(1.5);
    }

    raw_predictor_->Build(config, places, passes);
  } else {
    raw_predictor_->PrepareFeedFetch();
    CHECK(raw_predictor_) << "The Predictor can not be nullptr in Clone mode.";
  }

#ifdef LITE_WITH_METAL
  raw_predictor_->ConfigMetalContext(config);
#endif

#ifdef LITE_WITH_NPU
  // Store the model-level configuration into scope for kernels, and use
  // exe_scope to store the execution-level configuration
  Context<TargetType::kNPU>::SetSubgraphModelCacheDir(
      raw_predictor_->scope(), config.subgraph_model_cache_dir());
#endif

#if (defined LITE_WITH_X86) && (defined PADDLE_WITH_MKLML) && \
    !(defined LITE_ON_MODEL_OPTIMIZE_TOOL)
  int num_threads = config.x86_math_num_threads();
  int real_num_threads = num_threads > 1 ? num_threads : 1;
#ifdef LITE_WITH_STATIC_MKL
  MKL_Set_Num_Threads(real_num_threads);
#else
  x86::MKL_Set_Num_Threads(real_num_threads);
#endif
#if !defined(__APPLE__)
  omp_set_num_threads(real_num_threads);
#endif
  VLOG(3) << "x86_math_num_threads() is set successfully and the "
             "number of threads is:"
          << real_num_threads;
#endif

#ifdef LITE_WITH_XPU
  auto preferred_inputs = config.preferred_inputs_for_warmup();
  for (auto &preferred_input : preferred_inputs) {
    auto &input_tensors = preferred_input.second;
    if (input_tensors.empty()) continue;
    for (size_t i = 0; i < input_tensors.size(); i++) {
      auto input_tensor = static_cast<lite::Tensor *>(input_tensors[i].get());
      auto shape = input_tensor->dims().Vectorize();
      CHECK(!shape.empty())
          << "tensor is not set, with group_id: " << preferred_input.first
          << ", tensor_id: " << i;

      auto in_tensor = GetInput(i);
      in_tensor->Resize(shape);
      in_tensor->SetLoD(input_tensor->lod());
      int64_t size = std::accumulate(
          shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
      switch (input_tensor->precision()) {
        case lite_api::PrecisionType::kFloat:
          memcpy(in_tensor->mutable_data<float>(),
                 input_tensor->data<float>(),
                 sizeof(float) * size);
          break;
        case lite_api::PrecisionType::kFP64:
          memcpy(in_tensor->mutable_data<double>(),
                 input_tensor->data<double>(),
                 sizeof(double) * size);
          break;
        case lite_api::PrecisionType::kInt32:
          memcpy(in_tensor->mutable_data<int32_t>(),
                 input_tensor->data<int32_t>(),
                 sizeof(int32_t) * size);
          break;
        case lite_api::PrecisionType::kInt64:
          memcpy(in_tensor->mutable_data<int64_t>(),
                 input_tensor->data<int64_t>(),
                 sizeof(int64_t) * size);
          break;
        default:
          LOG(FATAL) << "unsupport data type: "
                     << lite_api::PrecisionToStr(input_tensor->precision());
      }
    }
    Run();
  }
#endif
}

CxxPaddleApiImpl::~CxxPaddleApiImpl() {
#ifdef LITE_USE_THREAD_POOL
  ThreadPool::ReleaseThreadPool();
#endif
}

std::unique_ptr<lite_api::Tensor> CxxPaddleApiImpl::GetInputByName(
    const std::string &name) {
  auto *x = raw_predictor_->GetInputByName(name);
  return std::unique_ptr<lite_api::Tensor>(new lite_api::Tensor(x));
}

std::unique_ptr<const lite_api::Tensor> CxxPaddleApiImpl::GetOutputByName(
    const std::string &name) const {
  const auto *x = raw_predictor_->GetOutputByName(name);
  return std::unique_ptr<lite_api::Tensor>(new lite_api::Tensor(x));
}

std::unique_ptr<lite_api::Tensor> CxxPaddleApiImpl::GetInput(int i) {
  auto *x = raw_predictor_->GetInput(i);
  return std::unique_ptr<lite_api::Tensor>(new lite_api::Tensor(x));
}

std::unique_ptr<const lite_api::Tensor> CxxPaddleApiImpl::GetOutput(
    int i) const {
  const auto *x = raw_predictor_->GetOutput(i);
  return std::unique_ptr<lite_api::Tensor>(new lite_api::Tensor(x));
}

std::vector<std::string> CxxPaddleApiImpl::GetInputNames() {
  return raw_predictor_->GetInputNames();
}

std::vector<std::string> CxxPaddleApiImpl::GetParamNames() {
  return raw_predictor_->GetParamNames();
}

std::vector<std::string> CxxPaddleApiImpl::GetOutputNames() {
  return raw_predictor_->GetOutputNames();
}

void CxxPaddleApiImpl::Run() {
#ifdef LITE_WITH_ARM
  lite::DeviceInfo::Global().SetRunMode(mode_, threads_);
#endif
  raw_predictor_->Run();
}

std::shared_ptr<lite_api::PaddlePredictor> CxxPaddleApiImpl::Clone() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto predictor =
      std::make_shared<lite::CxxPaddleApiImpl>(raw_predictor_->Clone());
  predictor->Init(config_);
  return predictor;
}

std::shared_ptr<lite_api::PaddlePredictor> CxxPaddleApiImpl::Clone(
    const std::vector<std::string> &var_names) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto predictor = std::make_shared<lite::CxxPaddleApiImpl>(
      raw_predictor_->Clone(var_names));
  predictor->Init(config_);
  return predictor;
}

std::string CxxPaddleApiImpl::GetVersion() const { return version(); }

std::unique_ptr<const lite_api::Tensor> CxxPaddleApiImpl::GetTensor(
    const std::string &name) const {
  auto *x = raw_predictor_->GetTensor(name);
  return std::unique_ptr<const lite_api::Tensor>(new lite_api::Tensor(x));
}

std::unique_ptr<lite_api::Tensor> CxxPaddleApiImpl::GetMutableTensor(
    const std::string &name) {
  return std::unique_ptr<lite_api::Tensor>(
      new lite_api::Tensor(raw_predictor_->GetMutableTensor(name)));
}

void CxxPaddleApiImpl::SaveOptimizedModel(const std::string &model_dir,
                                          lite_api::LiteModelType model_type,
                                          bool record_info) {
  raw_predictor_->SaveModel(model_dir, model_type, record_info);
}

bool CxxPaddleApiImpl::TryShrinkMemory() {
  return raw_predictor_->TryShrinkMemory();
}

}  // namespace lite

namespace lite_api {

template <>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(
    const CxxConfig &config) {
  static std::mutex mutex_conf;
  std::unique_lock<std::mutex> lck(mutex_conf);
  auto x = std::make_shared<lite::CxxPaddleApiImpl>();
  x->Init(config);
  return x;
}

}  // namespace lite_api
}  // namespace paddle
