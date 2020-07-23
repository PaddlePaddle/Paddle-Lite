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

#pragma once
#include <map>
#include <memory>
#include <mutex>  //NOLINT
#include <string>
#include <utility>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/core/device_info.h"
#include "lite/core/op_lite.h"
#include "lite/core/optimizer.h"
#include "lite/core/program.h"
#include "lite/core/types.h"
#include "lite/model_parser/model_parser.h"

#ifdef LITE_WITH_CUDA
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/stream_guard.h"
#endif

namespace paddle {
namespace lite {

static const char TAILORD_OPS_SOURCE_LIST_FILENAME[] =
    ".tailored_ops_source_list";
static const char TAILORD_OPS_LIST_NAME[] = ".tailored_ops_list";
static const char TAILORD_KERNELS_SOURCE_LIST_FILENAME[] =
    ".tailored_kernels_source_list";
static const char TAILORD_KERNELS_LIST_NAME[] = ".tailored_kernels_list";

std::vector<std::string> GetAllOps();

/*
 * Predictor for inference, input a model, it will optimize and execute it.
 */
class LITE_API Predictor {
 public:
  // Create an empty predictor.
  Predictor() {
    scope_ = std::make_shared<Scope>();
    program_desc_ = std::make_shared<cpp::ProgramDesc>();
  }

  // Create a predictor with the weight variable scope set.
  explicit Predictor(const std::shared_ptr<lite::Scope>& root_scope)
      : scope_(root_scope) {}
  Predictor(const std::shared_ptr<cpp::ProgramDesc>& program_desc,
            const std::shared_ptr<Scope>& root_scope,
            const std::vector<Place>& valid_places,
            const std::vector<std::string>& vars_to_clone = {})
      : program_desc_(program_desc), scope_(root_scope) {
    Program program(program_desc_, scope_, valid_places, vars_to_clone);
    optimizer_ = Optimizer(std::move(program), valid_places);
    exec_scope_ = optimizer_.exec_scope();
    valid_places_ = valid_places;
  }

  // Build from a model, with places set for hardware config.
  void Build(
      const lite_api::CxxConfig& config,
      const std::vector<Place>& valid_places,
      const std::vector<std::string>& passes = {},
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf);

  void Build(
      const std::string& model_path,
      const std::string& model_file_path,
      const std::string& param_file_path,
      const std::vector<Place>& valid_places,
      const std::vector<std::string>& passes = {},
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf,
      bool memory_from_memory = false);

  void Build(const std::shared_ptr<cpp::ProgramDesc>& program_desc,
             const std::vector<Place>& valid_places,
             const std::vector<std::string>& passes = {});

  std::shared_ptr<Predictor> Clone() const {
    return std::make_shared<Predictor>(program_desc_, scope_, valid_places_);
  }

  std::shared_ptr<Predictor> Clone(
      const std::vector<std::string>& vars_to_clone) const {
    CHECK(program_desc_) << "Both program and scope of current predicotr "
                            "should be not be nullptr in Clone mode.";
    CHECK(scope_) << "Both program and scope of current predicotr should be "
                     "not be nullptr in Clone mode.";
    auto predictor = std::make_shared<Predictor>(
        program_desc_, scope_, valid_places_, vars_to_clone);

    for (auto var_name : vars_to_clone) {
      predictor->exec_scope_->LocalVar(var_name);
      auto* tensor = predictor->scope_->Var(var_name)->GetMutable<Tensor>();
      auto* sub_tensor =
          predictor->exec_scope_->Var(var_name)->GetMutable<Tensor>();
      sub_tensor->CopyDataFrom(*tensor);
    }
    return predictor;
  }

  void GenRuntimeProgram();

  // Run the predictor for a single batch of data.
  void Run() {
    if (!program_generated_) {
      GenRuntimeProgram();
    }
    program_->Run();
  }

  // Get offset-th col of feed inputs.
  lite::Tensor* GetInput(size_t offset);
  // get input by name.
  lite::Tensor* GetInputByName(const std::string& name);
  // get inputnames and get outputnames.
  std::vector<std::string> GetInputNames();
  std::vector<std::string> GetOutputNames();
  // get param names
  std::vector<std::string> GetParamNames();

  void PrepareFeedFetch();

  // Get offset-th col of fetch results.
  const lite::Tensor* GetOutput(size_t offset) const;
  std::vector<const lite::Tensor*> GetOutputs() const;

  const cpp::ProgramDesc& program_desc() const;
  // get a mutable tensor according to its name
  lite::Tensor* GetMutableTensor(const std::string& name);
  // get a const tensor according to its name
  const lite::Tensor* GetTensor(const std::string& name) const;
  const RuntimeProgram& runtime_program() const;
  Scope* scope() { return scope_.get(); }

  // This method is disabled in mobile, for unnecessary dependencies required.
  void SaveModel(
      const std::string& dir,
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf,
      bool record_info = false);
  void SaveOpKernelInfo(const std::string& model_dir);

#ifdef LITE_WITH_CUDA
  void set_cuda_use_multi_stream(bool multi_stream) {
    cuda_use_multi_stream_ = multi_stream;
  }
  bool cuda_use_multi_stream() { return cuda_use_multi_stream_; }
  void set_cuda_exec_stream(cudaStream_t stream) { cuda_exec_stream_ = stream; }
  void set_cuda_io_stream(cudaStream_t stream) { cuda_io_stream_ = stream; }
  cudaStream_t cuda_exec_stream() { return cuda_exec_stream_; }
  cudaStream_t cuda_io_stream() { return cuda_io_stream_; }
#endif

 private:
  Optimizer optimizer_;
  std::shared_ptr<cpp::ProgramDesc> program_desc_;
  std::shared_ptr<Scope> scope_;
  Scope* exec_scope_;
  std::unique_ptr<RuntimeProgram> program_;
  bool program_generated_{false};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<Place> valid_places_;

#ifdef LITE_WITH_CUDA
  bool cuda_use_multi_stream_{false};
  cudaStream_t cuda_io_stream_;
  cudaStream_t cuda_exec_stream_;
#endif
};

class CxxPaddleApiImpl : public lite_api::PaddlePredictor {
 public:
  CxxPaddleApiImpl() {
    raw_predictor_ = std::make_shared<Predictor>();
    status_is_cloned_ = false;
  }
  explicit CxxPaddleApiImpl(const std::shared_ptr<Predictor>& raw_predictor)
      : raw_predictor_(raw_predictor) {
    status_is_cloned_ = true;
  }

  ~CxxPaddleApiImpl();

  /// Create a new predictor from a config.
  void Init(const lite_api::CxxConfig& config);

  std::unique_ptr<lite_api::Tensor> GetInput(int i) override;

  std::unique_ptr<const lite_api::Tensor> GetOutput(int i) const override;

  void Run() override;

  std::shared_ptr<lite_api::PaddlePredictor> Clone() override;

  std::shared_ptr<lite_api::PaddlePredictor> Clone(
      const std::vector<std::string>& var_names) override;

  std::string GetVersion() const override;

  // get inputs names and get outputs names
  std::vector<std::string> GetInputNames() override;
  std::vector<std::string> GetOutputNames() override;
  // get param names
  std::vector<std::string> GetParamNames() override;

  // get tensor according to tensor's name
  std::unique_ptr<const lite_api::Tensor> GetTensor(
      const std::string& name) const override;
  // get a mutable tensor according to tensor's name
  std::unique_ptr<lite_api::Tensor> GetMutableTensor(
      const std::string& name) override;

  // Get InputTebsor by name
  std::unique_ptr<lite_api::Tensor> GetInputByName(
      const std::string& name) override;

  void SaveOptimizedModel(
      const std::string& model_dir,
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf,
      bool record_info = false) override;

 private:
#ifdef LITE_WITH_CUDA
  // Cuda related environment initialization, including setting stream pointers,
  // initializing synchronization events, setting predictor_id, etc.
  void InitCudaEnv(std::vector<std::string>* passes);
  // Due to the asynchronous nature of cuda kernel execution, synchronization is
  // required before setting input and getting output.
  void SyncCudaInputs();
  void SyncCudaOutputs();
#endif

 private:
  std::shared_ptr<Predictor> raw_predictor_;
  lite_api::CxxConfig config_;
  std::mutex mutex_;
  bool status_is_cloned_;

#ifdef LITE_WITH_CUDA
  bool cuda_use_multi_stream_{false};
  std::unique_ptr<lite::CudaStreamGuard> cuda_io_stream_;
  std::unique_ptr<lite::CudaStreamGuard> cuda_exec_stream_;
  cudaEvent_t cuda_input_event_;
  std::vector<cudaEvent_t> cuda_output_events_;
  // only used for multi exec stream mode.
  std::vector<lite::CudaStreamGuard> cuda_exec_streams_;
#endif
};

}  // namespace lite
}  // namespace paddle
