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

/*
 * This file defines PaddlePredictor, the api for lite. It supports multiple
 * hardware including ARM, X86, OpenCL, CUDA and so on.
 */

#ifndef PADDLE_LITE_API_H_  // NOLINT
#define PADDLE_LITE_API_H_
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle_place.h"  // NOLINT

namespace paddle {
namespace lite_api {

using shape_t = std::vector<int64_t>;
using lod_t = std::vector<std::vector<uint64_t>>;

enum class LiteModelType { kProtobuf = 0, kNaiveBuffer, UNK };
// Methods for allocating L3Cache on Arm platform
enum class L3CacheSetMethod {
  kDeviceL3Cache = 0,  // Use the system L3 Cache size, best performance.
  kDeviceL2Cache = 1,  // Use the system L2 Cache size, trade off performance
                       // with less memory consumption.
  kAbsolute = 2,       // Use the external setting.
  // kAutoGrow = 3,   // Not supported yet, least memory consumption.
};

// return true if current device supports OpenCL model
LITE_API bool IsOpenCLBackendValid(bool check_fp16_valid = false);

struct LITE_API Tensor {
  explicit Tensor(void* raw);
  explicit Tensor(const void* raw);
  void Resize(const shape_t& shape);
  /// Readonly data.
  template <typename T>
  const T* data() const;

  template <typename T>
  T* mutable_data(TargetType type = TargetType::kHost) const;

  // Share external memory. Note: ensure that the data pointer is in a valid
  // state
  // during the prediction process.
  void ShareExternalMemory(void* data, size_t memory_size, TargetType target);

  template <typename T, TargetType type = TargetType::kHost>
  void CopyFromCpu(const T* data);

  template <typename T>
  void CopyToCpu(T* data) const;
  /// Shape of the tensor.
  shape_t shape() const;
  TargetType target() const;
  PrecisionType precision() const;
  void SetPrecision(PrecisionType precision);

  // LoD of the tensor
  lod_t lod() const;

  // Set LoD of the tensor
  void SetLoD(const lod_t& lod);
  bool IsInitialized() const;

 private:
  void* raw_tensor_;
};

/// The PaddlePredictor defines the basic interfaces for different kinds of
/// predictors.
class LITE_API PaddlePredictor {
 public:
  PaddlePredictor() = default;

  /// Get i-th input.
  virtual std::unique_ptr<Tensor> GetInput(int i) = 0;

  /// Get i-th output.
  virtual std::unique_ptr<const Tensor> GetOutput(int i) const = 0;

  virtual void Run() = 0;
  virtual std::shared_ptr<PaddlePredictor> Clone() = 0;
  virtual std::shared_ptr<PaddlePredictor> Clone(
      const std::vector<std::string>& var_names) = 0;

  virtual std::string GetVersion() const = 0;

  // Get input names
  virtual std::vector<std::string> GetInputNames() = 0;
  // Get output names
  virtual std::vector<std::string> GetOutputNames() = 0;
  // Get output names
  virtual std::vector<std::string> GetParamNames();

  // Get Input by name
  virtual std::unique_ptr<Tensor> GetInputByName(const std::string& name) = 0;

  /// Get a readonly tensor, return null if no one called `name` exists.
  virtual std::unique_ptr<const Tensor> GetTensor(
      const std::string& name) const = 0;
  /// Get a mutable tensor, return null if on one called `name` exists
  /// internal infereces API, not recommanded.
  virtual std::unique_ptr<Tensor> GetMutableTensor(const std::string& name);

  /// Persist the optimized model to disk. This API is only supported by
  /// CxxConfig, and the persisted model can be reused for MobileConfig.
  virtual void SaveOptimizedModel(
      const std::string& model_dir,
      LiteModelType model_type = LiteModelType::kProtobuf,
      bool record_info = false);

  virtual ~PaddlePredictor() = default;

 protected:
  int threads_{1};
  lite_api::PowerMode mode_{lite_api::LITE_POWER_NO_BIND};
};

class LITE_API CxxModelBuffer {
 public:
  CxxModelBuffer(const char* program_buffer,
                 size_t program_buffer_size,
                 const char* params_buffer,
                 size_t params_buffer_size);
  CxxModelBuffer(std::string&& program_buffer, std::string&& params_buffer);
  const std::string& get_program() const;
  const std::string& get_params() const;
  bool is_empty() const;

  CxxModelBuffer() = default;
  CxxModelBuffer(const CxxModelBuffer&) = delete;

 private:
  std::string program_;
  std::string params_;
};

/// CxxConfig is the config for the Full feature predictor.
class LITE_API CxxConfig {
 public:
  explicit CxxConfig(PowerMode mode = LITE_POWER_NO_BIND, int threads = 1);
  ///////////////////////////////////////////////////////////////////////////////////////
  // Basics
  // set Thread
  void set_threads(int threads);
  int threads() const;
  // set Power_mode
  void set_power_mode(PowerMode mode);
  PowerMode power_mode() const;
  // set GPU opencl tune
  void set_opencl_tune(CLTuneMode tune_mode = CL_TUNE_NONE,
                       size_t lws_repeats = 4);
  // set GPU opencl precision
  void set_opencl_precision(CLPrecisionType p = CL_PRECISION_AUTO);
  // set subgraph_model_dir
  void set_subgraph_model_cache_dir(std::string subgraph_model_cache_dir);
  const std::string& subgraph_model_cache_dir() const;
  void set_subgraph_model_cache_buffers(const std::string& key,
                                        const std::vector<char>& cfg,
                                        const std::vector<char>& bin);
  const std::map<std::string, std::pair<std::vector<char>, std::vector<char>>>&
  subgraph_model_cache_buffers() const;
  // set Device ID
  void set_device_id(int device_id);
  int get_device_id() const;
  // set x86_math_num_threads
  void set_x86_math_num_threads(int threads);
  int x86_math_num_threads() const;
  ///////////////////////////////////////////////////////////////////////////////////////
  // set Model_dir
  void set_model_dir(const std::string& x);
  const std::string& model_dir() const;
  void set_valid_places(const std::vector<Place>& x);
  void set_model_file(const std::string& path);
  void set_param_file(const std::string& path);
  void set_model_buffer(const char* model_buffer,
                        size_t model_buffer_size,
                        const char* param_buffer,
                        size_t param_buffer_size);
  void set_model_buffer(std::shared_ptr<CxxModelBuffer> model_buffer);
  const CxxModelBuffer& get_model_buffer() const;
  // internal inference to choose passes for model optimizing,
  // it's designed for internal developer and not recommanded
  // for comman users.
  void set_passes_internal(
      const std::vector<std::string>& passes_internal = {});
  const std::vector<std::string>& get_passes_internal() const;
  const std::vector<Place>& valid_places() const;
  std::string model_file() const;
  std::string param_file() const;
  bool is_model_from_memory() const;
  // note: `model_from_memory` has the same effect as `is_model_from_memory`,
  // but is_model_from_memory is recommended and `model_from_memory` will be
  // abandoned in v3.0.
  bool model_from_memory() const;

  void set_multi_stream(bool multi_stream);
  bool multi_stream() const;

  // set MLU core version, which is used when compiling MLU kernels
  void set_mlu_core_version(lite_api::MLUCoreVersion core_version);
  // set MLU core number, which is used when compiling MLU kernels
  void set_mlu_core_number(int core_number);
  // whether use MLU's first conv kernel. First conv is a special kernel
  // provided by MLU, its input is uint8, and also needs two 3-dimentional
  // vectors which save all inputs' mean and std values
  // set the 3-dimentional mean vector and 3-dimentional std vector used by
  // MLU's first conv
  void set_mlu_firstconv_param(const std::vector<float>& mean,
                               const std::vector<float>& std);
  // set MLU input layout. User can specify layout of input data to be NHWC,
  // default is NCHW
  void set_mlu_input_layout(DataLayoutType layout);

  lite_api::MLUCoreVersion mlu_core_version() const;
  int mlu_core_number() const;
  DataLayoutType mlu_input_layout() const;
  // std::pair<mean, std>
  std::pair<std::vector<float>, std::vector<float>> mlu_firstconv_param() const;

  // XPU only, set the size of the workspace memory from L3 cache for the
  // current thread.
  void set_xpu_workspace_l3_size_per_thread(int l3_size = 0xfffc00);
  // XPU only, specify the target device ID for the current thread.
  // **DEPRECATED**, use xpu_set_device() at the very beginning of each worker
  // thread
  void set_xpu_dev_per_thread(int dev_no = 0);
  void set_xpu_multi_encoder_precision(const std::string& precision = "int16");

  // set input tensor for warmup.
  // It is optional. If you set prefered_inputs, model wil run immediately when
  // predictor is created
  template <class T>
  void set_preferred_inputs_for_warmup(const int group_idx,
                                       const int tensor_idx,
                                       const shape_t& shape,
                                       const lod_t& lod = {},
                                       const T fill_value = 0,
                                       const void* data = nullptr);
  const std::map<int, std::vector<std::shared_ptr<void>>>&
  preferred_inputs_for_warmup() const;

  void set_quant_model(bool quant_model);
  bool quant_model() const;
  void set_quant_type(QuantType quant_type);
  QuantType quant_type() const;

 private:
  class CxxConfigImpl;
  std::unique_ptr<CxxConfigImpl> pImpl;
};

/// MobileConfig is the config for light weight predictor (ARM platform)
class LITE_API MobileConfig {
 public:
  explicit MobileConfig(PowerMode mode = LITE_POWER_NO_BIND, int threads = 1);
  ///////////////////////////////////////////////////////////////////////////////////////
  // Basics
  // set Thread
  void set_threads(int threads);
  int threads() const;
  // set Power_mode
  void set_power_mode(PowerMode mode);
  PowerMode power_mode() const;
  // set GPU opencl tune
  void set_opencl_tune(CLTuneMode tune_mode = CL_TUNE_NONE,
                       size_t lws_repeats = 4);
  // set GPU opencl precision
  void set_opencl_precision(CLPrecisionType p = CL_PRECISION_AUTO);
  // set subgraph_model_dir
  void set_subgraph_model_cache_dir(std::string subgraph_model_cache_dir);
  const std::string& subgraph_model_cache_dir() const;
  void set_subgraph_model_cache_buffers(const std::string& key,
                                        const std::vector<char>& cfg,
                                        const std::vector<char>& bin);
  const std::map<std::string, std::pair<std::vector<char>, std::vector<char>>>&
  subgraph_model_cache_buffers() const;
  // set Device ID
  void set_device_id(int device_id);
  int get_device_id() const;
  // set x86_math_num_threads
  void set_x86_math_num_threads(int threads);
  int x86_math_num_threads() const;
  ///////////////////////////////////////////////////////////////////////////////////////

  // Load model from file or memory
  void set_model_from_file(const std::string& x);
  void set_model_from_buffer(const std::string& x);
  const std::string& lite_model_file() const;
  bool is_model_from_memory() const;

  // Modify L3Cache
  void SetArmL3CacheSize(
      L3CacheSetMethod method = L3CacheSetMethod::kDeviceL3Cache,
      int absolute_val = -1);

 private:
  class MobileConfigImpl;
  std::unique_ptr<MobileConfigImpl> pImpl;
};

template <typename ConfigT>
LITE_API std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT&);

}  // namespace lite_api
}  // namespace paddle

#endif  // NOLINT
