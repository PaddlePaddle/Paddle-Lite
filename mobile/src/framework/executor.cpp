/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "framework/executor.h"
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/enforce.h"
#include "common/log.h"
#include "framework/context.h"
#include "framework/framework.pb-c.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/program-optimize/program_optimize.h"
#include "framework/program/program_desc.h"
#include "framework/program/var_desc.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "memory/t_malloc.h"
#include "pass/memory_optimize.h"
#include "pass/model_obfuscate.h"
#ifdef PADDLE_MOBILE_CL
#include "framework/cl/cl_image.h"
#include "pass/memory_optimize_super.h"
#endif

namespace paddle_mobile {
namespace framework {

#pragma mark - executor

template <typename Device, typename T>
void Executor<Device, T>::SetThreadNum(int thread_num, PowerMode power_mode) {
  CPUContext::Context()->set_thread_num(thread_num, power_mode);
}

template <typename Device, typename T>
Executor<Device, T>::Executor(const Program<Device> &program,
                              paddle_mobile::PaddleMobileConfigInternal config,
                              int batch_size, const bool use_optimize,
                              const bool lod_mode)
    : program_(program),
      batch_size_(batch_size),
      use_optimize_(use_optimize),
      lod_mode_(lod_mode),
      config_(config) {
  DLOG << "executor in lod mode: " << lod_mode;

  Variable *variable_ptr = program_.scope->Var("batch_size");
  variable_ptr->SetValue<int>(batch_size);

  program_desc_ =
      use_optimize_ ? program_.optimizeProgram : program_.originProgram;
  PADDLE_MOBILE_ENFORCE(program_desc_ != nullptr,
                        "program_desc_ should not be nullptr");
#if !defined(PADDLE_MOBILE_FPGA) && !defined(PADDLE_MOBILE_FPGA_KD) && \
    !defined(PADDLE_MOBILE_CL)
  if (config_.memory_optimization_level != NoMemoryOptimization) {
    pass::MemoryOptPass()(program_desc_.get(), program_.scope.get(),
                          config_.memory_optimization_level);
  }
#endif
  // resize feed and fetch list
  // should init feed and fetch variables before infer shape
  InitFeedFetchList();
  const auto &blocks = program_desc_->Blocks();
  std::shared_ptr<BlockDesc> block_desc = blocks[0];
  std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
  for (int j = 0; j < ops.size(); ++j) {
    std::shared_ptr<OpDesc> op_desc = ops[j];
    DLOG << "create op: " << op_desc->Type();

    auto op_handler = OpRegistry<Device>::CreateOp(
        op_desc->Type(), op_desc->GetInputs(), op_desc->GetOutputs(),
        op_desc->GetAttrMap(), program_.scope.get());
    // infer shape to reshape inputs and outputs before predict,
    // but for lod mode, it still need to infer shape in runtime
    if (!lod_mode) {
      op_handler->InferShape();
    }
    ops_of_block0_.push_back(op_handler);
  }
#ifdef PADDLE_MOBILE_FPGA_V2
  InitQuantMemory();
#endif
  if (program_.combined) {
    InitCombineMemory();
  } else {
    InitMemory();
  }
  int count = 0;
#ifdef PADDLE_MOBILE_PROFILE
    std::vector<ProfInfo> profile(ops_of_block0_.size());
    struct timespec ts;
    int op_index = 0;
#endif
  for (auto &op_handler : ops_of_block0_) {
#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[op_index].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
    DLOG << "Initialize op[" << count++ << "]: " << op_handler->Type();
    op_handler->Init();
#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
      profile[op_index].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
      ++op_index;
#endif
  }
#ifdef PADDLE_MOBILE_PROFILE
  printf("================[ op init profile ]==================\n");
  PrintProfile(profile);
#endif
}

template <typename Device, typename T>
void Executor<Device, T>::InitFeedFetchList() {
  std::unordered_map<std::string, int> feed_indices, fetch_indices;
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &op_desc : block->Ops()) {
      if (op_desc->Type() == "feed") {
        std::string name = op_desc->Output("Out")[0];
        feed_indices[name] = op_desc->GetAttr("col").Get<int>();
      } else if (op_desc->Type() == "fetch") {
        std::string name = op_desc->Input("X")[0];
        fetch_indices[name] = op_desc->GetAttr("col").Get<int>();
      }
    }
  }
  feed_indices_.swap(feed_indices);
  fetch_indices_.swap(fetch_indices);

  auto *feed_var = program_.scope->Var("feed");
  auto *feed_list = feed_var->template GetMutable<framework::LoDTensorArray>();
  feed_list->resize(feed_indices_.size());

  auto *fetch_var = program_.scope->Var("fetch");
  auto *fetch_list =
      fetch_var->template GetMutable<framework::LoDTensorArray>();
  fetch_list->resize(fetch_indices_.size());
}

template <typename T>
static void LoadMemInternal(void **data, LoDTensor *tensor,
                            bool quant_uint8 = false) {
  char **data_buf = reinterpret_cast<char **>(data);
  int64_t size = tensor->numel();
  T *tensor_data = tensor->mutable_data<T>();
  if (quant_uint8) {
    // should be moved into operator init function
    float min_value;
    float max_value;
    memory::Copy(&min_value, *data_buf, sizeof(float));
    memory::Copy(&max_value, *data_buf + sizeof(float), sizeof(float));
    *data_buf += 2 * sizeof(float);
    const float factor = (max_value - min_value) / 255.0;
    const uint8_t *uint8_data = reinterpret_cast<uint8_t *>(*data_buf);
    for (int k = 0; k < size; ++k) {
      tensor_data[k] = uint8_data[k] * factor + min_value;
    }
    *data_buf += size * sizeof(uint8_t);
  } else {
    memory::Copy(tensor_data, *data_buf, size * sizeof(T));
    *data_buf += size * sizeof(T);
  }
}

template <typename Device, typename T>
void Executor<Device, T>::LoadMemory(void **data,
                                     const std::shared_ptr<VarDesc> var_desc,
                                     LoDTensor *tensor) {
  char **data_buf = reinterpret_cast<char **>(data);
  // version
  uint32_t version = *(reinterpret_cast<uint32_t *>(*data_buf));
  *data_buf += sizeof(uint32_t);
  // lod information
  // uint64_t lod_level = *(reinterpret_cast<uint64_t *>(*data_buf));
  uint64_t lod_level = 0;
  memory::Copy(&lod_level, *data_buf, sizeof(uint64_t));
  *data_buf += sizeof(uint64_t);

  auto *lod = tensor->mutable_lod();
  lod->resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size = *(reinterpret_cast<uint64_t *>(*data_buf));
    *data_buf += sizeof(uint64_t);
    std::vector<size_t> tmp_dim(size / sizeof(size_t));
    memory::Copy(tmp_dim.data(), *data_buf, size);
    (*lod)[i] = std::move(tmp_dim);
    *data_buf += size;
  }
  // tensor version
  uint32_t tensor_version = *(reinterpret_cast<uint32_t *>(*data_buf));
  *data_buf += sizeof(uint32_t);
  // tensor desc size
  int32_t tensor_desc_size = *(reinterpret_cast<int32_t *>(*data_buf));
  *data_buf += sizeof(int32_t);
  // skip tensor desc
  *data_buf += tensor_desc_size;

  const TensorDesc &tensor_desc = var_desc->Tensor_desc();
  tensor->Resize(make_ddim(tensor_desc.Dims()));
  // parse tensor from stream
  switch (tensor_desc.DataType()) {
    case VARTYPE_TYPE_FP32:
      LoadMemInternal<float>(reinterpret_cast<void **>(data_buf), tensor,
                             program_.quantification);
      break;
    case VARTYPE_TYPE_INT8:
      LoadMemInternal<int8_t>(reinterpret_cast<void **>(data_buf), tensor);
      break;
    case VARTYPE_TYPE_INT32:
      LoadMemInternal<int>(reinterpret_cast<void **>(data_buf), tensor);
      break;
    default:
      LOG(kLOG_ERROR) << "data type is not supported";
  }
}

template <typename Device, typename T>
void Executor<Device, T>::InitMemory() {
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          var->template GetMutable<framework::LoDTensorArray>();
          continue;
        }
        DLOG << "init persistable var: " << var_desc->Name();
        char *origin_data =
            ReadFileToBuff(program_.model_path + "/" + var_desc->Name());
        char *data = origin_data;
        auto tensor = var->template GetMutable<LoDTensor>();
        LoadMemory(reinterpret_cast<void **>(&data), var_desc, tensor);
        delete[] origin_data;
      } else {
        DLOG << "init no persistable var: " << var_desc->Name();
        varInputMemory(var_desc, var);
      }
    }
  }
}

template <typename Device, typename T>
void Executor<Device, T>::InitCombineMemory() {
  char *origin_data = nullptr;
  bool self_alloc = false;
  if (program_.combined_params_buf && program_.combined_params_len) {
    origin_data = reinterpret_cast<char *>(
        const_cast<uint8_t *>(program_.combined_params_buf));
    if (config_.model_obfuscate_key != "") {
      auto obfuscator = pass::ModelObfuscatePass(config_.model_obfuscate_key);
      obfuscator.convert_data(origin_data, program_.combined_params_len);
    }
  } else {
    self_alloc = true;
    origin_data = ReadFileToBuff(program_.para_path);
    if (config_.model_obfuscate_key != "") {
      auto obfuscator = pass::ModelObfuscatePass(config_.model_obfuscate_key);
      obfuscator.convert_data(origin_data, GetFileLength(program_.para_path));
    }
  }
  PADDLE_MOBILE_ENFORCE(origin_data != nullptr, "data == nullptr");
  char *data = origin_data;
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          var->template GetMutable<framework::LoDTensorArray>();
          continue;
        }

        DLOG << " init combine memory persistable: " << var_desc->Name();
        auto tensor = var->template GetMutable<LoDTensor>();
        LoadMemory(reinterpret_cast<void **>(&data), var_desc, tensor);
      } else {
        DLOG << " init combine memory no persistable: " << var_desc->Name();
        varInputMemory(var_desc, var);
      }
    }
  }
  if (self_alloc) {
    delete[] origin_data;
  }
  LOG(kLOG_INFO) << "init combine memory finish";
}

static void ClearNoPersistableTensorArray(const framework::ProgramDesc *program,
                                          framework::Scope *scope) {
  for (const auto &block : program->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      if (!var_desc->Persistable() &&
          var_desc->Type() == VARTYPE_TYPE_STEP_LOD_TENSOR_ARRAY) {
        auto var = scope->Var(var_desc->Name());
        auto array = var->template GetMutable<framework::LoDTensorArray>();
        array->resize(1);
      }
    }
  }
}

template <typename Device, typename T>
void Executor<Device, T>::InitNoPersistableMemory(const Tensor &input_tensor) {
  if (input_tensor.dims().size() != 4) {
    return;
  }
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (!var_desc->Persistable() &&
          var_desc->Type() == VARTYPE_TYPE_LOD_TENSOR) {
        DLOG << "InitNoPersistableMemory var " << var_desc->Name();
        auto tensor = var->template GetMutable<LoDTensor>();
        if (tensor->IsInitialized() && tensor->dims().size() == 4) {
          DLOG << "var's tensor is Initialized or dims size != 4";
          DDim tensor_dim = tensor->dims();
          DDim new_dim =
              make_ddim({tensor_dim[0], tensor_dim[1], input_tensor.dims()[2],
                         input_tensor.dims()[3]});
          tensor->Resize(new_dim);
          tensor->template mutable_data_new<T>();
          DLOG << "var's tensor dims " << tensor_dim;
          DLOG << "var's tensor new dims " << new_dim;
        } else {
          DLOG << "var's tensor is not Initialized ???";
        }
      }
    }
  }
}

template <typename Device, typename T>
bool Executor<Device, T>::varInputMemory(
    const std::shared_ptr<VarDesc> &var_desc, Variable *var) const {
#ifdef PADDLE_MOBILE_FPGA
  framework::LoDTensor *tensor = var->template GetMutable<LoDTensor>();
#ifdef PADDLE_MOBILE_FPGA_V2
  tensor->init(type_id<int8_t>().hash_code());
#else
  tensor->init(type_id<float>().hash_code());
#endif
  return true;
#endif

  auto type = var_desc->Type();
  if (type == VARTYPE_TYPE_LOD_TENSOR) {
    auto data_type = var_desc->Tensor_desc().DataType();
    framework::LoDTensor *tensor = var->template GetMutable<LoDTensor>();
  } else if (type == VARTYPE_TYPE_STEP_SCOPES) {
    std::vector<framework::Scope *> *step_scopes =
        var->template GetMutable<std::vector<framework::Scope *>>();
  } else if (type == VARTYPE_TYPE_STEP_LOD_TENSOR_ARRAY) {
    framework::LoDTensorArray *tensor_array =
        var->template GetMutable<framework::LoDTensorArray>();
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION("got unhandled var type `%d`", type);
  }
  return true;
}

template <typename Device, typename T>
PMStatus Executor<Device, T>::Predict(
    const std::vector<std::pair<std::string, Tensor>> &inputs) {
  for (const auto &input : inputs) {
    SetInput(input.second, input.first);
  }
  return this->Predict();
}

template <typename Device, typename T>
PMStatus Executor<Device, T>::Predict(
    const std::vector<std::pair<std::string, LoDTensor>> &inputs) {
  for (const auto &input : inputs) {
    SetInput(input.second, input.first);
  }
  return this->Predict();
}

template <typename Device, typename T>
std::vector<T> Executor<Device, T>::Predict(const std::vector<T> &input,
                                            const std::vector<int64_t> &dims) {
  PADDLE_MOBILE_ENFORCE(feed_indices_.size() != 0,
                        "We don't know which tensor should be assign, since no "
                        "feed op found in this model");
  PADDLE_MOBILE_ENFORCE(fetch_indices_.size() != 0,
                        "We don't know which tensor should be fetch out, since "
                        "no fetch op found in this model");
  std::string input_name = feed_indices_.begin()->first;
  Tensor feed_tensor(input, make_ddim(dims));
  SetInput(feed_tensor, input_name);
  std::vector<T> output;
  if (this->Predict() == PMSuccess) {
    std::string output_name = fetch_indices_.begin()->first;
    const auto output_tensor = GetOutput(output_name);
    output.resize(output_tensor->numel());
    memcpy(output.data(), output_tensor->template data<T>(),
           output.size() * sizeof(T));
  }
  return output;
}

template <typename Device, typename T>
void Executor<Device, T>::SetInput(const Tensor &input,
                                   const std::string &var_name) {
  int index = 0;
  if (feed_indices_.find(var_name) != feed_indices_.end()) {
    index = feed_indices_.find(var_name)->second;
  }
  auto *feed_var = program_.scope->Var("feed");
  framework::LoDTensor &target =
      feed_var->template GetMutable<framework::LoDTensorArray>()->at(index);

  target.Resize(input.dims());
  target.ShareDataWith(input);
  if (feed_indices_.size() == 1) {
    auto &dim = input.dims();
    if (lod_mode_ && product(dim) < 0.9 * product(input_dim_last_)) {
      InitNoPersistableMemory(target);
    }
    input_dim_has_changed_ = input_dim_last_ != dim;
    input_dim_last_ = static_cast<DDim>(dim);
  }
}

template <typename Device, typename T>
void Executor<Device, T>::SetInput(const LoDTensor &input,
                                   const std::string &var_name) {
  int index = 0;
  if (feed_indices_.find(var_name) != feed_indices_.end()) {
    index = feed_indices_.find(var_name)->second;
  }
  auto *feed_var = program_.scope->Var("feed");
  framework::LoDTensor &target =
      feed_var->template GetMutable<framework::LoDTensorArray>()->at(index);

  target.Resize(input.dims());
  target.ShareDataWith(input);
  target.set_lod(input.lod());
  if (feed_indices_.size() == 1) {
    auto &dim = input.dims();
    if (lod_mode_ && product(dim) < 0.9 * product(input_dim_last_)) {
      InitNoPersistableMemory(target);
    }
    input_dim_has_changed_ = input_dim_last_ != dim;
    input_dim_last_ = static_cast<DDim>(dim);
  }
}

template <typename Device, typename T>
std::shared_ptr<LoDTensor> Executor<Device, T>::GetOutput(
    const std::string &var_name) {
  const auto &iter = fetch_indices_.find(var_name);
  if (var_name == "fetch" || iter != fetch_indices_.end()) {
    int index = 0;
    if (iter != fetch_indices_.end()) {
      index = iter->second;
    }
    auto *fetch_var = program_.scope->Var("fetch");
    framework::LoDTensor &target =
        fetch_var->template GetMutable<framework::LoDTensorArray>()->at(index);

    return std::make_shared<LoDTensor>(target);
  } else {
    auto *fetch_var = program_.scope->Var(var_name);
    framework::LoDTensor *target =
        fetch_var->template GetMutable<framework::LoDTensor>();
    return std::make_shared<LoDTensor>(*target);
  }
}

#ifdef PADDLE_MOBILE_CL
template <typename Device, typename T>
const CLImage *Executor<Device, T>::GetOutputImage(
    const std::string &var_name) {
  auto var = program_.scope->FindVar(var_name);
  if (var->IsInitialized() && var->template IsType<framework::CLImage>()) {
    const CLImage *cl_image = var->template Get<framework::CLImage>();
    return cl_image;
  } else {
    return nullptr;
  }
}
#endif

template <typename Device, typename T>
PMStatus Executor<Device, T>::Predict() {
  try {
#if _OPENMP
    omp_set_num_threads(CPUContext::Context()->get_thread_num());
#endif
    // clear all no persistable tensor array since write_to_array
    // is always push back a new tensor in the array
    ClearNoPersistableTensorArray(program_desc_.get(), program_.scope.get());

#ifdef PADDLE_MOBILE_PROFILE
    std::vector<ProfInfo> profile(ops_of_block0_.size());
    struct timespec ts;
    int op_index = 0;
#endif
    for (int i = 0; i < ops_of_block0_.size(); ++i) {
      auto &op_handler = ops_of_block0_[i];
#ifdef PADDLE_MOBILE_PROFILE
      clock_gettime(CLOCK_MONOTONIC, &ts);
      profile[op_index].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
      DLOG << i << "th, "
           << "run op: " << op_handler->Type();
      if (lod_mode_ && input_dim_has_changed_) {
        op_handler->InferShape();
      }
      op_handler->Run();
#ifdef PADDLE_MOBILE_PROFILE
      clock_gettime(CLOCK_MONOTONIC, &ts);
      profile[op_index].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
      ++op_index;
#endif
    }
    if (feed_indices_.size() == 1) {
      input_dim_has_changed_ = false;
    }

#ifdef PADDLE_MOBILE_PROFILE
    PrintProfile(profile);
#endif
    return PMSuccess;
  } catch (PaddleMobileException &e) {
    exception_msg_ = e.what();
    return PMException;
  } catch (std::exception &e) {
    exception_msg_ = e.what();
    return PMException;
  }
}

#ifdef PADDLE_MOBILE_PROFILE
template <typename Device, typename T>
void Executor<Device, T>::PrintProfile(
    const vector<Executor<Device, T>::ProfInfo> &profile) const {
  std::unordered_map<std::string, uint64_t> _tp;
  for (int i = 0; i < profile.size(); i++) {
    const auto &pInfo = profile[i];
    uint64_t timeCost = pInfo.runEnd - pInfo.runBegin;
    if (this->ops_of_block0_[i]->Type() == "conv2d" ||
        this->ops_of_block0_[i]->Type() == "depthwise_conv2d") {
      auto inputs = this->ops_of_block0_[i]->Inputs();

      auto *filter = GetVarValue<ProfileTensorType>("Filter", inputs,
                                                    *(this->program_.scope));
      int kernel_size = filter->dims()[2];
      _tp[this->ops_of_block0_[i]->Type() + "_" +
          std::to_string(kernel_size)] += timeCost;
    } else {
      _tp[this->ops_of_block0_[i]->Type()] += timeCost;
    }
  }
  printf("====================[ profile ]======================\n");
  typedef std::pair<std::string, uint64_t> prof_t;
  std::vector<prof_t> _tv(_tp.begin(), _tp.end());
  uint64_t _ptotal = 0;
  for (auto const &p : _tv) {
    _ptotal += p.second;
  }
  auto compf = [](const prof_t &a, const prof_t &b) {
    return a.second > b.second;
  };
  std::sort(_tv.begin(), _tv.end(), compf);
  _tv.push_back(std::make_pair("total", _ptotal));
  for (auto const &p : _tv) {
    printf("%-16s\t%-10.0f\t%-2.4f\n", p.first.c_str(),
           static_cast<float>(p.second),
           static_cast<float>(p.second) / _ptotal * 100.0);
  }
  printf("====================[---------]======================\n");
}
#endif

template <typename Device, typename T>
void Executor<Device, T>::FeedTensorData(const vector<framework::Tensor> &v) {
  auto input_size = v.size();
  auto *feed_var = program_.scope->Var("feed");

  PADDLE_MOBILE_ENFORCE(input_size == feed_indices_.size(),
                        "input data number not correct");
  for (int i = 0; i < input_size; i++) {
    framework::LoDTensor &target =
        feed_var->template GetMutable<framework::LoDTensorArray>()->at(i);
    target.ShareDataWith(v[input_size - i - 1]);
  }
}

template <typename Device, typename T>
void Executor<Device, T>::GetTensorResults(
    std::vector<framework::Tensor *> *v) {
  auto *fetch_var = program_.scope->Var("fetch");
  auto output_size = fetch_indices_.size();
  for (int i = 0; i < output_size; i++) {
    framework::LoDTensor &target =
        fetch_var->template GetMutable<framework::LoDTensorArray>()->at(i);
    v->push_back(&target);
  }
}

template <typename Device, typename T>
std::string Executor<Device, T>::GetExceptionMsg() {
  return exception_msg_;
}

#ifdef PADDLE_MOBILE_FPGA
template <typename Device, typename T>
void Executor<Device, T>::InjectVariable(const Tensor &t,
                                         std::string var_name) {
  Variable *g_feed_value = program_.scope->Var(var_name);
  Tensor *feed_tensor = g_feed_value->template GetMutable<LoDTensor>();
  feed_tensor->Resize(t.dims());
  feed_tensor->ShareDataWith(t);
}

template <typename Device, typename T>
void Executor<Device, T>::FeedData(const Tensor &t) {
  InjectVariable(t, "feed0");
}

template <typename Device, typename T>
void Executor<Device, T>::FeedData(const std::vector<void *> &v) {
  auto input_size = v.size();
  int index = 0;
  // auto vars = program_.scope->VarContain("feed", &index);
  // PADDLE_MOBILE_ENFORCE(input_size == vars.size(),
  //                    "input data number not correct");
  for (int i = 0; i < input_size; i++) {
    auto var = program_.scope->Var("feed", i + index);
    auto feed_tensor = var->template GetMutable<LoDTensor>();
    feed_tensor->external_data = v[i];
  }
}

template <typename Device, typename T>
void Executor<Device, T>::GetResults(std::vector<void *> *v) {
  auto output_size = v->size();
  PADDLE_MOBILE_ENFORCE(output_size > 0, "Empty output");
  int index = 0;
  auto vars = program_.scope->VarContain("fetch", &index);
  PADDLE_MOBILE_ENFORCE(output_size == vars.size(),
                        "output data number not correct");

  for (int i = 0; i < output_size; i++) {
    auto var = program_.scope->Var("fetch", i + index);
    auto fetch_tensor = var->template GetMutable<LoDTensor>();
    (*v)[i] = fetch_tensor->template data<float>();
  }
}

template <typename Device, typename T>
framework::Tensor *Executor<Device, T>::GetTensorByName(
    const std::string &name) {
  auto var = program_.scope->Var(name);
  return var->template GetMutable<LoDTensor>();
}

template <typename Device, typename T>
std::shared_ptr<Tensor> Executor<Device, T>::FetchResult(int id) {
  auto &ops = ops_of_block0_;

  PADDLE_MOBILE_ENFORCE(id < (int)ops.size(), "Index out of range");
  auto op = id < 0 ? ops[ops.size() - 1] : ops[id];
  auto output_map = op->Outputs();
  std::vector<std::string> out_keys = op->GetOutKeys();
  PADDLE_MOBILE_ENFORCE(!out_keys.empty(), "this op contains no output");
  auto *output_tensor =
      GetVarValue<LoDTensor>(out_keys[0], output_map, *(program_.scope));
  return std::make_shared<Tensor>(Tensor(*output_tensor));
}

template <typename Device, typename T>
void Executor<Device, T>::Predict_From_To(int start, int end) {
  auto &ops = ops_of_block0_;
  end = end < 0 ? static_cast<int>(ops.size()) : end;
  PADDLE_MOBILE_ENFORCE(start >= 0 && start < end && end <= ops.size(),
                        "start or end parameter is wrong");

#ifdef PADDLE_MOBILE_PROFILE
  std::vector<ProfInfo> profile(ops.size());
#endif
  for (int i = start; i < end; i++) {
#ifdef PADDLE_MOBILE_PROFILE
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
    DLOG << "Running op: " << i << "  " << ops[i]->Type();
    ops[i]->Run();

#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
  }
}

template <typename Device, typename T>
void Executor<Device, T>::Predict_From(int start) {
  Predict_From_To(start);
}

template <typename Device, typename T>
void Executor<Device, T>::Predict_To(int end) {
  Predict_From_To(0, end);
}
#ifdef PADDLE_MOBILE_FPGA_V2
std::map<std::string, float> LoadQuantValFromFile(std::string filename) {
  std::map<std::string, float> quantValList;
  std::ifstream in;
  in.open(filename, std::ios::in);
  if (!in.is_open()) {
    // std::cout << "open File Failed." << std::endl;
    DLOG << "open File Failed.";
    exit(-1);
  }

  std::string line;
  while (getline(in, line)) {
    std::string splitStr = " : ";
    std::string::size_type pos;
    pos = line.find(splitStr);
    std::string subStr[2];
    subStr[0] = line.substr(0, pos);
    subStr[1] = line.substr(pos + splitStr.size(), line.size());
    quantValList.insert(std::make_pair(subStr[0], atof(subStr[1].c_str())));
  }
  in.close();
  return quantValList;
}

template <typename Device, typename T>
void Executor<Device, T>::InitQuantMemory() {
  std::string quantValFilePath;
  if (program_.combined) {
    quantValFilePath = program_.para_path;
    quantValFilePath =
        quantValFilePath.substr(0, (quantValFilePath.length() - 6));
    quantValFilePath = quantValFilePath + "scale";
  } else {
    quantValFilePath = program_.model_path + "/scale";
  }
  std::map<std::string, float> quantValList =
      LoadQuantValFromFile(quantValFilePath);
  auto ops = ops_of_block0_;
  for (int id = 0; id < ops.size(); id++) {
    auto op = ops[id];
    auto input_keys = op->GetInputKeys();
    auto inputs = op->Inputs();
    for (auto key = input_keys.begin(); key != input_keys.end(); key++) {
      auto inputs_vars = inputs[*key];
      int count = inputs_vars.size();
      for (int i = 0; i < count; i++) {
        if (inputs_vars[i] != "feed") {
          auto tensor = GetTensorByName(inputs_vars[i]);
          tensor->scale[0] = quantValList[inputs_vars[i]];
          DLOG << "input variance name : " << inputs_vars[i]
               << ", scale value : " << tensor->scale[0];
        }
      }
    }
    auto output_keys = op->GetOutKeys();
    auto outputs = op->Outputs();
    for (auto key = output_keys.begin(); key != output_keys.end(); key++) {
      auto outputs_vars = outputs[*key];
      int count = outputs_vars.size();
      for (int i = 0; i < count; i++) {
        if (outputs_vars[i] != "fetch") {
          auto tensor = GetTensorByName(outputs_vars[i]);
          tensor->scale[0] = quantValList[outputs_vars[i]];
          DLOG << "output variance name : " << outputs_vars[i]
               << ", scale value : " << tensor->scale[0];
        }
      }
    }
  }
}
#endif
#endif
#ifdef PADDLE_MOBILE_CL
template <>
void Executor<GPU_CL, float>::InitNoPersistableMemory(
    const Tensor &input_tensor) {
  DLOG << "CL InitNoPersistableMemory ";
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());

      if (var_desc->Persistable()) {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          var->template GetMutable<framework::LoDTensorArray>();
          continue;
        }
      } else {
        if (var_desc->Type() == VARTYPE_TYPE_LOD_TENSOR) {
          auto cl_image = var->template GetMutable<CLImage>();
          cl_context context = program_.scope->GetCLScpoe()->Context();
          cl_command_queue command_queue =
              program_.scope->GetCLScpoe()->CommandQueue();

          DDim tensor_dim = cl_image->dims();
          DDim new_dim =
              make_ddim({tensor_dim[0], tensor_dim[1], input_tensor.dims()[2],
                         input_tensor.dims()[3]});
          cl_image->Resize(new_dim);
          cl_image->InitEmptyImage(context, command_queue, new_dim);
        }
      }
    }
  }
  std::shared_ptr<LoDTensor> output = GetOutput("fetch");
  output->Resize(input_tensor.dims());
  output->mutable_data<float>();
}

template <>
void Executor<GPU_CL, float>::SetInput(const Tensor &input,
                                       const std::string &var_name) {
  int index = 0;
  if (feed_indices_.find(var_name) != feed_indices_.end()) {
    index = feed_indices_.find(var_name)->second;
  }
  auto *feed_var = program_.scope->Var("feed");
  framework::LoDTensor *input_tensor =
      &(feed_var->template GetMutable<framework::LoDTensorArray>()->at(index));

  DLOG << "config_.load_when_predict   " << config_.load_when_predict;
  DLOG << "target_tensor->IsInitialized() " << input_tensor->IsInitialized();
  DLOG << "target_tensor->dims()   " << input_tensor->dims();
  DLOG << "input.dims()   " << input.dims();
  DLOG << "input_dim_last_   " << input_dim_last_;
  if (config_.load_when_predict) {
    if (input_dim_last_ != input.dims()) {
      DLOG << "SetInput ---- > resize1";
      input_tensor->Resize(input.dims());
      input_tensor->mutable_data<float>();
      //     InitNoPersistableMemory(*input_tensor);
      pass::MemoryOptPassSuper()(program_desc_.get(), program_.scope.get(),
                                 config_.memory_optimization_level,
                                 input.dims());
    }
  } else {
    DLOG << "SetInput ---- > resize2";
    input_tensor->Resize(input.dims());
    DLOG << "SetInput ---- > ShareDataWith";
  }
  input_tensor->ShareDataWith(input);
  if (feed_indices_.size() == 1) {
    input_dim_has_changed_ = input_dim_last_ != input.dims();
  }
  auto &dim = input.dims();
  input_dim_last_ = static_cast<DDim>(dim);
}

template <typename Device, typename T>
void Executor<Device, T>::LoadMemory(const VarDesc var_desc, float *tensorInput,
                                     char **data) {}

template <>
void Executor<GPU_CL, float>::LoadMemory(const VarDesc var_desc,
                                         float *tensorInput, char **data) {
  // 1. version
  uint32_t version = *reinterpret_cast<uint32_t *>(*data);

  (*data) += sizeof(uint32_t);

  // 2 Lod information
  uint64_t *lod_level_ptr = new uint64_t();
  memcpy(lod_level_ptr, (*data), sizeof(uint64_t));
  uint64_t lod_level = *lod_level_ptr;
  delete lod_level_ptr;
  (*data) += sizeof(uint64_t);

  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size = *reinterpret_cast<uint64_t *>(*data);
    (*data) += sizeof(uint64_t);
    std::vector<size_t> tmp(size / sizeof(size_t));

    for (int k = 0; k < tmp.size(); ++k) {
      tmp[k] = *reinterpret_cast<size_t *>(*data);
      (*data) += sizeof(size_t);
    }
  }

  // 3. tensor version
  uint32_t tensor_version = *reinterpret_cast<uint32_t *>(*data);
  (*data) += sizeof(uint32_t);

  // 4. tensor desc
  int32_t size = *reinterpret_cast<int32_t *>(*data);
  (*data) += sizeof(int32_t);

  std::unique_ptr<char[]> buf(new char[size]);
  for (int m = 0; m < size; ++m) {
    buf.get()[m] = (*data)[m];
  }
  (*data) += (sizeof(char) * size);

  const TensorDesc &desc = var_desc.Tensor_desc();
  int memory_size = 1;
  for (auto l : desc.Dims()) {
    memory_size *= l;
  }

  void *memory = nullptr;
  int type_size = 4;
  memory = tensorInput;
  if (program_.quantification) {
    float min_value;
    float max_value;

    memcpy(&min_value, *data, sizeof(float));
    memcpy(&max_value, *data + sizeof(float), sizeof(float));
    *data += 2 * sizeof(float);
    const float factor = (max_value - min_value) / 255.0;
    uint8_t *uint8_data = reinterpret_cast<uint8_t *>(*data);
    for (int k = 0; k < memory_size; ++k) {
      static_cast<float *>(memory)[k] = uint8_data[k] * factor + min_value;
    }
    *data += (memory_size * sizeof(uint8_t));
  } else {
    for (int n = 0; n < memory_size; n++) {
      float value;
      memcpy(&value, *data + n * type_size, type_size);
      if (value < 1e-30 && value > -1e-30) {
        static_cast<float *>(memory)[n] = 0.0;
      } else {
        static_cast<float *>(memory)[n] = value;
      }
    }
    (*data) += (sizeof(char) * memory_size * type_size);
  }
}

template <>
void Executor<GPU_CL, float>::InitMemory() {
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        CLImage *cl_image = nullptr;
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          var->template GetMutable<framework::LoDTensorArray>();
          continue;
        } else {
          cl_image = var->template GetMutable<CLImage>();
        }

        char *origin_data =
            ReadFileToBuff(program_.model_path + "/" + var_desc->Name());
        char *data = origin_data;
        cl_context context = program_.scope->GetCLScpoe()->Context();
        const TensorDesc &desc = var_desc->Tensor_desc();
        int numel = 1;
        for (auto l : desc.Dims()) {
          numel *= l;
        }
        DLOG << var_desc->Name();
        float *tensorInput = static_cast<float *>(
            paddle_mobile::memory::Alloc(sizeof(float) * numel));
        LoadMemory(*var_desc, tensorInput, &data);

        DDim ddim = make_ddim(desc.Dims());

        // has not init
        cl_image->SetTensorData(tensorInput, ddim);

        delete origin_data;
        paddle_mobile::memory::Free(tensorInput);
      } else {
        if (var_desc->Type() == VARTYPE_TYPE_LOD_TENSOR) {
          auto cl_image = var->template GetMutable<CLImage>();
          cl_context context = program_.scope->GetCLScpoe()->Context();
          cl_command_queue command_queue =
              program_.scope->GetCLScpoe()->CommandQueue();

          const TensorDesc &desc = var_desc->Tensor_desc();
          //          DDim ddim = make_ddim(desc.Dims());
          DDim ddim = cl_image->dims();
          DLOG << var_desc->Name();
          cl_image->InitEmptyImage(context, command_queue, ddim);
        }
      }
    }
  }
}

template <>
void Executor<GPU_CL, float>::InitCombineMemory() {
  DLOG << "CL InitCombineMemory---- "
       << "config_.load_when_predict: " << config_.load_when_predict;
  char *origin_data = nullptr;
  bool self_alloc = false;
  if (program_.combined_params_buf && program_.combined_params_len) {
    LOG(kLOG_INFO) << "use outter memory";
    origin_data = reinterpret_cast<char *>(program_.combined_params_buf);
    if (config_.model_obfuscate_key != "") {
      auto obfuscator = pass::ModelObfuscatePass(config_.model_obfuscate_key);
      obfuscator.convert_data(origin_data, program_.combined_params_len);
    }
  } else {
    LOG(kLOG_INFO) << " begin init combine memory";
    self_alloc = true;
    origin_data = ReadFileToBuff(program_.para_path);
    if (config_.model_obfuscate_key != "") {
      auto obfuscator = pass::ModelObfuscatePass(config_.model_obfuscate_key);
      obfuscator.convert_data(origin_data, GetFileLength(program_.para_path));
    }
  }
  PADDLE_MOBILE_ENFORCE(origin_data != nullptr, "origin_data==nullptr!!!");
  float *data = reinterpret_cast<float *>(origin_data);

  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        CLImage *cl_image = nullptr;
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          var->template GetMutable<framework::LoDTensorArray>();
          continue;
        } else {
          cl_image = var->template GetMutable<CLImage>();
        }

        cl_context context = program_.scope->GetCLScpoe()->Context();

        const TensorDesc &desc = var_desc->Tensor_desc();
        DDim ddim = make_ddim(desc.Dims());

        int numel = 1;
        for (int i = 0; i < ddim.size(); i++) {
          numel = numel * ddim[i];
        }
        float *tensorInput = static_cast<float *>(
            paddle_mobile::memory::Alloc(sizeof(float) * numel));
        LoadMemory(*var_desc, tensorInput, &origin_data);

        // has not init
        cl_image->SetTensorData(tensorInput, ddim);

        paddle_mobile::memory::Free(tensorInput);
      } else {
        auto cl_image = var->template GetMutable<CLImage>();
        cl_context context = program_.scope->GetCLScpoe()->Context();
        cl_command_queue command_queue =
            program_.scope->GetCLScpoe()->CommandQueue();
        const TensorDesc &desc = var_desc->Tensor_desc();
        DDim ddim = cl_image->dims();
        bool shouldResize = true;
        if (ddim.size() > 4) {
          for (int i = 0; i < ddim.size() - 4; ++i) {
            if (ddim[i] != 0 && ddim[i] != 1) {
              shouldResize = false;
              break;
            }
          }
          if (shouldResize) {
            std::vector<int64_t> temp_intput_dims;
            temp_intput_dims.reserve(static_cast<size_t>(4));
            for (int i = ddim.size() - 4; i < ddim.size(); ++i) {
              temp_intput_dims.push_back(ddim[i]);
            }
            ddim = framework::make_ddim(temp_intput_dims);
          }
        }
        //  DDim ddim = make_ddim(desc.Dims());
        cl_image->InitEmptyImage(context, command_queue, ddim);
      }
    }
  }
  if (self_alloc) {
    delete data;
  }
  LOG(kLOG_INFO) << " end init combine memory ";
}

#endif

template class Executor<CPU, float>;

template class Executor<FPGA, float>;

template class Executor<GPU_CL, float>;

}  // namespace framework
}  // namespace paddle_mobile
