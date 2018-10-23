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

#include "io/executor.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "common/enforce.h"
#include "common/log.h"
#include "framework/framework.pb-c.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/program-optimize/program_optimize.h"
#include "framework/program/program_desc.h"
#include "framework/program/var_desc.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "operators/math/gemm.h"

namespace paddle_mobile {

using framework::Variable;

template <typename Dtype, Precision P>
Executor<Dtype, P>::Executor(const framework::Program<Dtype> p, int batch_size,
                             const bool use_optimize, const bool loddable)
    : program_(p),
      batch_size_(batch_size),
      use_optimize_(use_optimize),
      loddable_(loddable) {
  Variable *variable_ptr = program_.scope->Var("batch_size");
  variable_ptr->SetValue<int>(batch_size);
  to_predict_program_ =
      use_optimize_ ? program_.optimizeProgram : program_.originProgram;
  PADDLE_MOBILE_ENFORCE(to_predict_program_ != nullptr,
                        "to_predict_program_ == NULL!");
  const std::vector<std::shared_ptr<framework::BlockDesc>> &blocks =
      to_predict_program_->Blocks();

  DLOG << "executor in loaddable mode: " << loddable_;
  for (int i = 0; i < blocks.size(); ++i) {
    std::shared_ptr<framework::BlockDesc> block_desc = blocks[i];
    std::vector<std::shared_ptr<framework::OpDesc>> ops = block_desc->Ops();
    for (int j = 0; j < ops.size(); ++j) {
      std::shared_ptr<framework::OpDesc> op = ops[j];
      DLOG << "create op: " << op->Type();
      auto op_base = framework::OpRegistry<Dtype>::CreateOp(
          op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap(),
          program_.scope);
      // infer shape to reshape tensor before predict,
      // but for lod tensor, it will need to reshape in runtime
      if (!loddable_) {
        op_base->InferShape();
      }
      ops_of_block_[*block_desc.get()].push_back(op_base);
    }
  }
  if (program_.combined) {
    InitCombineMemory();
  } else {
    InitMemory();
  }
  std::shared_ptr<framework::BlockDesc> to_predict_block =
      to_predict_program_->Block(0);
  auto &ops = ops_of_block_[*to_predict_block.get()];
  for (const auto &op : ops) {
    op->Init();
  }
}

template <typename Dtype>
static void LoadMemInternal(void **data, framework::LoDTensor *tensor,
                            bool quant_uint8 = false) {
  char **data_buf = reinterpret_cast<char **>(data);
  int64_t size = tensor->numel();
  Dtype *tensor_data = tensor->mutable_data<Dtype>();
  if (quant_uint8) {
    // should be moved into operator init function
    float min_value;
    float max_value;
    memcpy(&min_value, data_buf, sizeof(float));
    memcpy(&max_value, data_buf + sizeof(float), sizeof(float));
    data_buf += 2 * sizeof(float);
    const float factor = (max_value - min_value) / 255.0;
    const uint8_t *uint8_data = reinterpret_cast<uint8_t *>(data_buf);
    for (int k = 0; k < size; ++k) {
      tensor_data[k] = uint8_data[k] * factor + min_value;
    }
    data_buf += size * sizeof(uint8_t);
  } else {
    memcpy(tensor_data, *data_buf, size * sizeof(Dtype));
    *data_buf += size * sizeof(Dtype);
  }
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::LoadMemory(
    void **data, const std::shared_ptr<framework::VarDesc> var_desc,
    framework::LoDTensor *tensor) {
  char **data_buf = reinterpret_cast<char **>(data);
  // version
  uint32_t version = *(reinterpret_cast<uint32_t *>(*data_buf));
  *data_buf += sizeof(uint32_t);
  // lod information
  // uint64_t lod_level = *(reinterpret_cast<uint64_t *>(*data_buf));
  uint64_t lod_level = 0;
  memcpy(&lod_level, *data_buf, sizeof(uint64_t));
  *data_buf += sizeof(uint64_t);

  auto *lod = tensor->mutable_lod();
  lod->resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size = *(reinterpret_cast<uint64_t *>(*data_buf));
    *data_buf += sizeof(uint64_t);
    std::vector<size_t> tmp_dim(size / sizeof(size_t));
    memcpy(tmp_dim.data(), *data_buf, size);
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

  const framework::TensorDesc &tensor_desc = var_desc->Tensor_desc();
  tensor->Resize(framework::make_ddim(tensor_desc.Dims()));
  // parse tensor from stream
  switch (tensor_desc.DataType()) {
    case framework::VARTYPE_TYPE_FP32:
      LoadMemInternal<float>(reinterpret_cast<void **>(data_buf), tensor,
                             program_.quantification);
      break;
    case framework::VARTYPE_TYPE_INT8:
      LoadMemInternal<int8_t>(reinterpret_cast<void **>(data_buf), tensor);
      break;
    case framework::VARTYPE_TYPE_INT32:
      LoadMemInternal<int>(reinterpret_cast<void **>(data_buf), tensor);
      break;
    default:
      LOG(kLOG_ERROR) << "data type is not supported";
  }
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::InitMemory() {
  for (const auto &block : to_predict_program_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      auto tensor = var->template GetMutable<framework::LoDTensor>();
      if (var_desc->Persistable()) {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          continue;
        }
        char *origin_data =
            ReadFileToBuff(program_.model_path + "/" + var_desc->Name());
        char *data = origin_data;
        LoadMemory(reinterpret_cast<void **>(&data), var_desc, tensor);
        delete[] origin_data;
      } else {
        if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
          varInputMemory(var_desc, var, tensor);
        }
      }
    }
  }
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::InitCombineMemory() {
  char *origin_data = nullptr;
  bool self_alloc = false;
  if (program_.combined_params_buf && program_.combined_params_len) {
    origin_data = reinterpret_cast<char *>(
        const_cast<uint8_t *>(program_.combined_params_buf));
  } else {
    self_alloc = true;
    origin_data = ReadFileToBuff(program_.para_path);
  }
  PADDLE_MOBILE_ENFORCE(origin_data != nullptr, "data == nullptr");
  char *data = origin_data;
  for (const auto &block : to_predict_program_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      auto tensor = var->template GetMutable<framework::LoDTensor>();
      if (var_desc->Persistable()) {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          continue;
        }
        LoadMemory(reinterpret_cast<void **>(&data), var_desc, tensor);
      } else {
        if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
          varInputMemory(var_desc, var, tensor);
        }
      }
    }
  }
  if (self_alloc) {
    delete[] origin_data;
  }
  LOG(kLOG_INFO) << "init combine memory finish";
}

template <typename Dtype, Precision P>
bool Executor<Dtype, P>::varInputMemory(
    const std::shared_ptr<framework::VarDesc> &var_desc, Variable *var,
    framework::LoDTensor *tensor) const {
  auto type = var_desc->Tensor_desc().DataType();
  switch (type) {
    case framework::VARTYPE_TYPE_FP32:
      tensor->mutable_data<float>();
      break;
    case framework::VARTYPE_TYPE_INT8:
      tensor->mutable_data<int8_t>();
      break;
    case framework::VARTYPE_TYPE_INT32:
      tensor->mutable_data<int32_t>();
      break;
    case framework::VARTYPE_TYPE_INT64:
      tensor->mutable_data<int64_t>();
      break;
    default:
      break;
  }
  bool is_mute_match = (type == framework::VARTYPE_TYPE_FP32) ||
                       (type == framework::VARTYPE_TYPE_INT8) ||
                       (type == framework::VARTYPE_TYPE_INT32) ||
                       (type == framework::VARTYPE_TYPE_INT64);
  PADDLE_MOBILE_ENFORCE(is_mute_match, "got unhandled data type : %d", type);
  return is_mute_match;
}

template <typename Dtype, Precision P>
std::shared_ptr<framework::Tensor> Executor<Dtype, P>::Predict(
    const framework::Tensor &t) {
  framework::Variable *g_feed_value = program_.scope->Var("feed");
  framework::Tensor *feed_tensor =
      g_feed_value->GetMutable<framework::LoDTensor>();
  feed_tensor->Resize(t.dims());
  feed_tensor->ShareDataWith(t);
  std::shared_ptr<framework::BlockDesc> to_predict_block =
      to_predict_program_->Block(0);
  auto &ops = ops_of_block_[*to_predict_block.get()];

#ifdef PADDLE_MOBILE_PROFILE
  std::vector<ProfInfo> profile(ops.size());
#endif
  for (int i = 0; i < ops.size(); i++) {
#ifdef PADDLE_MOBILE_PROFILE
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
    // to Run
    ops[i]->Run();
#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
  }
  auto last_op = ops.rbegin();
  auto output_map = (*last_op)->Outputs();
  std::vector<std::string> out_keys = (*last_op)->GetOutKeys();
  PADDLE_MOBILE_ENFORCE(out_keys.size() > 0, "the last op contains no output");
  framework::LoDTensor *output_tensor =
      framework::GetVarValue<framework::LoDTensor>(out_keys[0], output_map,
                                                   *(program_.scope));
#ifdef PADDLE_MOBILE_PROFILE
  std::unordered_map<std::string, uint64_t> _tp;
  for (int i = 0; i < profile.size(); i++) {
    const auto &pInfo = profile[i];
    uint64_t timeCost = pInfo.runEnd - pInfo.runBegin;
    _tp[ops[i]->Type()] += timeCost;
  }
  printf("====================[ profile ]======================\n");
  using prof_t = std::pair<std::string, uint64_t>;
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
#endif
  return std::make_shared<framework::Tensor>(framework::Tensor(*output_tensor));
}

template <typename Dtype, Precision P>
std::shared_ptr<framework::LoDTensor> Executor<Dtype, P>::PredictLod(
    const framework::LoDTensor &t) {
  framework::Variable *g_feed_value = program_.scope->Var("feed");
  framework::LoDTensor *feed_tensor =
      g_feed_value->GetMutable<framework::LoDTensor>();
  feed_tensor->Resize(t.dims());
  feed_tensor->ShareDataWith(t);
  feed_tensor->set_lod(t.lod());

  std::shared_ptr<framework::BlockDesc> to_predict_block =
      to_predict_program_->Block(0);

  auto &ops = ops_of_block_[*to_predict_block.get()];

#ifdef PADDLE_MOBILE_PROFILE
  std::vector<ProfInfo> profile(ops.size());
#endif
  for (int i = 0; i < ops.size(); i++) {
#ifdef PADDLE_MOBILE_PROFILE
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
    if (loddable_) {
      ops[i]->InferShape();
    }
    ops[i]->Run();
#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
  }
  auto last_op = ops.rbegin();

  auto output_map = (*last_op)->Outputs();
  std::vector<std::string> out_keys = (*last_op)->GetOutKeys();
  PADDLE_MOBILE_ENFORCE(out_keys.size() > 0, "the last op contains no output");
  framework::LoDTensor *output_tensor =
      framework::GetVarValue<framework::LoDTensor>(out_keys[0], output_map,
                                                   *(program_.scope));
#ifdef PADDLE_MOBILE_PROFILE
  std::unordered_map<std::string, uint64_t> _tp;
  for (int i = 0; i < profile.size(); i++) {
    const auto &pInfo = profile[i];
    uint64_t timeCost = pInfo.runEnd - pInfo.runBegin;
    _tp[ops[i]->Type()] += timeCost;
  }
  printf("====================[ profile ]======================\n");
  using prof_t = std::pair<std::string, uint64_t>;
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
#endif
  return std::make_shared<framework::LoDTensor>(
      framework::LoDTensor(*output_tensor));
}

template <typename Dtype, Precision P>
std::shared_ptr<framework::Tensor> Executor<Dtype, P>::Predict(
    const framework::Tensor &t, int block_id) {
  return Predict(t);
}

template <typename Dtype, Precision P>
std::vector<typename Executor<Dtype, P>::Ptype> Executor<Dtype, P>::Predict(
    const std::vector<Ptype> &input, const std::vector<int64_t> &dims) {
  framework::Tensor tensor(input, framework::make_ddim(dims));
  std::shared_ptr<framework::Tensor> output_tensor = Predict(tensor, 0);
  Executor<Dtype, P>::Ptype *output_ptr =
      output_tensor->data<typename Executor<Dtype, P>::Ptype>();
  std::vector<typename Executor<Dtype, P>::Ptype> result_vector;
  for (int j = 0; j < output_tensor->numel(); ++j) {
    result_vector.push_back(output_ptr[j]);
  }
  return result_vector;
}

#ifdef PADDLE_MOBILE_FPGA
template <typename Dtype, Precision P>
void Executor<Dtype, P>::InjectVariable(const framework::Tensor &t,
                                        std::string var_name) {
  framework::Variable *g_feed_value = program_.scope->Var(var_name);
  framework::Tensor *feed_tensor =
      g_feed_value->GetMutable<framework::LoDTensor>();
  feed_tensor->Resize(t.dims());
  feed_tensor->ShareDataWith(t);
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::FeedData(const framework::Tensor &t) {
  InjectVariable(t, "feed");
}

template <typename Dtype, Precision P>
std::shared_ptr<framework::Tensor> Executor<Dtype, P>::FetchResult(int id) {
  std::shared_ptr<framework::BlockDesc> to_predict_block =
      to_predict_program_->Block(0);
  auto &ops = ops_of_block_[*to_predict_block.get()];

  PADDLE_MOBILE_ENFORCE(id < (int)ops.size(), "Index out of range");
  auto op = id < 0 ? ops[ops.size() - 1] : ops[id];
  auto output_map = op->Outputs();
  std::vector<std::string> out_keys = op->GetOutKeys();
  PADDLE_MOBILE_ENFORCE(!out_keys.empty(), "this op contains no output");
  auto *output_tensor = framework::GetVarValue<framework::LoDTensor>(
      out_keys[0], output_map, *(program_.scope));
  return std::make_shared<framework::Tensor>(framework::Tensor(*output_tensor));
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::Predict_From_To(int start, int end) {
  std::shared_ptr<framework::BlockDesc> to_predict_block =
      to_predict_program_->Block(0);
  auto &ops = ops_of_block_[*to_predict_block.get()];
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

template <typename Dtype, Precision P>
void Executor<Dtype, P>::Predict_From(int start) {
  Predict_From_To(start);
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::Predict_To(int end) {
  Predict_From_To(0, end);
}
#endif

template class Executor<CPU, Precision::FP32>;
template class Executor<GPU_MALI, Precision::FP32>;
template class Executor<FPGA, Precision::FP32>;

}  // namespace paddle_mobile
