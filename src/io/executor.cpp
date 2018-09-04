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
#include <operators/math/gemm.h>
#include <algorithm>
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
#ifdef PADDLE_EXECUTOR_MULTITHREAD
#include <queue>
#include <utility>
#include "common/threadpool.h"
#endif

namespace paddle_mobile {
using framework::Variable;

char *Get_binary_data(std::string filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                        filename.c_str());
  fseek(file, 0, SEEK_END);
  int64_t size = ftell(file);
  PADDLE_MOBILE_ENFORCE(size > 0, "size is too small");
  rewind(file);
  char *data = new char[size];
  size_t bytes_read = fread(data, 1, size, file);
  PADDLE_MOBILE_ENFORCE(bytes_read == size,
                        "read binary file bytes do not match with fseek");
  fclose(file);
  return data;
}

#pragma mark - executor
template <typename Dtype, Precision P>
Executor<Dtype, P>::Executor(const framework::Program<Dtype> p, int batch_size,
                             bool use_optimize, bool loddable)
    : program_(p),
      batch_size_(batch_size),
      use_optimize_(use_optimize),
      loddable_(loddable) {
  if (use_optimize_) {
    to_predict_program_ = program_.optimizeProgram;
  } else {
    to_predict_program_ = program_.originProgram;
  }
  Variable *variable_ptr = program_.scope->Var("batch_size");
  variable_ptr[0].SetValue<int>(batch_size);
  PADDLE_MOBILE_ENFORCE(to_predict_program_ != nullptr,
                        "to_predict_program_ == NULL!");
  const std::vector<std::shared_ptr<framework::BlockDesc>> blocks =
      to_predict_program_->Blocks();
#ifdef PADDLE_EXECUTOR_MULTITHREAD
  depManager.resize(blocks.size());
#endif
  for (int i = 0; i < blocks.size(); ++i) {
    std::shared_ptr<framework::BlockDesc> block_desc = blocks[i];
    std::vector<std::shared_ptr<framework::OpDesc>> ops = block_desc->Ops();
    for (int j = 0; j < ops.size(); ++j) {
      std::shared_ptr<framework::OpDesc> op = ops[j];
      DLOG << "create op: " << op->Type();
      auto op_base = framework::OpRegistry<Dtype>::CreateOp(
          op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap(),
          program_.scope);
      DLOG << "executer in loaddable mode: " << loddable_;
      // use pre_infershape to pre resize , but if u use an lod mode tensor u
      // need to resize in runtime
      if (!loddable_) {
        op_base->InferShape();
      }
      ops_of_block_[*block_desc.get()].push_back(op_base);
#ifdef PADDLE_EXECUTOR_MULTITHREAD
      depManager[i].analysisDep(ops_of_block_[*block_desc.get()]);
#endif
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

template <typename Dtype, Precision P>
void Executor<Dtype, P>::LoadMemory(const framework::VarDesc var_desc,
                                    framework::LoDTensor *tensor, char **data) {
  // 1. version
  uint32_t version = *reinterpret_cast<uint32_t *>(*data);

  (*data) += sizeof(uint32_t);

  // 2 Lod information
  uint64_t *lod_level_ptr = new uint64_t();
  memcpy(lod_level_ptr, (*data), sizeof(uint64_t));
  uint64_t lod_level = *lod_level_ptr;
  delete lod_level_ptr;
  (*data) += sizeof(uint64_t);

  auto &lod = *tensor->mutable_lod();
  lod.resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size = *reinterpret_cast<uint64_t *>(*data);
    (*data) += sizeof(uint64_t);
    DLOG << "lod size: " << i << size;
    std::vector<size_t> tmp(size / sizeof(size_t));

    for (int k = 0; k < tmp.size(); ++k) {
      tmp[k] = *reinterpret_cast<size_t *>(*data);
      (*data) += sizeof(size_t);
    }

    for (auto j : tmp) {
      LOG(kLOG_DEBUG1) << "    lod - " << j;
    }
    lod[i] = tmp;
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

  const framework::TensorDesc &desc = var_desc.Tensor_desc();
  int memory_size = 1;
  for (auto l : desc.Dims()) {
    memory_size *= l;
  }

  tensor->Resize(framework::make_ddim(desc.Dims()));

  void *memory = nullptr;
  int type_size = 0;
  switch (desc.DataType()) {
    case framework::VARTYPE_TYPE_FP16:
      type_size = 2;
      break;
    case framework::VARTYPE_TYPE_FP32:
      type_size = 4;
      memory = tensor->mutable_data<float>();
      break;
    case framework::VARTYPE_TYPE_FP64:
      type_size = 8;
      break;
    case framework::VARTYPE_TYPE_INT32:
      type_size = 4;
      break;
    case framework::VARTYPE_TYPE_INT64:
      type_size = 8;
      break;
    case framework::VARTYPE_TYPE_BOOL:
      type_size = 1;
      break;
    default:
      break;
  }
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

template <typename Dtype, Precision P>
void Executor<Dtype, P>::InitMemory() {
  for (const auto &block : to_predict_program_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        auto tensor = var->template GetMutable<framework::LoDTensor>();
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          continue;
        }

        char *origin_data =
            Get_binary_data(program_.model_path + "/" + var_desc->Name());
        char *data = origin_data;
        LoadMemory(*var_desc, tensor, &data);
        delete origin_data;
      } else {
        if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
          DLOG << "var_desc->Name():  " << var_desc->Name();
          DLOG << "var_desc->Tensor_desc().DataType():  "
               << var_desc->Tensor_desc().DataType();
          bool is_mute_match;
          framework::LoDTensor *tensor = nullptr;

          is_mute_match = varInputMemory(var_desc, var, tensor);

          PADDLE_MOBILE_ENFORCE(
              is_mute_match,
              "got unhandled var_desc->Tensor_desc().DataType(): %d",
              var_desc->Tensor_desc().DataType());
        }
      }
    }
  }
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::InitCombineMemory() {
  char *origin_data;
  if (program_.combined_params_buf && program_.combined_params_len) {
    LOG(kLOG_INFO) << "use outter memory";
    origin_data = (char *)program_.combined_params_buf;
  } else {
    LOG(kLOG_INFO) << " begin init combine memory";
    origin_data = Get_binary_data(program_.para_path);
  }
  PADDLE_MOBILE_ENFORCE(origin_data != nullptr, "origin_data==nullptr!!!");
  char *data = origin_data;
  for (const auto &block : to_predict_program_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        auto tensor = var->template GetMutable<framework::LoDTensor>();
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          continue;
        }
        LoadMemory(*var_desc, tensor, &data);
      } else {
        if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
          DLOG << "var_desc->Name():  " << var_desc->Name();
          DLOG << "var_desc->Tensor_desc().DataType():  "
               << var_desc->Tensor_desc().DataType();
          bool is_mute_match = false;
          framework::LoDTensor *tensor;

          is_mute_match = varInputMemory(var_desc, var, tensor);

          PADDLE_MOBILE_ENFORCE(
              is_mute_match,
              "got unhandled var_desc->Tensor_desc().DataType(): %d",
              var_desc->Tensor_desc().DataType());
        }
      }
    }
  }
  delete origin_data;
  LOG(kLOG_INFO) << " end init combine memory ";
}
template <typename Dtype, Precision P>
bool Executor<Dtype, P>::varInputMemory(
    const std::shared_ptr<framework::VarDesc> &var_desc, Variable *var,
    framework::LoDTensor *tensor) const {
  bool is_mute_match = false;
  switch (var_desc->Tensor_desc().DataType()) {
    case framework::VARTYPE_TYPE_FP16: {
      break;
    }

    case framework::VARTYPE_TYPE_FP32: {
      tensor = var->template GetMutable<framework::LoDTensor>();
      tensor->template mutable_data<Ptype>();
      is_mute_match = true;
      break;
    }

    case framework::VARTYPE_TYPE_FP64: {
      break;
    }

    case framework::VARTYPE_TYPE_INT32: {
      break;
    }

    case framework::VARTYPE_TYPE_INT64: {
      tensor = var->template GetMutable<framework::LoDTensor>();
      tensor->template mutable_data<int64_t>();
      is_mute_match = true;
      break;
    }
    case framework::VARTYPE_TYPE_BOOL: {
      break;
    }

    default: { break; }
  }

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
#ifdef PADDLE_EXECUTOR_MULTITHREAD
  std::mutex m;
  std::condition_variable cv;
  std::queue<int> next;
  next.push(0);
  int rsize = ops.size();
  std::vector<int> status(rsize, 0);
  auto &threadPool = ThreadPool::getThreadPool();
  auto &dep = depManager[0];
  auto finishF = [&ops, &m, &cv, &next, &status, &rsize, &dep](int opi) {
    std::lock_guard<std::mutex> lk(m);
    rsize--;
    status[opi] = 2;
    for (int i : dep.getNext(opi)) {
      bool ok = true;
      for (int j : dep.getDeps(i)) {
        if (status[j] != 2) {
          ok = false;
          break;
        }
      }
      if (ok && (status[i] == 0)) {
        next.push(i);
      }
    }
    cv.notify_one();
  };
  for (;;) {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&next, &rsize] { return rsize == 0 || !next.empty(); });
    if (rsize == 0) {
      break;
    }
    while (next.size() > 0) {
      int opi = next.front();
      next.pop();
      status[opi] = 1;
      threadPool.enqueue([opi, &ops, &finishF, &profile] {
        auto &op = ops[opi];
#ifdef PADDLE_MOBILE_PROFILE
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        profile[opi].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
        profile[opi].tid = ThreadPool::getThreadPoolThreadId();
#endif
        ops[opi]->Run();
#ifdef PADDLE_MOBILE_PROFILE
        clock_gettime(CLOCK_MONOTONIC, &ts);
        profile[opi].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
        finishF(opi);
      });
    }
  }
#else
  for (int i = 0; i < ops.size(); i++) {
#ifdef PADDLE_MOBILE_PROFILE
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
    DLOG << "executer Predict in3.3";

    // to Run
    ops[i]->Run();
#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
  }
#endif
  DLOG << "executer Predict in4";

  auto last_op = ops.rbegin();

  auto output_map = (*last_op)->Outputs();
  std::vector<std::string> out_keys = (*last_op)->GetOutKeys();
  PADDLE_MOBILE_ENFORCE(out_keys.size() > 0, "the last op contains no output");
  framework::LoDTensor *output_tensor =
      framework::GetVarValue<framework::LoDTensor>(out_keys[0], output_map,
                                                   *(program_.scope));
#ifdef PADDLE_MOBILE_PROFILE
#ifdef PADDLE_EXECUTOR_MULTITHREAD
  // TODO(haipeng): expose profile info as an interface, user can get them to
  // analysis
  //      the performance of their deepnet.
  FILE *df = fopen("net.dot", "w");
  fprintf(df, "digraph {\n");
  for (int i = 0; i < ops.size(); i++) {
    for (int j : dep.getNext(i)) {
      fprintf(df, "op_%d -> op_%d\n", i, j);
    }
  }
  for (int i = 0; i < ops.size(); i++) {
    fprintf(df, "op_%d[label=\"%s (%d)\"]\n", i, ops[i]->Type().c_str(), i);
  }
  fprintf(df, "}\n");
  fclose(df);
#endif
  DLOG << "executer Predict in5";

  //  FILE *pf = fopen("profile.out", "w");
  std::unordered_map<std::string, uint64_t> _tp;
  for (int i = 0; i < profile.size(); i++) {
    const auto &pInfo = profile[i];
    uint64_t timeCost = pInfo.runEnd - pInfo.runBegin;
    _tp[ops[i]->Type()] += timeCost;
    //    fprintf(pf, "%d\t%s\t%d\t%llu\t%llu\t%llu\n", i,
    //    ops[i]->Type().c_str(),
    //            pInfo.tid, pInfo.runBegin, pInfo.runEnd, timeCost);
  }
  //  fclose(pf);
  DLOG << "executer Predict in6";

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
  DLOG << "executer Predict out";

  return std::make_shared<framework::Tensor>(framework::Tensor(*output_tensor));
}

template <typename Dtype, Precision P>
std::shared_ptr<framework::LoDTensor> Executor<Dtype, P>::PredictLod(
    const framework::LoDTensor &t) {
  DLOG << "execute  PredictLod :lod" << t.lod();

  DLOG << "executer Predict in";
  framework::Variable *g_feed_value = program_.scope->Var("feed");
  framework::LoDTensor *feed_tensor =
      g_feed_value->GetMutable<framework::LoDTensor>();

  DLOG << "executer Predict in2";

  feed_tensor->Resize(t.dims());
  feed_tensor->ShareDataWith(t);
  feed_tensor->set_lod(t.lod());
  DLOG << "feed_tensor .lod : " << feed_tensor->lod();

  DLOG << "executer Predict in3";

  std::shared_ptr<framework::BlockDesc> to_predict_block =
      to_predict_program_->Block(0);
  DLOG << "executer Predict in3.1";

  auto &ops = ops_of_block_[*to_predict_block.get()];
  DLOG << "executer Predict in3.2";

#ifdef PADDLE_MOBILE_PROFILE
  std::vector<ProfInfo> profile(ops.size());
#endif
#ifdef PADDLE_EXECUTOR_MULTITHREAD
  std::mutex m;
  std::condition_variable cv;
  std::queue<int> next;
  next.push(0);
  int rsize = ops.size();
  std::vector<int> status(rsize, 0);
  auto &threadPool = ThreadPool::getThreadPool();
  auto &dep = depManager[0];
  auto finishF = [&ops, &m, &cv, &next, &status, &rsize, &dep](int opi) {
    std::lock_guard<std::mutex> lk(m);
    rsize--;
    status[opi] = 2;
    for (int i : dep.getNext(opi)) {
      bool ok = true;
      for (int j : dep.getDeps(i)) {
        if (status[j] != 2) {
          ok = false;
          break;
        }
      }
      if (ok && (status[i] == 0)) {
        next.push(i);
      }
    }
    cv.notify_one();
  };
  for (;;) {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&next, &rsize] { return rsize == 0 || !next.empty(); });
    if (rsize == 0) {
      break;
    }
    while (next.size() > 0) {
      int opi = next.front();
      next.pop();
      status[opi] = 1;
      threadPool.enqueue([opi, &ops, &finishF, &profile] {
        auto &op = ops[opi];
#ifdef PADDLE_MOBILE_PROFILE
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        profile[opi].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
        profile[opi].tid = ThreadPool::getThreadPoolThreadId();
#endif
        ops[opi]->Run();
#ifdef PADDLE_MOBILE_PROFILE
        clock_gettime(CLOCK_MONOTONIC, &ts);
        profile[opi].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
        finishF(opi);
      });
    }
  }
#else
  for (int i = 0; i < ops.size(); i++) {
#ifdef PADDLE_MOBILE_PROFILE
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
    DLOG << "executer Predict in3.3 infer";
    if (loddable_) {
      ops[i]->InferShape();
    }

    DLOG << "executer Predict in3.3 after infer";

    // to Run
    ops[i]->Run();
#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
  }
#endif
  DLOG << "executer Predict in4";

  auto last_op = ops.rbegin();

  auto output_map = (*last_op)->Outputs();
  std::vector<std::string> out_keys = (*last_op)->GetOutKeys();
  PADDLE_MOBILE_ENFORCE(out_keys.size() > 0, "the last op contains no output");
  framework::LoDTensor *output_tensor =
      framework::GetVarValue<framework::LoDTensor>(out_keys[0], output_map,
                                                   *(program_.scope));
#ifdef PADDLE_MOBILE_PROFILE
#ifdef PADDLE_EXECUTOR_MULTITHREAD
  // TODO(haipeng): expose profile info as an interface, user can get them to
  // analysis
  //      the performance of their deepnet.
  FILE *df = fopen("net.dot", "w");
  fprintf(df, "digraph {\n");
  for (int i = 0; i < ops.size(); i++) {
    for (int j : dep.getNext(i)) {
      fprintf(df, "op_%d -> op_%d\n", i, j);
    }
  }
  for (int i = 0; i < ops.size(); i++) {
    fprintf(df, "op_%d[label=\"%s (%d)\"]\n", i, ops[i]->Type().c_str(), i);
  }
  fprintf(df, "}\n");
  fclose(df);
#endif
  DLOG << "executer Predict in5";

  //  FILE *pf = fopen("profile.out", "w");
  std::unordered_map<std::string, uint64_t> _tp;
  for (int i = 0; i < profile.size(); i++) {
    const auto &pInfo = profile[i];
    uint64_t timeCost = pInfo.runEnd - pInfo.runBegin;
    _tp[ops[i]->Type()] += timeCost;
    //    fprintf(pf, "%d\t%s\t%d\t%llu\t%llu\t%llu\n", i,
    //    ops[i]->Type().c_str(),
    //            pInfo.tid, pInfo.runBegin, pInfo.runEnd, timeCost);
  }
  //  fclose(pf);
  DLOG << "executer Predict in6";

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
  DLOG << "executer Predict out";

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

template class Executor<CPU, Precision::FP32>;
template class Executor<GPU_MALI, Precision::FP32>;
template class Executor<FPGA, Precision::FP32>;

}  // namespace paddle_mobile
