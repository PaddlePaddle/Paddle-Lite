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

#include "io.h"
#include <vector>
#ifdef PADDLE_MOBILE_PROFILE
#include <ctime>
#include <map>
#endif

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

namespace paddle_mobile {
using framework::Variable;

char *Get_binary_data(std::string filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                        filename.c_str());
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  PADDLE_MOBILE_ENFORCE(size > 0, "size is too small");
  rewind(file);
  char *data = new char[size];
  size_t bytes_read = fread(data, 1, size, file);
  PADDLE_MOBILE_ENFORCE(bytes_read == size,
                        "read binary file bytes do not match with fseek");
  fclose(file);
  return data;
}

static size_t ReadBuffer(const char *file_name, uint8_t **out) {
  printf("%s \n", file_name);
  FILE *fp;
  fp = fopen(file_name, "rb");
  PADDLE_MOBILE_ENFORCE(fp != NULL, " %s open failed !", file_name);

  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  rewind(fp);

  DLOG << "model size: " << size;

  *out = reinterpret_cast<uint8_t *>(malloc(size));

  size_t cur_len = 0;
  size_t nread;
  while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
    cur_len += nread;
  }
  fclose(fp);
  return cur_len;
}

template <typename Dtype, Precision P>
const framework::Program<Dtype, P> Loader<Dtype, P>::Load(
    const std::string &dirname, bool optimize) {
  auto program = this->LoadProgram(dirname + "/__model__", optimize);
  program.model_path = dirname;
  return program;
}

template <typename Dtype, Precision P>
const framework::Program<Dtype, P> Loader<Dtype, P>::Load(
    const std::string &model_path, const std::string &para_path,
    bool optimize) {
  auto program = this->LoadProgram(model_path, optimize);
  program.para_path = para_path;
  program.is_commbine = true;
  return program;
}

template <typename Dtype, Precision P>
const framework::Program<Dtype, P> Loader<Dtype, P>::LoadProgram(
    const std::string &model_path, bool optimize) {
  std::string model_filename = model_path;
  PaddleMobile__Framework__Proto__ProgramDesc *c_program;
  uint8_t *buf = NULL;
  size_t read_size = ReadBuffer(model_filename.c_str(), &buf);

  PADDLE_MOBILE_ENFORCE(buf != NULL, "read from __model__ is null");

  c_program = paddle_mobile__framework__proto__program_desc__unpack(
      NULL, read_size, buf);
  //
  PADDLE_MOBILE_ENFORCE(c_program != NULL, "program is null");
  //
  DLOG << "n_ops: " << (*c_program->blocks)->n_ops;
  //
  auto originProgramDesc = std::make_shared<framework::ProgramDesc>(c_program);

  framework::Program<Dtype, P> program;
  program.originProgram = originProgramDesc;

  auto scope = std::make_shared<framework::Scope>();
  program.scope = scope;

  for (const auto &block : originProgramDesc->Blocks()) {
    for (auto var_desc : block->Vars()) {
      auto var = scope->Var(var_desc->Name());

      if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
        if (var_desc->Persistable() &&
            var_desc->Type() != framework::VARTYPE_TYPE_FEED_MINIBATCH &&
            var_desc->Type() != framework::VARTYPE_TYPE_FETCH_LIST) {
          auto dim = var_desc->Tensor_desc().Dims();
          auto tensor = var->GetMutable<framework::LoDTensor>();
          tensor->Resize(framework::make_ddim(dim));
        } else {
          auto dim = var_desc->Tensor_desc().Dims();
          PADDLE_MOBILE_ENFORCE(dim.size() > 0, "dim size is 0");
          dim[0] = 1;
          auto tensor = var->GetMutable<framework::LoDTensor>();
          tensor->Resize(framework::make_ddim(dim));
        }
      } else {
        // TODO(codeWorm): some.
      }
    }
  }

  //  originProgramDesc->Description("program: ");

  if (optimize) {
    framework::ProgramOptimize program_optimize;
    program.optimizeProgram =
        program_optimize.FushionOptimize(originProgramDesc);
  }
  if (optimize) {
    program.optimizeProgram->Description("optimize: ");
  } else {
    originProgramDesc->Description("program: ");
  }

  paddle_mobile__framework__proto__program_desc__free_unpacked(c_program, NULL);
  return program;
}

template class Loader<CPU, Precision::FP32>;

#pragma mark - executor

template <typename Dtype, Precision P>
Executor<Dtype, P>::Executor(const framework::Program<Dtype> p, int batch_size,
                             bool use_optimize)
    : program_(p), batch_size_(batch_size), use_optimize_(use_optimize) {
  if (use_optimize_) {
    to_predict_program_ = program_.optimizeProgram;
  } else {
    to_predict_program_ = program_.originProgram;
  }
  Variable *variable_ptr = program_.scope->Var("batch_size");
  variable_ptr[0].SetValue<int>(batch_size);
  const std::vector<std::shared_ptr<framework::BlockDesc>> blocks =
      to_predict_program_->Blocks();
  for (int i = 0; i < blocks.size(); ++i) {
    std::shared_ptr<framework::BlockDesc> block_desc = blocks[i];
    std::vector<std::shared_ptr<framework::OpDesc>> ops = block_desc->Ops();
    for (int j = 0; j < ops.size(); ++j) {
      std::shared_ptr<framework::OpDesc> op = ops[j];
      DLOG << "create op: " << op->Type();
      auto op_base = framework::OpRegistry<Dtype>::CreateOp(
          op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap(),
          program_.scope);
      op_base->InferShape();

      ops_of_block_[*block_desc.get()].push_back(op_base);
    }
  }
  if (program_.is_commbine) {
    InitCombineMemory();
  } else {
    InitMemory();
  }
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::LoadMemory(const framework::VarDesc var_desc,
                                    framework::LoDTensor *tensor, char *&data) {
  // 1. version
  uint32_t version = *(uint32_t *)data;
  data += sizeof(uint32_t);

  // 2 Lod information
  uint64_t lod_level = *(uint64_t *)data;
  data += sizeof(uint64_t);

  auto &lod = *tensor->mutable_lod();
  lod.resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size = *(uint64_t *)data;
    data += sizeof(uint64_t);
    DLOG << "lod size: " << i << size;
    std::vector<size_t> tmp(size / sizeof(size_t));

    for (int k = 0; k < tmp.size(); ++k) {
      tmp[k] = *(size_t *)data;
      DLOG << "tmp[k]: " << k << *(size_t *)data;
      data += sizeof(size_t);
    }

    for (auto j : tmp) {
      LOG(kLOG_DEBUG1) << "    lod - " << j;
    }
    lod[i] = tmp;
  }

  // 3. tensor version
  uint32_t tensor_version = *(uint32_t *)data;
  data += sizeof(uint32_t);

  // 4. tensor desc
  int32_t size = *(int32_t *)data;
  data += sizeof(int32_t);

  std::unique_ptr<char[]> buf(new char[size]);
  for (int m = 0; m < size; ++m) {
    buf.get()[m] = data[m];
  }
  data += (sizeof(char) * size);

  const framework::TensorDesc &desc = var_desc.Tensor_desc();
  int memory_size = 1;
  for (auto l : desc.Dims()) {
    memory_size *= l;
  }

  tensor->Resize(framework::make_ddim(desc.Dims()));

  void *memory = tensor;
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

  for (int n = 0; n < memory_size * type_size; ++n) {
    static_cast<char *>(memory)[n] = data[n];
  }
  data += (sizeof(char) * memory_size * type_size);
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
        LoadMemory(*var_desc, tensor, data);
        delete origin_data;
      } else {
        if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
          auto tensor = var->template GetMutable<framework::LoDTensor>();

          tensor->template mutable_data<Ptype>();
        }
      }
    }
  }
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::InitCombineMemory() {
  char *origin_data = Get_binary_data(program_.para_path);
  char *data = origin_data;
  for (const auto &block : to_predict_program_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        auto tensor = var->template GetMutable<framework::LoDTensor>();
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          continue;
        }
        LoadMemory(*var_desc, tensor, data);
      } else {
        if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
          auto tensor = var->template GetMutable<framework::LoDTensor>();
          tensor->template mutable_data<Ptype>();
        }
      }
    }
  }
  delete origin_data;
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
#ifdef PADDLE_MOBILE_PROFILE
  std::map<std::string, clock_t> _profile;
#endif
  for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
    auto op = ops_of_block_[*to_predict_block.get()][j];
#ifdef PADDLE_MOBILE_PROFILE
  _profile[op->Type()] = clock();
#endif
    op->Run();
#ifdef PADDLE_MOBILE_PROFILE
  _profile[op->Type()] = clock() - _profile[op->Type()];
#endif
  }
#ifdef PADDLE_MOBILE_PROFILE
  {
    DLOG << "========================[ profile ]==========================";
    clock_t _ptotal = 0;
    for (auto const & p : _profile) {
      _ptotal += p.second;
    }
    for (auto const & p : _profile) {
      DLOG << p.first << std::string(16-p.first.size(), ' ')
           << "\t" << (float)p.second
           << "\t\t" << (float)p.second / (float)_ptotal * 100.0;
    }
    DLOG << "========================[         ]==========================";
  }
#endif
  auto ops = ops_of_block_[*to_predict_program_->Block(0)];
  auto last_op = ops.rbegin();
  auto output_map = (*last_op)->Outputs();
  std::vector<std::string> out_keys = (*last_op)->GetOutKeys();
  PADDLE_MOBILE_ENFORCE(out_keys.size() > 0, "the last op contains no output");
  framework::LoDTensor *output_tensor =
      framework::GetVarValue<framework::LoDTensor>(out_keys[0], output_map,
                                                   *(program_.scope));
  return std::shared_ptr<framework::Tensor>(output_tensor);
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

}  // namespace paddle_mobile
