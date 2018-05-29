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
#include <fstream>
#include <vector>

#include "common/enforce.h"
#include "common/log.h"
#include "framework/framework.pb-c.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/program_desc.h"
#include "framework/program/var_desc.h"
#include "framework/scope.h"
#include "framework/tensor.h"

namespace paddle_mobile {
using framework::Variable;

void ReadBinaryFile(const std::string &filename, std::string *contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_MOBILE_ENFORCE(fin.is_open(), "open file: %s failed",
                        filename.c_str());
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
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

  *out = (uint8_t *)malloc(size);

  size_t cur_len = 0;
  size_t nread;
  while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
    cur_len += nread;
  }
  fclose(fp);
  return cur_len;
}

template <typename Dtype, Precision P>
void Loader<Dtype, P>::LoadVar(framework::Variable *variable,
                               const framework::VarDesc &var_desc,
                               const std::string &file_path) {
  auto tensor = variable->GetMutable<framework::LoDTensor>();
  std::ifstream is(file_path);
  PADDLE_MOBILE_ENFORCE(is.is_open(), "open file: %s failed",
                        file_path.c_str());

  std::fpos<mbstate_t> pos;
  pos = is.tellg();  // save   current   position
  is.seekg(0, std::ios::end);
  is.seekg(pos);  // restore   saved   position

  // 1. version
  uint32_t version;
  is.read(reinterpret_cast<char *>(&version), sizeof(version));

  // 2 Lod information
  uint64_t lod_level;
  is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
  auto &lod = *tensor->mutable_lod();
  lod.resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::vector<size_t> tmp(size / sizeof(size_t));
    is.read(reinterpret_cast<char *>(tmp.data()),
            static_cast<std::streamsize>(size));
    for (auto j : tmp) {
      LOG(kLOG_DEBUG1) << "    lod - " << j;
    }
    lod[i] = tmp;
  }

  // 3. tensor version
  uint32_t tensor_version;
  is.read(reinterpret_cast<char *>(&tensor_version), sizeof(tensor_version));

  // 4. tensor desc
  int32_t size;
  is.read(reinterpret_cast<char *>(&size), sizeof(size));
  std::unique_ptr<char[]> buf(new char[size]);
  is.read(reinterpret_cast<char *>(buf.get()), size);

  const framework::TensorDesc &desc = var_desc.Tensor_desc();

  PaddleMobile__Framework__Proto__VarType__TensorDesc *tensor_desc = NULL;
  //  void *v;
  //  PaddleMobile__Framework__Proto__VarType__TensorDesc_Closure()(tensor_desc,
  //  buf.get());

  //  DLOG << "PaddleMobile__Framework__Proto__VarType__TensorDesc_Closure- " <<
  //  tensor_desc;

  //  framework::TensorDesc &tensor_desc = variable->
  //  PaddleMobile__Framework__Proto__ProgramDesc *c_program;
  //  uint8_t *proto_buf = NULL;
  //  size_t read_size = ReadBuffer(file_path.c_str(), &proto_buf);
  //  c_program = paddle_mobile__framework__proto__program_desc__unpack(NULL,
  //  read_size, buf);

  //  paddle_mobile__framework__proto__var_type__tensor_desc__init()

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

  is.read(static_cast<char *>(memory), memory_size * type_size);
  is.close();
}

template <typename Dtype, Precision P>
const framework::Program<Dtype, P> Loader<Dtype, P>::Load(
    const std::string &dirname) {
  std::string model_filename = dirname + "/__model__";
  PaddleMobile__Framework__Proto__ProgramDesc *c_program;
  uint8_t *buf = NULL;
  size_t read_size = ReadBuffer(model_filename.c_str(), &buf);

  PADDLE_MOBILE_ENFORCE(buf != NULL, "read from __model__ is null");

  c_program = paddle_mobile__framework__proto__program_desc__unpack(
      NULL, read_size, buf);

  PADDLE_MOBILE_ENFORCE(c_program != NULL, "program is null");

  DLOG << "n_ops: " << (*c_program->blocks)->n_ops;

  std::shared_ptr<framework::ProgramDesc> originProgramDesc =
      std::make_shared<framework::ProgramDesc>(c_program);

  framework::Program<Dtype, P> program;
  program.model_path = dirname;
  program.originProgram = originProgramDesc;

  std::shared_ptr<framework::Scope> scope =
      std::make_shared<framework::Scope>();
  program.scope = scope;
  originProgramDesc->Block(0);

  for (const auto &block : originProgramDesc->Blocks()) {
    for (int i = 0; i < block->Vars().size(); ++i) {
      std::shared_ptr<framework::VarDesc> var_desc = block->Vars()[i];
      //      DLOG << "var name-- " << var_desc->Name();
      auto var = scope->Var(var_desc->Name());

      if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
        if (var_desc->Persistable() &&
            var_desc->Type() != framework::VARTYPE_TYPE_FEED_MINIBATCH &&
            var_desc->Type() != framework::VARTYPE_TYPE_FETCH_LIST) {
          //          DLOG << "to load var ";
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

  paddle_mobile__framework__proto__program_desc__free_unpacked(c_program, NULL);
  return program;
}

template class Loader<CPU, Precision::FP32>;

#pragma mark - executor

template <typename Dtype, Precision P>
Executor<Dtype, P>::Executor(const framework::Program<Dtype> p) : program_(p) {
  if (use_optimize_) {
    to_predict_program_ = program_.optimizeProgram;
  } else {
    to_predict_program_ = program_.originProgram;
  }

  const std::vector<std::shared_ptr<framework::BlockDesc>> blocks =
      to_predict_program_->Blocks();
  for (int i = 0; i < blocks.size(); ++i) {
    std::shared_ptr<framework::BlockDesc> block_desc = blocks[i];
    std::vector<std::shared_ptr<framework::OpDesc>> ops = block_desc->Ops();
    for (int j = 0; j < ops.size(); ++j) {
      std::shared_ptr<framework::OpDesc> op = ops[j];
      auto op_base = framework::OpRegistry<Dtype>::CreateOp(
          op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap(),
          program_.scope);
      op_base->InferShape();
      ops_of_block_[*block_desc.get()].push_back(op_base);
    }
  }
  InitMemory();
}

template <typename Dtype, Precision P>
Executor<Dtype, P>::Executor(const framework::Program<Dtype> p, int batch_size)
    : program_(p), batch_size_(batch_size) {
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
      auto op_base = framework::OpRegistry<Dtype>::CreateOp(
          op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap(),
          program_.scope);
      op_base->InferShape();

      ops_of_block_[*block_desc.get()].push_back(op_base);
    }
  }
  InitMemory();
}

template <typename Dtype, Precision P>
void Executor<Dtype, P>::LoadMemory(const framework::VarDesc var_desc,
                                    framework::LoDTensor *tensor,
                                    const std::string &file_path) {
  std::ifstream is(file_path);
  PADDLE_MOBILE_ENFORCE(is.is_open(), "open file: %s failed",
                        file_path.c_str());
  std::fpos<mbstate_t> pos;
  pos = is.tellg();  // save   current   position
  is.seekg(0, std::ios::end);
  is.seekg(pos);  // restore   saved   position

  // 1. version
  uint32_t version;
  is.read(reinterpret_cast<char *>(&version), sizeof(version));

  // 2 Lod information
  uint64_t lod_level;
  is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
  auto &lod = *tensor->mutable_lod();
  lod.resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::vector<size_t> tmp(size / sizeof(size_t));
    is.read(reinterpret_cast<char *>(tmp.data()),
            static_cast<std::streamsize>(size));
    for (auto j : tmp) {
      LOG(kLOG_DEBUG1) << "    lod - " << j;
    }
    lod[i] = tmp;
  }

  // 3. tensor version
  uint32_t tensor_version;
  is.read(reinterpret_cast<char *>(&tensor_version), sizeof(tensor_version));

  // 4. tensor desc
  int32_t size;
  is.read(reinterpret_cast<char *>(&size), sizeof(size));
  std::unique_ptr<char[]> buf(new char[size]);
  is.read(reinterpret_cast<char *>(buf.get()), size);

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

  is.read(static_cast<char *>(memory), memory_size * type_size);
  is.close();
};

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
        LoadMemory(*var_desc, tensor,
                   program_.model_path + "/" + var_desc->Name());
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
void Executor<Dtype, P>::predict(const framework::Tensor &t, int block_id) {
  framework::Variable *g_feed_value = program_.scope->Var("feed");
  auto feed_tensor = g_feed_value->GetMutable<framework::LoDTensor>();
  feed_tensor->Resize(t.dims());
  feed_tensor->ShareDataWith(t);
  std::shared_ptr<framework::BlockDesc> to_predict_block =
      to_predict_program_->Block(block_id);
  for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
    auto op = ops_of_block_[*to_predict_block.get()][j];
      op->Run();
  }
}

template <typename Dtype, Precision P>
std::vector<typename Executor<Dtype, P>::Ptype> Executor<Dtype, P>::predict(
    const std::vector<Ptype> &input, const std::vector<int64_t> &dims) {
  DLOG << "start predict: ";

  framework::Tensor tensor;
  auto ddim = framework::make_ddim(dims);

  auto input_ptr = tensor.mutable_data<Ptype>(ddim);
  for (int i = 0; i < input.size(); ++i) {
    input_ptr[i] = input[i];
  }

  predict(tensor, 0);

  framework::Variable *g_feed_value = program_.scope->Var("col");
  auto feed_tensor = g_feed_value->GetMutable<framework::Tensor>();

  return {};
}

template class Executor<CPU, Precision::FP32>;

}  // namespace paddle_mobile
