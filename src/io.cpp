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
#include "framework/framework.pb.h"
#include "framework/lod_tensor.h"
#include "framework/program/program_desc.h"
#include "framework/scope.h"
#include "framework/tensor.h"

namespace paddle_mobile {

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

template <typename Dtype, Precision P>
void Loader<Dtype, P>::LoadVar(framework::LoDTensor *tensor,
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

  framework::proto::VarType::TensorDesc desc;
  desc.ParseFromArray(buf.get(), size);

  int memory_size = 1;
  for (auto l : desc.dims()) {
    memory_size *= l;
  }

  std::vector<int64_t> dims;
  dims.reserve(static_cast<size_t>(desc.dims().size()));
  std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
  tensor->Resize(framework::make_ddim(dims));

  void *memory = tensor;
  int type_size = 0;
  switch (desc.data_type()) {
    case framework::proto::VarType::FP16:
      type_size = 2;
      break;
    case framework::proto::VarType::FP32:
      type_size = 4;
      memory = tensor->mutable_data<float>();
      break;
    case framework::proto::VarType::FP64:
      type_size = 8;
      break;
    case framework::proto::VarType::INT32:
      type_size = 4;
      break;
    case framework::proto::VarType::INT64:
      type_size = 8;
      break;
    case framework::proto::VarType::BOOL:
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
  std::string program_desc_str;
  ReadBinaryFile(model_filename, &program_desc_str);
  framework::proto::ProgramDesc program_desc_proto;
  program_desc_proto.ParseFromString(program_desc_str);

  std::shared_ptr<framework::ProgramDesc> originProgramDesc =
      std::make_shared<framework::ProgramDesc>(program_desc_proto);

  framework::Program<Dtype, P> program;
  program.originProgram = originProgramDesc;

  std::shared_ptr<framework::Scope> scope =
      std::make_shared<framework::Scope>();
  program.scope = scope;

  originProgramDesc->Block(0);

  for (const auto &block : originProgramDesc->Blocks()) {
    for (int i = 0; i < block->Vars().size(); ++i) {
      std::shared_ptr<framework::VarDesc> var_desc = block->Vars()[i];
      auto var = scope->Var(var_desc->Name());
      if (var_desc->GetType() == framework::proto::VarType::LOD_TENSOR) {
        if (var_desc->Persistable() &&
            var_desc->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
            var_desc->GetType() != framework::proto::VarType::FETCH_LIST) {
          auto tensor = var->GetMutable<framework::LoDTensor>();
          // to load
          LoadVar(tensor, dirname + "/" + var_desc->Name());
        }
      } else {
        // TODO(codeWorm): some.
      }
    }
  }

#ifdef PADDLE_MOBILE_DEBUG
  for (const auto &block : program_desc_proto.blocks()) {
    LOG(kLOG_DEBUG) << "block: " << block.idx();
    for (int j = 0; j < block.ops().size(); ++j) {
      //      if (j == 2) {
      //        break;
      //      }
      framework::proto::OpDesc op = block.ops()[j];
      LOG(kLOG_DEBUG1) << "op: " << op.type();
      for (int m = 0; m < op.inputs_size(); ++m) {
        const framework::proto::OpDesc::Var &var = op.inputs(m);
        LOG(kLOG_DEBUG2) << "input parameter: " << var.parameter();
        for (const auto &n : var.arguments()) {
          LOG(kLOG_DEBUG3) << "argument - " << n;
        }
      }

      for (int y = 0; y < op.outputs_size(); ++y) {
        const framework::proto::OpDesc::Var &var = op.outputs(y);
        LOG(kLOG_DEBUG2) << "out parameter: " << var.parameter();
        for (const auto &z : var.arguments()) {
          LOG(kLOG_DEBUG3) << "argument - " << z;
        }
      }

      for (const auto &attr : op.attrs()) {
        LOG(kLOG_DEBUG2) << "attr name: " << attr.name();

        switch (attr.type()) {
          case framework::proto::AttrType::BOOLEAN:
            LOG(kLOG_DEBUG3) << "boolen: " << attr.b();
            break;
          case framework::proto::AttrType::INT:
            LOG(kLOG_DEBUG3) << "int: " << attr.i();
            break;
          case framework::proto::AttrType::FLOAT:
            LOG(kLOG_DEBUG3) << "float: " << attr.f();
          case framework::proto::AttrType::STRING:
            LOG(kLOG_DEBUG3) << "string: " << attr.s();
          case framework::proto::AttrType::BOOLEANS:
            for (int y = 0; y < attr.bools_size(); ++y) {
              LOG(kLOG_DEBUG3) << "bools: " << attr.bools(y);
            }
          case framework::proto::AttrType::LONG:
            LOG(kLOG_DEBUG3) << "long: " << attr.l();
          case framework::proto::AttrType::FLOATS:
            for (int y = 0; y < attr.floats_size(); ++y) {
              LOG(kLOG_DEBUG3) << "floats: " << attr.floats(y);
            }
          case framework::proto::AttrType::INTS:
            for (int y = 0; y < attr.ints_size(); ++y) {
              LOG(kLOG_DEBUG3) << "ints: " << attr.ints(y);
            }
          case framework::proto::AttrType::STRINGS:
            for (int y = 0; y < attr.strings_size(); ++y) {
              LOG(kLOG_DEBUG3) << "strings: " << attr.strings(y);
            }
          case framework::proto::BLOCK:
            break;
        }
      }
    }

    for (const auto &var : block.vars()) {
      if (var.type().type() == framework::proto::VarType::LOD_TENSOR) {
        LOG(kLOG_DEBUG1) << "var name: " << var.name();
        const framework::proto::VarType::TensorDesc &tensor_desc =
            var.type().lod_tensor().tensor();
        LOG(kLOG_DEBUG2) << "in var tensor desc dims size: "
                         << tensor_desc.dims().size();
        for (int l = 0; l < tensor_desc.dims().size(); ++l) {
          LOG(kLOG_DEBUG3) << "var tensor desc dim " << l
                           << " value: " << tensor_desc.dims()[l];
        }
      }

      if (var.persistable() &&
          var.type().type() != framework::proto::VarType::FEED_MINIBATCH &&
          var.type().type() != framework::proto::VarType::FETCH_LIST) {
        std::string file_path = dirname + "/" + var.name();
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
        for (uint64_t i = 0; i < lod_level; ++i) {
          uint64_t size;
          is.read(reinterpret_cast<char *>(&size), sizeof(size));
          std::vector<size_t> tmp(size / sizeof(size_t));
          is.read(reinterpret_cast<char *>(tmp.data()),
                  static_cast<std::streamsize>(size));
          for (int j = 0; j < tmp.size(); ++j) {
          }
        }

        is.read(reinterpret_cast<char *>(&version), sizeof(version));

        int32_t size;
        is.read(reinterpret_cast<char *>(&size), sizeof(size));
        std::unique_ptr<char[]> buf(new char[size]);
        is.read(reinterpret_cast<char *>(buf.get()), size);

        framework::proto::VarType::TensorDesc desc;
        desc.ParseFromArray(buf.get(), size);

        int memory_size = 1;
        for (long long l : desc.dims()) {
          memory_size *= l;
        }

        int type_size = 0;
        switch (desc.data_type()) {
          case framework::proto::VarType::FP16:
            type_size = 2;
            break;
          case framework::proto::VarType::FP32:
            type_size = 4;
            break;
          case framework::proto::VarType::FP64:
            type_size = 8;
            break;
          case framework::proto::VarType::INT32:
            type_size = 4;
            break;
          case framework::proto::VarType::INT64:
            type_size = 8;
            break;
          case framework::proto::VarType::BOOL:
            type_size = 1;
            break;
          default:
            break;
        }

        void *memory = malloc(memory_size * type_size);
        is.read(static_cast<char *>(memory), memory_size * type_size);
        is.close();
      } else {
        // TODO
      }
    }
  }

#endif
  return program;
}

template class Loader<CPU, Precision::FP32>;

}  // namespace paddle_mobile
